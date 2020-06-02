"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from ccai.nn.model.networks import (
    SpadeGen,
    MsImageDis,
    VAEGen,
    MultiscaleDiscriminator,
    VGGLoss,
    FullSpadeGen,
    TVLoss,
)
from ccai.nn.model.utils import (
    weights_init,
    get_model_list,
    vgg_preprocess,
    load_vgg16,
    get_scheduler,
    load_flood_classifier,
    transform_torchVar,
    seg_batch_transform,
    seg_transform,
    load_segmentation_model,
    decode_segmap,
    domainClassifier,
    merge_classes,
    normalize_batch,
)
from torch.autograd import Variable
from torchvision import transforms
import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision.utils import save_image


class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters["lr"]
        self.newsize = hyperparameters["crop_image_height"]
        self.semantic_w = hyperparameters["semantic_w"] > 0
        self.recon_mask = hyperparameters["recon_mask"] == 1
        self.dann_scheduler = None
        self.full_adaptation = hyperparameters["adaptation"]["full_adaptation"] == 1
        self.hyperparameters = hyperparameters
        dim = hyperparameters["gen"]["dim"]
        n_downsample = hyperparameters["gen"]["n_downsample"]
        latent_dim = dim * (2 ** n_downsample)

        if "domain_adv_w" in hyperparameters.keys():
            self.domain_classif_ab = hyperparameters["domain_adv_w"] > 0
        else:
            self.domain_classif_ab = False

        if hyperparameters["adaptation"]["dfeat_lambda"] > 0:
            self.use_classifier_sr = True
        else:
            self.use_classifier_sr = False

        if hyperparameters["adaptation"]["sem_seg_lambda"] > 0:
            self.train_seg = True
        else:
            self.train_seg = False

        if hyperparameters["adaptation"]["output_classifier_lambda"] > 0:
            self.use_output_classifier_sr = True
        else:
            self.use_output_classifier_sr = False

        self.gen = FullSpadeGen(
            input_dim=hyperparameters["gen"]["dim"],
            size=hyperparameters["new_size"],
            batch_size=hyperparameters["batch_size"],
            params=hyperparameters["gen"],
        )

        self.latent_size = self.hyperparameters["new_size"] // (
            2 ** self.hyperparameters["gen"]["n_downsample"]
        )

        self.dis_b = MultiscaleDiscriminator(
            hyperparameters["input_dim_b"], hyperparameters["dis"]
        )  # discriminator for domain b

        # fix the noise usd in sampling
        display_size = int(hyperparameters["display_size"])
        # Setup the optimizers
        beta1 = hyperparameters["beta1"]
        beta2 = hyperparameters["beta2"]
        dis_params = list(self.dis_b.parameters())

        gen_params = list(self.gen.parameters())

        self.dis_opt = torch.optim.Adam(
            [p for p in dis_params if p.requires_grad],
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=hyperparameters["weight_decay"],
        )
        self.gen_opt = torch.optim.Adam(
            [p for p in gen_params if p.requires_grad],
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=hyperparameters["weight_decay"],
        )
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters["init"]))
        self.dis_b.apply(weights_init("gaussian"))

        # Load VGG model if needed
        if hyperparameters["vgg_w"] > 0:
            self.criterionVGG = VGGLoss()

        if hyperparameters["tv_w"] > 0:
            self.criterionTV = TVLoss(hyperparameters["tv_w"])

        # Load semantic segmentation model if needed
        # if "semantic_w" in hyperparameters.keys() and hyperparameters["semantic_w"] > 0:
        self.segmentation_model = load_segmentation_model(hyperparameters["semantic_ckpt_path"], 19)
        self.segmentation_model.eval()
        for param in self.segmentation_model.parameters():
            param.requires_grad = False

        # Load domain classifier if needed
        if "domain_adv_w" in hyperparameters.keys() and hyperparameters["domain_adv_w"] > 0:
            self.domain_classifier_ab = domainClassifier(input_dim=latent_dim, dim=256)
            dann_params = list(self.domain_classifier_ab.parameters())
            self.dann_opt = torch.optim.Adam(
                [p for p in dann_params if p.requires_grad],
                lr=lr,
                betas=(beta1, beta2),
                weight_decay=hyperparameters["weight_decay"],
            )
            self.domain_classifier_ab.apply(weights_init("gaussian"))
            self.dann_scheduler = get_scheduler(self.dann_opt, hyperparameters)

        # Load classifier on features for syn, real adaptation
        if self.use_classifier_sr:
            #! Hardcoded
            self.domain_classifier_sr_b = domainClassifier(input_dim=latent_dim, dim=256)
            self.domain_classifier_sr_a = domainClassifier(input_dim=latent_dim, dim=256)

            dann_params = list(self.domain_classifier_sr_a.parameters()) + list(
                self.domain_classifier_sr_b.parameters()
            )
            self.classif_opt_sr = torch.optim.Adam(
                [p for p in dann_params if p.requires_grad],
                lr=lr,
                betas=(beta1, beta2),
                weight_decay=hyperparameters["weight_decay"],
            )
            self.domain_classifier_sr_a.apply(weights_init("gaussian"))
            self.domain_classifier_sr_b.apply(weights_init("gaussian"))
            self.classif_sr_scheduler = get_scheduler(self.classif_opt_sr, hyperparameters)

        if self.use_output_classifier_sr:
            if self.hyperparameters["dis"]["type"] == "patchgan":
                self.output_classifier_sr_a = MultiscaleDiscriminator(
                    hyperparameters["input_dim_a"], hyperparameters["dis"]
                )  # discriminator for domain a,sr
                self.output_classifier_sr_b = MultiscaleDiscriminator(
                    hyperparameters["input_dim_a"], hyperparameters["dis"]
                )  # discriminator for domain b,sr

            else:
                self.output_classifier_sr_a = MsImageDis(
                    hyperparameters["input_dim_a"], hyperparameters["dis"]
                )  # discriminator for domain a,sr
                self.output_classifier_sr_b = MsImageDis(
                    hyperparameters["input_dim_a"], hyperparameters["dis"]
                )  # discriminator for domain b,sr

            dann_params = list(self.output_classifier_sr_a.parameters()) + list(
                self.output_classifier_sr_b.parameters()
            )
            self.output_classif_opt_sr = torch.optim.Adam(
                [p for p in dann_params if p.requires_grad],
                lr=lr,
                betas=(beta1, beta2),
                weight_decay=hyperparameters["weight_decay"],
            )
            self.output_classifier_sr_b.apply(weights_init("gaussian"))
            self.output_classifier_sr_a.apply(weights_init("gaussian"))
            self.output_scheduler_sr = get_scheduler(self.output_classif_opt_sr, hyperparameters)

        if self.train_seg:
            pretrained = load_segmentation_model(hyperparameters["semantic_ckpt_path"], 19)
            last_layer = nn.Conv2d(512, 10, kernel_size=1)
            model = torch.nn.Sequential(
                *list(pretrained.resnet34_8s.children())[7:-1], last_layer.cuda()
            )
            self.segmentation_head = model

            for param in self.segmentation_head.parameters():
                param.requires_grad = True

            dann_params = list(self.segmentation_head.parameters())
            self.segmentation_opt = torch.optim.Adam(
                [p for p in dann_params if p.requires_grad],
                lr=lr,
                betas=(beta1, beta2),
                weight_decay=hyperparameters["weight_decay"],
            )
            self.scheduler_seg = get_scheduler(self.segmentation_opt, hyperparameters)

    def recon_criterion(self, input, target):
        """
        Compute pixelwise L1 loss between two images input and target

        Arguments:
            input {torch.Tensor} -- Image tensor
            target {torch.Tensor} -- Image tensor

        Returns:
            torch.Float -- pixelwise L1 loss
        """
        return torch.mean(torch.abs(input - target))

    def recon_criterion_mask(self, input, target, mask):
        """
        Compute a weaker version of the recon_criterion between two images input and target
        where the L1 is only computed on the unmasked region

        Arguments:
            input {torch.Tensor} -- Image (original image such as x_a)
            target {torch.Tensor} -- Image (after cycle-translation image x_aba)
            mask {} -- binary Mask of size HxW (input.shape ~ CxHxW)

        Returns:
            torch.Float -- L1 loss over input.(1-mask) and target.(1-mask)
        """
        return torch.mean(torch.abs(torch.mul((input - target), 1 - mask)))

    def forward(self, x_a, x_b, m_a, m_b):
        """
        Perform the translation from domain A (resp B) to domain B (resp A): x_a to x_ab (resp: x_b to x_ba).

        Arguments:
            x_a {torch.Tensor} -- Image from domain A after transform in tensor format
            x_b {torch.Tensor} -- Image from domain B after transform in tensor format

        Returns:
            torch.Tensor, torch.Tensor -- Translated version of x_a in domain B, Translated version of x_b in domain A
        """
        self.eval()
        # m_a_seg = self.merge_seg_with_mask(x_a, m_a)
        # m_b_seg = self.merge_seg_with_mask(x_b, m_b)

        # avg_mask_a = self.average_mask(x_a, m_a)
        # avg_mask_b = self.average_mask(x_b, m_b)

        x_a_augment = torch.cat([x_a, m_a], dim=1)
        x_a_masked = x_a * (1.0 - m_a)
        x_b_masked = x_b * (1.0 - m_b)

        z = (
            torch.empty(
                self.hyperparameters["batch_size"],
                self.hyperparameters["gen"]["dim"],
                self.latent_size,
                self.latent_size,
            )
            .normal_(mean=0, std=1.0)
            .cuda()
        )

        x_ab = self.gen(z, x_a_masked)

        self.train()
        return x_ab

    def gen_update(
        self,
        x_a,
        x_b,
        hyperparameters,
        mask_a,
        mask_b,
        comet_exp=None,
        synth=False,
        semantic_gt_a=None,
        semantic_gt_b=None,
    ):
        """
        Update the generator parameters

        Arguments:
            x_a {torch.Tensor} -- Image from domain A after transform in tensor format
            x_b {torch.Tensor} -- Image from domain B after transform in tensor format
            hyperparameters {dictionnary} -- dictionnary with all hyperparameters

        Keyword Arguments:
            mask_a {torch.Tensor} -- binary mask (0,1) corresponding to the ground in x_a (default: {None})
            mask_b {torch.Tensor} -- binary mask (0,1) corresponding to the water in x_b (default: {None})
            comet_exp {cometExperience} -- CometML object use to log all the loss and images (default: {None})
            synth {boolean}  -- binary True or False stating if we have a synthetic pair or not

        Returns:
            [type] -- [description]
        """
        self.gen_opt.zero_grad()

        x_a_augment = torch.cat([x_a, mask_a], dim=1)
        x_a_masked = x_a * (1.0 - mask_a)
        x_b_masked = x_b * (1.0 - mask_b)
        z = (
            torch.empty(
                self.hyperparameters["batch_size"],
                self.hyperparameters["gen"]["dim"],
                self.latent_size,
                self.latent_size,
            )
            .normal_(mean=0, std=1.0)
            .cuda()
        )
        # decode (within domain)
        x_ab = self.gen(z, x_a_masked)
        x_bb = self.gen(z, x_b_masked)

        # reconstruction loss
        # self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a * (1.0 - mask_a) + avg_mask_a)
        # self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b * (1.0 - mask_b) + avg_mask_b)

        # Contex preserving loss
        self.context_loss = self.recon_criterion_mask(x_bb, x_b, mask_b)

        # Synthetic reconstruction loss
        if synth:
            # print('mask_b.shape', mask_b.shape)
            # Define the mask of exact same pixel among a pair
            mask_alignment = (torch.sum(torch.abs(x_a - x_b), 1) == 0).unsqueeze(1)
            mask_alignment = mask_alignment.type(torch.cuda.FloatTensor)
            # print('mask_alignment.shape', mask_alignment.shape)

        self.loss_gen_recon_synth = (
            self.recon_criterion_mask(x_ab, x_b, 1 - mask_alignment) if synth else 0
        )

        # GAN loss
        # Concat masks before feeding to loss

        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_bb, x_b, comet_exp, mode="b")

        self.loss_gen_vgg_b = (
            self.compute_vgg_loss(x_bb, x_b) if hyperparameters["vgg_w"] > 0 else 0
        )

        # Total Variational Loss:
        self.loss_tv = self.criterionTV(x_bb * mask_b) if hyperparameters["tv_w"] > 0 else 0

        # Domain adversarial loss (c_a and c_b are swapped because we want the feature to be less informative
        # minmax (accuracy but max min loss)
        """
        self.domain_adv_loss = (
            self.compute_domain_adv_loss(c_a, c_b, compute_accuracy=False, minimize=False)
            if hyperparameters["domain_adv_w"] > 0
            else 0
        )

        self.loss_classifier_sr = (
            self.compute_classifier_sr_loss(c_a, c_b, domain_synth=synth, fool=True)
            if hyperparameters["adaptation"]["adv_lambda"] > 0
            else 0
        )
        """

        if hyperparameters["adaptation"]["output_adv_lambda"] > 0:
            self.loss_output_classifier_sr = self.output_classifier_sr_a.calc_gen_loss_sr(
                x_ba
            ) + self.output_classifier_sr_b.calc_gen_loss_sr(x_ab)

        else:

            self.loss_output_classifier_sr = 0

        # total loss
        self.loss_gen_total = (
            +hyperparameters["gan_w"] * self.loss_gen_adv_b
            + hyperparameters["context_w"] * self.context_loss
            + hyperparameters["vgg_w"] * self.loss_gen_vgg_b
            + self.loss_tv
            # + hyperparameters["domain_adv_w"] * self.domain_adv_loss
            # + hyperparameters["recon_synth_w"] * self.loss_gen_recon_synth
            # + hyperparameters["adaptation"]["adv_lambda"] * self.loss_classifier_sr
            # + hyperparameters["adaptation"]["output_adv_lambda"] * self.loss_output_classifier_sr
        )

        self.loss_gen_total.backward()
        self.gen_opt.step()

        if comet_exp is not None:

            comet_exp.log_metric("loss_gen_adv_b", self.loss_gen_adv_b.cpu().detach())

            comet_exp.log_metric("loss_gen_total", self.loss_gen_total.cpu().detach())

            if hyperparameters["vgg_w"] > 0:
                comet_exp.log_metric("loss_gen_vgg_b", self.loss_gen_vgg_b.cpu().detach())

            if hyperparameters["tv_w"] > 0:
                comet_exp.log_metric("tv_loss", self.loss_tv.cpu().detach())

            if hyperparameters["context_w"] > 0:
                comet_exp.log_metric("context_preserve_loss", self.context_loss.cpu().detach())
            if hyperparameters["domain_adv_w"] > 0:
                comet_exp.log_metric("domain_adv_loss_gen", self.domain_adv_loss.cpu().detach())
            if synth:
                comet_exp.log_metric(
                    "loss_gen_recon_synth", self.loss_gen_recon_synth.cpu().detach()
                )
            if self.use_classifier_sr:
                comet_exp.log_metric(
                    "loss_classifier_adv_sr", self.loss_classifier_sr.cpu().detach()
                )
            if self.use_output_classifier_sr:
                comet_exp.log_metric(
                    "loss_output_classifier_adv_sr", self.loss_output_classifier_sr.cpu().detach()
                )

    def compute_vgg_loss(self, img, target, mask=None, preprocess=True):
        """
        Compute the domain-invariant perceptual loss

        Arguments:
            vgg {model} -- popular Convolutional Network for Classification and Detection
            img {torch.Tensor} -- image before translation
            target {torch.Tensor} -- image after translation

        Returns:
            torch.Float -- domain invariant perceptual loss
        """
        if preprocess:
            img_vgg = vgg_preprocess(img)
            target_vgg = vgg_preprocess(target)
        else:
            img_vgg = normalize_batch(img)
            target_vgg = normalize_batch(target)

        # Mask input to VGG:
        if mask is not None:
            img_vgg = img_vgg * (1.0 - mask)
            target_vgg = target_vgg * (1.0 - mask)

        loss_G_VGG = self.criterionVGG(img_vgg, target_vgg)

        return loss_G_VGG

    def compute_classifier_sr_loss(self, c_a, c_b, domain_synth=False, fool=False):
        """
        Compute classifier loss for the adaptation s/r

        Arguments:
            c_a {torch.Tensor} -- content of x_a
            c_b {torch.Tensor} -- content of x_b
            domain_synth {Boolean} -- Whether if the content is from s or r
            fool {Boolean} -- Wheter we want to fool the classifier or not

        Returns:
            torch.Float -- domain invariant perceptual loss
        """
        # Infer domain classifier on content extracted from an image of domainA
        output_a = self.domain_classifier_sr_a(c_a)

        # Infer domain classifier on content extracted from an image of domainB
        output_b = self.domain_classifier_sr_b(c_b)

        if fool:
            loss = torch.mean((output_a - 0.5) ** 2) + torch.mean((output_b - 0.5) ** 2)

        else:
            if domain_synth:
                loss = torch.mean((output_a - 0) ** 2) + torch.mean((output_b - 0) ** 2)

            else:
                loss = torch.mean((output_a - 1) ** 2) + torch.mean((output_b - 1) ** 2)

        return loss

    def compute_domain_adv_loss(self, c_a, c_b, compute_accuracy=False, minimize=True):
        """
        Compute a domain adversarial loss on the embedding of the classifier:
        we are trying to learn an anonymized representation of the content.

        Arguments:
            c_a {torch.tensor} -- content extracted from an image of domain A with encoder A
            c_b {torch.tensor} -- content extracted from an image of domain B with encoder B

        Keyword Arguments:
            compute_accuracy {bool} -- either return only the loss or loss and softmax probs
            (default: {False})
            minimize {bool} -- optimize classification accuracy(True) or anonymized the representation(False)

        Returns:
            torch.Float -- loss (optionnal softmax P(classifier(c_a)=a) and P(classifier(c_b)=b))
        """
        # Infer domain classifier on content extracted from an image of domainA
        output_a = self.domain_classifier_ab(c_a)

        # Infer domain classifier on content extracted from an image of domainB
        output_b = self.domain_classifier_ab(c_b)

        # Concatenate the output in a single vector
        output = torch.cat((output_a, output_b))

        if minimize:
            target = torch.tensor([1.0, 0.0, 0.0, 1.0], device="cuda")
        else:
            target = torch.tensor([0.5, 0.5, 0.5, 0.5], device="cuda")
        # mean square error loss
        loss = torch.nn.MSELoss()(output, target)
        if compute_accuracy:
            return loss, output_a[0], output_b[1]
        else:
            return loss

    def compute_semantic_seg_loss(self, img1, img2, mask=None, ground_truth=None):
        """
        Compute semantic segmentation loss between two images on the unmasked region or in the entire image
        Arguments:
            img1 {torch.Tensor} -- Image from domain A after transform in tensor format
            img2 {torch.Tensor} -- Image transformed
            mask {torch.Tensor} -- Binary mask where we force the loss to be zero
            ground_truth {torch.Tensor} -- If available palletized image of size (batch, h, w)
        Returns:
            torch.float -- Cross entropy loss on the unmasked region
        """
        new_class = 19
        # denorm
        img1_denorm = (img1 + 1) / 2.0
        img2_denorm = (img2 + 1) / 2.0
        # norm for semantic seg network
        input_transformed1 = seg_batch_transform(img1_denorm)
        input_transformed2 = seg_batch_transform(img2_denorm)

        # compute labels from original image and logits from translated version
        # target = (
        #   self.segmentation_model(input_transformed1).max(1)[1]
        # )
        # Infer x_ab or x_ba
        output = self.segmentation_model(input_transformed2)

        # If we have a ground truth (simulated data), merge classes to fit the ground truth of our simulated world (19 to 10)
        if ground_truth is not None:
            target = ground_truth.type(torch.long).cuda()
            target = target.squeeze(1)
            output = merge_classes(output).cuda()
            new_class = 10

        else:
            # Else use the pretrained model
            target = self.segmentation_model(input_transformed1).max(1)[1]

        # If we don't want to compute the loss on the masked region
        if not self.full_adaptation and mask is not None:
            # Resize mask to the size of the image
            # ADRIEN  DANGEROUS TO CHAAAANGE
            mask1 = torch.nn.functional.interpolate(mask, size=(self.newsize, self.newsize))

            mask1_tensor = torch.tensor(mask1, dtype=torch.long).cuda()
            mask1_tensor = mask1_tensor.squeeze(1)

            # we want the masked region to be labeled as unknown (19 is not an existing label)
            target_with_mask = (
                torch.mul(1 - mask1_tensor, target) + mask1_tensor * new_class
            )  # CATEGORICAL TENSOR (B 20 H W) (TARGET)

            mask2 = torch.nn.functional.interpolate(mask, size=(self.newsize, self.newsize))
            mask_tensor = torch.tensor(mask2, dtype=torch.float).cuda()
            output_with_mask = torch.mul(1 - mask_tensor, output)
            #
            # cat the mask as to the logits (loss=0 over the masked region)
            output_with_mask_cat = torch.cat((output_with_mask, mask_tensor), dim=1)
            loss = nn.CrossEntropyLoss()(output_with_mask_cat, target_with_mask)

        else:
            loss = nn.CrossEntropyLoss()(output, target)
        return loss

    def merge_seg_with_mask(self, img, mask):
        """
        Compute semantic segmentation loss between two images on the unmasked region or in the entire image
        Arguments:
            img1 {torch.Tensor} -- Image from domain A after transform in tensor format
            img2 {torch.Tensor} -- Image transformed
            mask {torch.Tensor} -- Binary mask where we force the loss to be zero
            ground_truth {torch.Tensor} -- If available palletized image of size (batch, h, w)
        Returns:
            torch.float -- Cross entropy loss on the unmasked region
        """

        # denorm
        img_denorm = (img + 1) / 2.0

        # norm for semantic seg network
        input_transformed = seg_batch_transform(img_denorm)

        # compute labels from original image and logits from translated version
        # target = (
        #   self.segmentation_model(input_transformed1).max(1)[1]
        # )
        # Infer x_ab or x_ba
        output = self.segmentation_model(input_transformed)
        max_value = output.size()[1]
        max_value = mask_value = max_value + 1  # make masked value the largest class
        output = output.argmax(1).unsqueeze(1)

        # Zero out masked values:
        output = output * (1 - mask) + (mask * mask_value)
        output_mask = output.to(torch.float) / max_value

        return output_mask

    def average_mask(self, img, mask):
        """
        # Apply mask:
        mask_region = img * mask

        # Get average
        avg = torch.mean(mask_region, dim=(2, 3))
        avg = avg.unsqueeze(-1).unsqueeze(-1)

        masked_avg = mask * avg
        """

        # Return random noise
        masked_avg = torch.normal(mean=0, std=1, size=mask.size(), device="cuda")
        masked_avg = masked_avg * mask
        return masked_avg

    def sample(self, x_a, x_b, m_a, m_b):
        """
        Infer the model on a batch of image

        Arguments:
            x_a {torch.Tensor} -- batch of image from domain A
            x_b {[type]} -- batch of image from domain B

        Returns:
            A list of torch images -- columnwise :x_a, autoencode(x_a), x_ab_1, x_ab_2
            Or if self.semantic_w is true: x_a, autoencode(x_a), Semantic segmentation x_a,
            x_ab_1,semantic segmentation x_ab_1, x_ab_2
        """

        self.eval()
        # m_a_seg = self.merge_seg_with_mask(x_a, m_a)
        # m_b_seg = self.merge_seg_with_mask(x_b, m_b)

        x_a_recon, x_b_recon, x_bb, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []

        x_a_augment = torch.cat([x_a, m_a], dim=1)
        x_b_augment = torch.cat([x_b, m_b], dim=1)
        x_a_masked = x_a * (1.0 - m_a)
        x_b_masked = x_b * (1.0 - m_b)

        for i in range(x_a.size(0)):
            z = (
                torch.empty(
                    1, self.hyperparameters["gen"]["dim"], self.latent_size, self.latent_size
                )
                .normal_(mean=0, std=1.0)
                .cuda()
            )
            x_ab1.append(self.gen(z, x_a_masked[i].unsqueeze(0)))  # s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen(z, x_a_masked[i].unsqueeze(0)))  # s_b2[i].unsqueeze(0)))
            x_bb.append(self.gen(z, x_b_masked[i].unsqueeze(0)))  # s_b2[i].unsqueeze(0)))

        x_ab1, x_bb = torch.cat(x_ab1), torch.cat(x_bb)

        self.train()
        # Overlay mask onto image:
        save_m_a = x_a - (x_a * m_a.repeat(1, 3, 1, 1)) + m_a.repeat(1, 3, 1, 1)

        return (x_a, x_ab1, x_ab1 * m_a, save_m_a, x_b, x_bb, x_bb * m_b, save_m_a)

    def sample_syn(self, x_a, x_b, m_a, m_b):
        """
        Infer the model on a batch of image

        Arguments:
            x_a {torch.Tensor} -- batch of image from domain A
            x_b {[type]} -- batch of image from domain B

        Returns:
            A list of torch images -- columnwise :x_a, autoencode(x_a), x_ab_1, x_ab_2
            Or if self.semantic_w is true: x_a, autoencode(x_a), Semantic segmentation x_a,
            x_ab_1,semantic segmentation x_ab_1, x_ab_2
        """
        self.eval()

        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        x_a_augment = torch.cat([x_a, m_a], dim=1)
        x_b_augment = torch.cat([x_b, m_b], dim=1)

        for i in range(x_a.size(0)):
            c_a = self.gen.encode(x_a[i].unsqueeze(0), 1)
            c_b = self.gen.encode(x_b[i].unsqueeze(0), 2)
            x_a_recon.append(self.gen.decode(c_a, m_a[i].unsqueeze(0), 1))
            x_b_recon.append(self.gen.decode(c_b, m_b[i].unsqueeze(0), 2))

            x_ba1.append(self.gen.decode(c_b, m_b[i].unsqueeze(0), 1))
            x_ba2.append(self.gen.decode(c_b, m_b[i].unsqueeze(0), 1))
            x_ab1.append(self.gen.decode(c_a, m_a[i].unsqueeze(0), 2))
            x_ab2.append(self.gen.decode(c_a, m_a[i].unsqueeze(0), 2))

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)

        if self.semantic_w:
            rgb_a_list, rgb_b_list, rgb_ab_list, rgb_ba_list = [], [], [], []

            for i in range(x_a.size(0)):

                # Inference semantic segmentation on original images
                im_a = (x_a[i].squeeze() + 1) / 2.0
                im_b = (x_b[i].squeeze() + 1) / 2.0

                input_transformed_a = seg_transform()(im_a).unsqueeze(0)
                input_transformed_b = seg_transform()(im_b).unsqueeze(0)
                output_a = self.segmentation_model(input_transformed_a).squeeze().max(0)[1]
                output_b = self.segmentation_model(input_transformed_b).squeeze().max(0)[1]

                rgb_a = decode_segmap(output_a.cpu().numpy())
                rgb_b = decode_segmap(output_b.cpu().numpy())
                rgb_a = Image.fromarray(rgb_a).resize((x_a.size(3), x_a.size(3)))
                rgb_b = Image.fromarray(rgb_b).resize((x_a.size(3), x_a.size(3)))

                rgb_a_list.append(transforms.ToTensor()(rgb_a).unsqueeze(0))
                rgb_b_list.append(transforms.ToTensor()(rgb_b).unsqueeze(0))

                # Inference semantic segmentation on fake images
                image_ab = (x_ab1[i].squeeze() + 1) / 2.0
                image_ba = (x_ba1[i].squeeze() + 1) / 2.0

                input_transformed_ab = seg_transform()(image_ab).unsqueeze(0).to("cuda")
                input_transformed_ba = seg_transform()(image_ba).unsqueeze(0).to("cuda")

                output_ab = self.segmentation_model(input_transformed_ab).squeeze().max(0)[1]
                output_ba = self.segmentation_model(input_transformed_ba).squeeze().max(0)[1]

                rgb_ab = decode_segmap(output_ab.cpu().numpy())
                rgb_ba = decode_segmap(output_ba.cpu().numpy())

                rgb_ab = Image.fromarray(rgb_ab).resize((x_a.size(3), x_a.size(3)))
                rgb_ba = Image.fromarray(rgb_ba).resize((x_a.size(3), x_a.size(3)))

                rgb_ab_list.append(transforms.ToTensor()(rgb_ab).unsqueeze(0))
                rgb_ba_list.append(transforms.ToTensor()(rgb_ba).unsqueeze(0))

            rgb1_a, rgb1_b, rgb1_ab, rgb1_ba = (
                torch.cat(rgb_a_list).cuda(),
                torch.cat(rgb_b_list).cuda(),
                torch.cat(rgb_ab_list).cuda(),
                torch.cat(rgb_ba_list).cuda(),
            )

        self.train()
        if self.semantic_w:
            self.segmentation_model.eval()
            return (
                x_a,
                x_a_recon,
                rgb1_a,
                x_ab1,
                rgb1_ab,
                x_ab2,
                x_b,
                x_b_recon,
                rgb1_b,
                x_ba1,
                rgb1_ba,
                x_ba2,
            )
        else:
            return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def dis_update(self, x_a, x_b, m_a, m_b, hyperparameters, comet_exp=None):
        """
        Update the weights of the discriminator

        Arguments:
            x_a {torch.Tensor} -- Image from domain A after transform in tensor format
            x_b {torch.Tensor} -- Image from domain B after transform in tensor format
            hyperparameters {dictionnary} -- dictionnary with all hyperparameters

        Keyword Arguments:
            comet_exp {cometExperience} -- CometML object use to log all the loss and images (default: {None})
        """

        self.dis_opt.zero_grad()
        z = (
            torch.empty(
                self.hyperparameters["batch_size"],
                self.hyperparameters["gen"]["dim"],
                self.latent_size,
                self.latent_size,
            )
            .normal_(mean=0, std=1.0)
            .cuda()
        )

        x_a_augment = torch.cat([x_a, m_a], dim=1)
        x_a_masked = x_a * (1.0 - m_a)
        x_b_masked = x_b * (1.0 - m_b)

        # Replace masked region with average
        x_bb = self.gen(z, x_b_masked)

        self.loss_dis_b = self.dis_b.calc_dis_loss(x_bb.detach(), x_b, comet_exp, mode="b")

        self.loss_dis_total = hyperparameters["gan_w"] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

        if comet_exp is not None:
            # comet_exp.log_metric("loss_dis_a", self.loss_dis_a.cpu().detach())
            comet_exp.log_metric("loss_dis_b", self.loss_dis_b.cpu().detach())

    def domain_classifier_update(self, x_a, x_b, hyperparameters, comet_exp=None):
        """
        Update the weights of the domain classifier

        Arguments:
            x_a {torch.Tensor} -- Image from domain A after transform in tensor format
            x_b {torch.Tensor} -- Image from domain B after transform in tensor format
            hyperparameters {dictionnary} -- dictionnary with all hyperparameters

        Keyword Arguments:
            comet_exp {cometExperience} -- CometML object use to log all the loss and images (default: {None})
        """
        self.dann_opt.zero_grad()

        # encode
        c_a = self.gen.encode(x_a, 1)
        c_b = self.gen.encode(x_b, 2)

        # domain classifier loss
        self.domain_class_loss, out_a, out_b = self.compute_domain_adv_loss(
            c_a, c_b, compute_accuracy=True, minimize=True
        )

        self.domain_class_loss.backward()
        self.dann_opt.step()

        if comet_exp is not None:
            comet_exp.log_metric("domain_class_loss", self.domain_class_loss.cpu().detach())
            comet_exp.log_metric("probability A being identified as A", out_a.cpu().detach())
            comet_exp.log_metric("probability B being identified as B", out_b.cpu().detach())

    def domain_classifier_sr_update(
        self, x_a, x_b, m_a, m_b, domain_synth, lambda_classifier, step, comet_exp=None
    ):

        self.classif_opt_sr.zero_grad()

        # encode
        c_a = self.gen.encode(x_a, 1)
        c_b = self.gen.encode(x_b, 2)

        # noise = c_a.data.new(c_a.size()).normal_(0, 1)
        loss = self.compute_classifier_sr_loss(c_a.detach(), c_b.detach(), domain_synth, fool=False)
        loss = lambda_classifier * loss
        loss.backward()
        self.classif_opt_sr.step()

        if comet_exp is not None:
            comet_exp.log_metric("loss_classifier_sr", loss.cpu().detach(), step=step)

    def output_domain_classifier_sr_update(
        self, x_ar, x_as, x_br, x_bs, hyperparameters, step, comet_exp=None
    ):

        self.output_classif_opt_sr.zero_grad()

        loss = self.output_classifier_sr_b.calc_dis_loss_sr(
            x_bs, x_br
        ) + self.output_classifier_sr_a.calc_dis_loss_sr(x_as, x_ar)
        loss = hyperparameters["adaptation"]["output_classifier_lambda"] * loss
        loss.backward()

        self.output_classif_opt_sr.step()

        if comet_exp is not None:
            comet_exp.log_metric("loss_output_classifier_sr", loss.cpu().detach(), step=step)

    def segmentation_head_update(self, x_a, x_b, target_a, target_b, lamb, comet_exp=None):

        self.segmentation_opt.zero_grad()

        # encode
        c_a = self.gen.encode(x_a, 1)
        c_b = self.gen.encode(x_b, 2)

        output_a = self.segmentation_head(c_a)
        output_b = self.segmentation_head(c_b)
        output_a = nn.functional.interpolate(
            input=output_a, size=(self.newsize, self.newsize), mode="bilinear"
        )
        output_b = nn.functional.interpolate(
            input=output_b, size=(self.newsize, self.newsize), mode="bilinear"
        )

        loss1 = nn.CrossEntropyLoss()(output_a, target_a.type(torch.long).squeeze(1).cuda())
        loss2 = nn.CrossEntropyLoss()(output_b, target_b.type(torch.long).squeeze(1).cuda())
        loss = (loss1 + loss2) * lamb

        loss.backward()
        self.segmentation_opt.step()

        if comet_exp is not None:
            comet_exp.log_metric("loss_semantic_head", loss.cpu().detach())

    def update_learning_rate(self):
        """
        Update the learning rate
        """
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.dann_scheduler is not None:
            self.dann_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        """
        Resume the training loading the network parameters

        Arguments:
            checkpoint_dir {string} -- path to the directory where the checkpoints are saved
            hyperparameters {dictionnary} -- dictionnary with all hyperparameters

        Returns:
            int -- number of iterations (used by the optimizer)
        """
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)

        self.gen.load_state_dict(state_dict["2"])

        # Load domain classifier
        if self.domain_classif_ab == 1:
            last_model_name = get_model_list(checkpoint_dir, "domain_classif")
            state_dict = torch.load(last_model_name)
            self.domain_classifier.load_state_dict(state_dict["d"])

        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_b.load_state_dict(state_dict["b"])

        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
        self.dis_opt.load_state_dict(state_dict["dis"])
        self.gen_opt.load_state_dict(state_dict["gen"])

        if self.domain_classif_ab == 1:
            self.dann_opt.load_state_dict(state_dict["dann"])
            self.dann_scheduler = get_scheduler(self.dann_opt, hyperparameters, iterations)
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print("Resume from iteration %d" % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        """
        Save generators, discriminators, and optimizers

        Arguments:
            snapshot_dir {string} -- directory path where to save the networks weights
            iterations {int} -- number of training iterations
        """
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, "gen_%08d.pt" % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, "dis_%08d.pt" % (iterations + 1))
        domain_classifier_name = os.path.join(
            snapshot_dir, "domain_classifier_%08d.pt" % (iterations + 1)
        )
        opt_name = os.path.join(snapshot_dir, "optimizer.pt")

        torch.save({"2": self.gen.state_dict()}, gen_name)
        torch.save({"b": self.dis_b.state_dict()}, dis_name)
        if self.domain_classif_ab:
            torch.save({"d": self.domain_classifier.state_dict()}, domain_classifier_name)
            torch.save(
                {
                    "gen": self.gen_opt.state_dict(),
                    "dis": self.dis_opt.state_dict(),
                    "dann": self.dann_opt.state_dict(),
                },
                opt_name,
            )
        else:
            torch.save(
                {"gen": self.gen_opt.state_dict(), "dis": self.dis_opt.state_dict()}, opt_name
            )

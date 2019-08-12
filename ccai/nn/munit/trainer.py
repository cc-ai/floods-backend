from ccai.nn.munit.networks import AdaINGen, AdaINGen_double, MsImageDis, VAEGen
from ccai.nn.munit.utils import (
    weights_init,
    get_model_list,
    vgg_preprocess,
    get_scheduler,
    load_flood_classifier,
    transform_torchVar,
    seg_transform,
    load_segmentation_model,
    decode_segmap,
    domainClassifier,
)
from torch.autograd import Variable
from torchvision import transforms
import torch
import torch.nn as nn
import os
from PIL import Image


class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters["lr"]
        self.gen_state = hyperparameters["gen_state"]
        self.guided = hyperparameters["guided"]
        self.newsize = hyperparameters["new_size"]
        self.semantic_w = hyperparameters["semantic_w"] > 0
        self.recon_mask = hyperparameters["recon_mask"] == 1
        self.dann_scheduler = None
        if "domain_adv_w" in hyperparameters.keys():
            self.domain_classif = hyperparameters["domain_adv_w"] > 0
        else:
            self.domain_classif = False

        if self.gen_state == 0:
            # Initiate the networks
            self.gen_a = AdaINGen(
                hyperparameters["input_dim_a"], hyperparameters["gen"]
            )  # auto-encoder for domain a
            self.gen_b = AdaINGen(
                hyperparameters["input_dim_b"], hyperparameters["gen"]
            )  # auto-encoder for domain b

        elif self.gen_state == 1:
            self.gen = AdaINGen_double(hyperparameters["input_dim_a"], hyperparameters["gen"])
        else:
            print("self.gen_state unknown value:", self.gen_state)

        self.dis_a = MsImageDis(
            hyperparameters["input_dim_a"], hyperparameters["dis"]
        )  # discriminator for domain a
        self.dis_b = MsImageDis(
            hyperparameters["input_dim_b"], hyperparameters["dis"]
        )  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters["gen"]["style_dim"]

        # fix the noise used in sampling
        display_size = int(hyperparameters["display_size"])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1)  # .cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1)  # .cuda()

        # Setup the optimizers
        beta1 = hyperparameters["beta1"]
        beta2 = hyperparameters["beta2"]
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())

        if self.gen_state == 0:
            gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        elif self.gen_state == 1:
            gen_params = list(self.gen.parameters())
        else:
            print("self.gen_state unknown value:", self.gen_state)

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
        self.dis_a.apply(weights_init("gaussian"))
        self.dis_b.apply(weights_init("gaussian"))

        # Load semantic segmentation model if needed
        if "semantic_w" in hyperparameters.keys() and hyperparameters["semantic_w"] > 0:
            self.segmentation_model = load_segmentation_model(hyperparameters["semantic_ckpt_path"])
            self.segmentation_model.eval()
            for param in self.segmentation_model.parameters():
                param.requires_grad = False

        # Load domain classifier if needed
        if "domain_adv_w" in hyperparameters.keys() and hyperparameters["domain_adv_w"] > 0:
            self.domain_classifier = domainClassifier(256)
            dann_params = list(self.domain_classifier.parameters())
            self.dann_opt = torch.optim.Adam(
                [p for p in dann_params if p.requires_grad],
                lr=lr,
                betas=(beta1, beta2),
                weight_decay=hyperparameters["weight_decay"],
            )
            self.domain_classifier.apply(weights_init("gaussian"))
            self.dann_scheduler = get_scheduler(self.dann_opt, hyperparameters)

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def recon_criterion_mask(self, input, target, mask):
        """
            Mask of the region you don't want to compute the recon loss.
        """
        return torch.mean(torch.abs(torch.mul((input - target), 1 - mask)))

    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        if self.gen_state == 0:
            c_a, s_a_fake = self.gen_a.encode(x_a)
            c_b, s_b_fake = self.gen_b.encode(x_b)
            x_ba = self.gen_a.decode(c_b, s_a)
            x_ab = self.gen_b.decode(c_a, s_b)
        elif self.gen_state == 1:
            c_a, s_a_fake = self.gen.encode(x_a, 1)
            c_b, s_b_fake = self.gen.encode(x_b, 2)
            x_ba = self.gen.decode(c_b, s_a, 1)
            x_ab = self.gen.decode(c_a, s_b, 2)
        else:
            print("self.gen_state unknown value:", self.gen_state)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hyperparameters, mask_a=None, mask_b=None, comet_exp=None):
        self.gen_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        if self.gen_state == 0:
            # encode
            c_a, s_a_prime = self.gen_a.encode(x_a)
            c_b, s_b_prime = self.gen_b.encode(x_b)
            # decode (within domain)
            x_a_recon = self.gen_a.decode(c_a, s_a_prime)
            x_b_recon = self.gen_b.decode(c_b, s_b_prime)
            # decode (cross domain)
            if self.guided == 0:
                x_ba = self.gen_a.decode(c_b, s_a)
                x_ab = self.gen_b.decode(c_a, s_b)
            elif self.guided == 1:
                x_ba = self.gen_a.decode(c_b, s_a_prime)
                x_ab = self.gen_b.decode(c_a, s_b_prime)
            else:
                print("self.guided unknown value:", self.guided)
            # encode again
            c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
            c_a_recon, s_b_recon = self.gen_b.encode(x_ab)
            # decode again (if needed)
            x_aba = (
                self.gen_a.decode(c_a_recon, s_a_prime)
                if hyperparameters["recon_x_cyc_w"] > 0
                else None
            )
            x_bab = (
                self.gen_b.decode(c_b_recon, s_b_prime)
                if hyperparameters["recon_x_cyc_w"] > 0
                else None
            )
        elif self.gen_state == 1:
            # encode
            c_a, s_a_prime = self.gen.encode(x_a, 1)
            c_b, s_b_prime = self.gen.encode(x_b, 2)
            # decode (within domain)
            x_a_recon = self.gen.decode(c_a, s_a_prime, 1)
            x_b_recon = self.gen.decode(c_b, s_b_prime, 2)
            # decode (cross domain)
            if self.guided == 0:
                x_ba = self.gen.decode(c_b, s_a, 1)
                x_ab = self.gen.decode(c_a, s_b, 2)
            elif self.guided == 1:
                x_ba = self.gen.decode(c_b, s_a_prime, 1)
                x_ab = self.gen.decode(c_a, s_b_prime, 2)
            else:
                print("self.guided unknown value:", self.guided)

            # encode again
            c_b_recon, s_a_recon = self.gen.encode(x_ba, 1)
            c_a_recon, s_b_recon = self.gen.encode(x_ab, 2)
            # decode again (if needed)
            x_aba = (
                self.gen.decode(c_a_recon, s_a_prime, 1)
                if hyperparameters["recon_x_cyc_w"] > 0
                else None
            )
            x_bab = (
                self.gen.decode(c_b_recon, s_b_prime, 2)
                if hyperparameters["recon_x_cyc_w"] > 0
                else None
            )
        else:
            print("self.gen_state unknown value:", self.gen_state)

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        if self.recon_mask:
            self.loss_gen_cycrecon_x_a = (
                self.recon_criterion_mask(x_aba, x_a, mask_a)
                if hyperparameters["recon_x_cyc_w"] > 0
                else 0
            )
            self.loss_gen_cycrecon_x_b = (
                self.recon_criterion_mask(x_bab, x_b, mask_b)
                if hyperparameters["recon_x_cyc_w"] > 0
                else 0
            )
        else:
            self.loss_gen_cycrecon_x_a = (
                self.recon_criterion(x_aba, x_a) if hyperparameters["recon_x_cyc_w"] > 0 else 0
            )
            self.loss_gen_cycrecon_x_b = (
                self.recon_criterion(x_bab, x_b) if hyperparameters["recon_x_cyc_w"] > 0 else 0
            )

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = (
            self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters["vgg_w"] > 0 else 0
        )
        self.loss_gen_vgg_b = (
            self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters["vgg_w"] > 0 else 0
        )

        # semantic-segmentation loss
        self.loss_sem_seg = (
            self.compute_semantic_seg_loss(x_a.squeeze(), x_ab.squeeze(), mask_a)
            + self.compute_semantic_seg_loss(x_b.squeeze(), x_ba.squeeze(), mask_b)
            if hyperparameters["semantic_w"] > 0
            else 0
        )
        # Domain adversarial loss (c_a and c_b are swapped because we want to the feature to be less informative
        # minmax (accuracy but max min loss)
        self.domain_adv_loss = (
            self.compute_domain_adv_loss(c_b, c_a) if hyperparameters["domain_adv_w"] > 0 else 0
        )

        # total loss
        self.loss_gen_total = (
            hyperparameters["gan_w"] * self.loss_gen_adv_a
            + hyperparameters["gan_w"] * self.loss_gen_adv_b
            + hyperparameters["recon_x_w"] * self.loss_gen_recon_x_a
            + hyperparameters["recon_s_w"] * self.loss_gen_recon_s_a
            + hyperparameters["recon_c_w"] * self.loss_gen_recon_c_a
            + hyperparameters["recon_x_w"] * self.loss_gen_recon_x_b
            + hyperparameters["recon_s_w"] * self.loss_gen_recon_s_b
            + hyperparameters["recon_c_w"] * self.loss_gen_recon_c_b
            + hyperparameters["recon_x_cyc_w"] * self.loss_gen_cycrecon_x_a
            + hyperparameters["recon_x_cyc_w"] * self.loss_gen_cycrecon_x_b
            + hyperparameters["vgg_w"] * self.loss_gen_vgg_a
            + hyperparameters["vgg_w"] * self.loss_gen_vgg_b
            + hyperparameters["semantic_w"] * self.loss_sem_seg
            + hyperparameters["domain_adv_w"] * self.domain_adv_loss
        )

        if comet_exp is not None:
            comet_exp.log_metric("loss_gen_adv_a", self.loss_gen_adv_a)
            comet_exp.log_metric("loss_gen_adv_b", self.loss_gen_adv_b)
            comet_exp.log_metric("loss_gen_recon_x_a", self.loss_gen_recon_x_a)
            comet_exp.log_metric("loss_gen_recon_s_a", self.loss_gen_recon_s_a)
            comet_exp.log_metric("loss_gen_recon_c_a", self.loss_gen_recon_c_a)
            comet_exp.log_metric("loss_gen_recon_x_b", self.loss_gen_recon_x_b)
            comet_exp.log_metric("loss_gen_recon_s_b", self.loss_gen_recon_s_b)
            comet_exp.log_metric("loss_gen_recon_c_b", self.loss_gen_recon_c_b)
            comet_exp.log_metric("loss_gen_cycrecon_x_a", self.loss_gen_cycrecon_x_a)
            comet_exp.log_metric("loss_gen_cycrecon_x_b", self.loss_gen_cycrecon_x_b)
            comet_exp.log_metric("loss_gen_total", self.loss_gen_total)
            if hyperparameters["vgg_w"] > 0:
                comet_exp.log_metric("loss_gen_vgg_a", self.loss_gen_vgg_a)
                comet_exp.log_metric("loss_gen_vgg_b", self.loss_gen_vgg_b)
            if hyperparameters["semantic_w"] > 0:
                comet_exp.log_metric("loss_sem_seg", self.loss_sem_seg)
            if hyperparameters["domain_adv_w"] > 0:
                comet_exp.log_metric("domain_adv_loss_gen", self.domain_adv_loss)

        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def compute_domain_adv_loss(self, c_a, c_b, compute_accuracy=False):
        output_a = self.domain_classifier(c_a)
        output_b = self.domain_classifier(c_b)

        label_domain_a = torch.tensor([0]).cuda()
        label_domain_b = torch.tensor([1]).cuda()

        loss = nn.CrossEntropyLoss()(output_a.unsqueeze(0), label_domain_a) + nn.CrossEntropyLoss()(
            output_b.unsqueeze(0), label_domain_b
        )
        if compute_accuracy:
            return loss, output_a[0], output_b[1]
        else:
            return loss

    def compute_semantic_seg_loss(self, img1, img2, mask):
        # denorm
        img1_denorm = (img1 + 1) / 2.0
        img2_denorm = (img2 + 1) / 2.0
        # norm for semantic seg network
        input_transformed1 = seg_transform()(img1_denorm).unsqueeze(0)
        input_transformed2 = seg_transform()(img2_denorm).unsqueeze(0)
        # compute labels from original image and logits from translated version
        target = self.segmentation_model(input_transformed1).squeeze().max(0)[1].unsqueeze(0)
        output = self.segmentation_model(input_transformed2)
        # Resize mask to the size of the image
        mask1 = torch.nn.functional.interpolate(mask, size=(self.newsize, self.newsize))
        mask1_tensor = torch.tensor(mask1, dtype=torch.long).cuda()
        # we want the masked region to be labeled as unknown (19 is not an existing label)
        target_with_mask = torch.mul(1 - mask1_tensor, target) + mask1_tensor * 19
        mask2 = torch.nn.functional.interpolate(mask, size=(self.newsize, self.newsize))
        mask_tensor = torch.tensor(mask2, dtype=torch.float).cuda()
        output_with_mask = torch.mul(1 - mask_tensor, output)
        # cat the mask as to the logits (loss=0 over the masked region)
        output_with_mask_cat = torch.cat(
            (output_with_mask.squeeze(), mask_tensor.squeeze().unsqueeze(0))
        )
        loss = nn.CrossEntropyLoss()(
            output_with_mask_cat.unsqueeze(0), target_with_mask.squeeze().unsqueeze(0)
        )
        return loss

    def sample(self, x_a, x_b):
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []

        if self.gen_state == 0:
            for i in range(x_a.size(0)):
                c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
                c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
                x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
                x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
                if self.guided == 0:
                    x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
                    x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
                    x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
                    x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
                elif self.guided == 1:
                    x_ba1.append(self.gen_a.decode(c_b, s_a_fake))  # s_a1[i].unsqueeze(0)))
                    x_ba2.append(self.gen_a.decode(c_b, s_a_fake))  # s_a2[i].unsqueeze(0)))
                    x_ab1.append(self.gen_b.decode(c_a, s_b_fake))  # s_b1[i].unsqueeze(0)))
                    x_ab2.append(self.gen_b.decode(c_a, s_b_fake))  # s_b2[i].unsqueeze(0)))
                else:
                    print("self.guided unknown value:", self.guided)

        elif self.gen_state == 1:
            for i in range(x_a.size(0)):
                c_a, s_a_fake = self.gen.encode(x_a[i].unsqueeze(0), 1)
                c_b, s_b_fake = self.gen.encode(x_b[i].unsqueeze(0), 2)
                x_a_recon.append(self.gen.decode(c_a, s_a_fake, 1))
                x_b_recon.append(self.gen.decode(c_b, s_b_fake, 2))
                if self.guided == 0:
                    x_ba1.append(self.gen.decode(c_b, s_a1[i].unsqueeze(0), 1))
                    x_ba2.append(self.gen.decode(c_b, s_a2[i].unsqueeze(0), 1))
                    x_ab1.append(self.gen.decode(c_a, s_b1[i].unsqueeze(0), 2))
                    x_ab2.append(self.gen.decode(c_a, s_b2[i].unsqueeze(0), 2))
                elif self.guided == 1:
                    x_ba1.append(self.gen.decode(c_b, s_a_fake, 1))  # s_a1[i].unsqueeze(0)))
                    x_ba2.append(self.gen.decode(c_b, s_a_fake, 1))  # s_a2[i].unsqueeze(0)))
                    x_ab1.append(self.gen.decode(c_a, s_b_fake, 2))  # s_b1[i].unsqueeze(0)))
                    x_ab2.append(self.gen.decode(c_a, s_b_fake, 2))  # s_b2[i].unsqueeze(0)))
                else:
                    print("self.guided unknown value:", self.guided)

        else:
            print("self.gen_state unknown value:", self.gen_state)

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

    def dis_update(self, x_a, x_b, hyperparameters, comet_exp=None):
        self.dis_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        if self.gen_state == 0:
            # encode
            c_a, s_a_prime = self.gen_a.encode(x_a)
            c_b, s_b_prime = self.gen_b.encode(x_b)
            # decode (cross domain)
            if self.guided == 0:
                x_ba = self.gen_a.decode(c_b, s_a)
                x_ab = self.gen_b.decode(c_a, s_b)
            elif self.guided == 1:
                x_ba = self.gen_a.decode(c_b, s_a_prime)
                x_ab = self.gen_b.decode(c_a, s_b_prime)
            else:
                print("self.guided unknown value:", self.guided)
        elif self.gen_state == 1:
            # encode
            c_a, s_a_prime = self.gen.encode(x_a, 1)
            c_b, s_b_prime = self.gen.encode(x_b, 2)
            # decode (cross domain)
            if self.guided == 0:
                x_ba = self.gen.decode(c_b, s_a, 1)
                x_ab = self.gen.decode(c_a, s_b, 2)
            elif self.guided == 1:
                x_ba = self.gen.decode(c_b, s_a_prime, 1)
                x_ab = self.gen.decode(c_a, s_b_prime, 2)
            else:
                print("self.guided unknown value:", self.guided)
        else:
            print("self.gen_state unknown value:", self.gen_state)

        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)

        if comet_exp is not None:
            comet_exp.log_metric("loss_dis_b", self.loss_dis_b)
            comet_exp.log_metric("loss_dis_a", self.loss_dis_a)

        self.loss_dis_total = (
            hyperparameters["gan_w"] * self.loss_dis_a + hyperparameters["gan_w"] * self.loss_dis_b
        )
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def domain_classifier_update(self, x_a, x_b, hyperparameters, comet_exp=None):
        self.dann_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        if self.gen_state == 0:
            # encode
            c_a, _ = self.gen_a.encode(x_a)
            c_b, _ = self.gen_b.encode(x_b)
        elif self.gen_state == 1:
            # encode
            c_a, _ = self.gen.encode(x_a, 1)
            c_b, _ = self.gen.encode(x_b, 2)
        else:
            print("self.gen_state unknown value:", self.gen_state)

        # domain classifier loss
        self.domain_class_loss, out_a, out_b = self.compute_domain_adv_loss(c_a, c_b, True)

        if comet_exp is not None:
            comet_exp.log_metric("domain_class_loss", self.domain_class_loss)
            comet_exp.log_metric("probability A being identified as A", out_a)
            comet_exp.log_metric("probability B being identified as B", out_b)

        self.domain_class_loss.backward()
        self.dann_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.dann_scheduler is not None:
            self.dann_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        if self.gen_state == 0:
            self.gen_a.load_state_dict(state_dict["a"])
            self.gen_b.load_state_dict(state_dict["b"])
        elif self.gen_state == 1:
            self.gen.load_state_dict(state_dict["2"])
        else:
            print("self.gen_state unknown value:", self.gen_state)

        # Load domain classifier
        if self.domain_classif == 1:
            last_model_name = get_model_list(checkpoint_dir, "domain_classif")
            state_dict = torch.load(last_model_name)
            self.domain_classifier.load_state_dict(state_dict["d"])

        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict["a"])
        self.dis_b.load_state_dict(state_dict["b"])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
        self.dis_opt.load_state_dict(state_dict["dis"])
        self.gen_opt.load_state_dict(state_dict["gen"])

        if self.domain_classif == 1:
            self.dann_opt.load_state_dict(state_dict["dann"])
            self.dann_scheduler = get_scheduler(self.dann_opt, hyperparameters, iterations)
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print("Resume from iteration %d" % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, "gen_%08d.pt" % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, "dis_%08d.pt" % (iterations + 1))
        domain_classifier_name = os.path.join(
            snapshot_dir, "domain_classifier_%08d.pt" % (iterations + 1)
        )
        opt_name = os.path.join(snapshot_dir, "optimizer.pt")
        if self.gen_state == 0:
            torch.save({"a": self.gen_a.state_dict(), "b": self.gen_b.state_dict()}, gen_name)
        elif self.gen_state == 1:
            torch.save({"2": self.gen.state_dict()}, gen_name)
        else:
            print("self.gen_state unknown value:", self.gen_state)
        torch.save({"a": self.dis_a.state_dict(), "b": self.dis_b.state_dict()}, dis_name)
        if self.domain_classif:
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


class UNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(UNIT_Trainer, self).__init__()
        lr = hyperparameters["lr"]
        # Initiate the networks
        self.gen_a = VAEGen(
            hyperparameters["input_dim_a"], hyperparameters["gen"]
        )  # auto-encoder for domain a
        self.gen_b = VAEGen(
            hyperparameters["input_dim_b"], hyperparameters["gen"]
        )  # auto-encoder for domain b
        self.dis_a = MsImageDis(
            hyperparameters["input_dim_a"], hyperparameters["dis"]
        )  # discriminator for domain a
        self.dis_b = MsImageDis(
            hyperparameters["input_dim_b"], hyperparameters["dis"]
        )  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # Setup the optimizers
        beta1 = hyperparameters["beta1"]
        beta2 = hyperparameters["beta2"]
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
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
        self.dis_a.apply(weights_init("gaussian"))
        self.dis_b.apply(weights_init("gaussian"))

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(h_a + n_a)
        x_b_recon = self.gen_b.decode(h_b + n_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = (
            self.gen_a.decode(h_a_recon + n_a_recon)
            if hyperparameters["recon_x_cyc_w"] > 0
            else None
        )
        x_bab = (
            self.gen_b.decode(h_b_recon + n_b_recon)
            if hyperparameters["recon_x_cyc_w"] > 0
            else None
        )

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = (
            self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters["vgg_w"] > 0 else 0
        )
        self.loss_gen_vgg_b = (
            self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters["vgg_w"] > 0 else 0
        )
        # total loss
        self.loss_gen_total = (
            hyperparameters["gan_w"] * self.loss_gen_adv_a
            + hyperparameters["gan_w"] * self.loss_gen_adv_b
            + hyperparameters["recon_x_w"] * self.loss_gen_recon_x_a
            + hyperparameters["recon_kl_w"] * self.loss_gen_recon_kl_a
            + hyperparameters["recon_x_w"] * self.loss_gen_recon_x_b
            + hyperparameters["recon_kl_w"] * self.loss_gen_recon_kl_b
            + hyperparameters["recon_x_cyc_w"] * self.loss_gen_cyc_x_a
            + hyperparameters["recon_kl_cyc_w"] * self.loss_gen_recon_kl_cyc_aba
            + hyperparameters["recon_x_cyc_w"] * self.loss_gen_cyc_x_b
            + hyperparameters["recon_kl_cyc_w"] * self.loss_gen_recon_kl_cyc_bab
            + hyperparameters["vgg_w"] * self.loss_gen_vgg_a
            + hyperparameters["vgg_w"] * self.loss_gen_vgg_b
        )
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = (
            hyperparameters["gan_w"] * self.loss_dis_a + hyperparameters["gan_w"] * self.loss_dis_b
        )
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict["a"])
        self.gen_b.load_state_dict(state_dict["b"])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict["a"])
        self.dis_b.load_state_dict(state_dict["b"])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
        self.dis_opt.load_state_dict(state_dict["dis"])
        self.gen_opt.load_state_dict(state_dict["gen"])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print("Resume from iteration %d" % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, "gen_%08d.pt" % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, "dis_%08d.pt" % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, "optimizer.pt")
        torch.save({"a": self.gen_a.state_dict(), "b": self.gen_b.state_dict()}, gen_name)
        torch.save({"a": self.dis_a.state_dict(), "b": self.dis_b.state_dict()}, dis_name)
        torch.save({"gen": self.gen_opt.state_dict(), "dis": self.dis_opt.state_dict()}, opt_name)

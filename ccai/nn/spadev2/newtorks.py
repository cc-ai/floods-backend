"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import functools
import math

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass

##################################################################################
# Discriminator
##################################################################################


class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params):
        super(MsImageDis, self).__init__()
        self.n_layer = params["n_layer"]
        self.gan_type = params["gan_type"]
        self.dim = params["dim"]
        self.norm = params["norm"]
        self.activ = params["activ"]
        self.num_scales = params["num_scales"]
        self.pad_type = params["pad_type"]
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [
            Conv2dBlock(
                self.input_dim,
                dim,
                4,
                2,
                1,
                norm="none",
                activation=self.activ,
                pad_type=self.pad_type,
            )
        ]
        for i in range(self.n_layer - 1):
            cnn_x += [
                Conv2dBlock(
                    dim,
                    dim * 2,
                    4,
                    2,
                    1,
                    norm=self.norm,
                    activation=self.activ,
                    pad_type=self.pad_type,
                )
            ]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real, comet_exp=None, mode=None):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0
        fake_loss = 0
        real_loss = 0
        # print(len(outs0), len(outs1))

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            #    print(out0.shape)
            #    print(out1.shape)

            if self.gan_type == "lsgan":
                loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
            elif self.gan_type == "nsgan":
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(
                    F.binary_cross_entropy(F.sigmoid(out0), all0)
                    + F.binary_cross_entropy(F.sigmoid(out1), all1)
                )
            elif self.gan_type == "wgan":
                loss += torch.mean(out1) - torch.mean(out0)
                real_loss += torch.mean(out1)
                fake_loss += torch.mean(out0)

            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        if self.gan_type == "wgan":
            loss_gp = self.calc_gradient_penalty(input_real, input_fake)
            loss += loss_gp

            if mode != None and type(mode) == str:
                comet_exp.log_metric("loss_dis_real_" + mode, real_loss.cpu().detach())
                comet_exp.log_metric("loss_dis_fake_" + mode, fake_loss.cpu().detach())

            # Get gradient penalty loss
            comet_exp.log_metric("loss_gp" + mode, loss_gp.cpu().detach())

        return loss

    def calc_gradient_penalty(self, real_data, fake_data):
        #! Hardcoded
        DIM = 256
        LAMBDA = 100
        nc = self.input_dim + 1
        alpha = torch.rand(real_data.shape)
        # alpha = alpha.view(batch_size, nc, DIM, DIM)
        # alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()

        alpha = alpha.cuda()
        interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

        interpolates = interpolates.cuda()
        interpolates.requires_grad_(True)

        disc_interpolates = self.forward(interpolates)

        total_gp = 0
        for count, i in enumerate(disc_interpolates):
            gradients = autograd.grad(
                outputs=i,
                inputs=interpolates,
                grad_outputs=torch.ones(i.size()).cuda(),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
            total_gp += gradient_penalty

        return total_gp

    def calc_gen_loss(self, input_fake, input_real, comet_exp=None, mode=None):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == "lsgan":
                loss += torch.mean((out0 - 1) ** 2)  # LSGAN
            elif self.gan_type == "nsgan":
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            elif self.gan_type == "wgan":
                loss += torch.mean(out0)
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        if mode != None and type(mode) == str and self.gan_type == "wgan":
            comet_exp.log_metric("loss_gen_wgan_" + mode, loss.cpu().detach())

        return loss

    def calc_dis_loss_sr(self, input_sim, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_sim)
        outs1 = self.forward(input_real)
        loss = 0
        # print(len(outs0), len(outs1))

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            #    print(out0.shape)
            #    print(out1.shape)

            if self.gan_type == "lsgan":
                loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
            elif self.gan_type == "nsgan":
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(
                    F.binary_cross_entropy(F.sigmoid(out0), all0)
                    + F.binary_cross_entropy(F.sigmoid(out1), all1)
                )
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss_sr(self, input_fake):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        loss = 0
        # print(len(outs0), len(outs1))

        for it, (out0) in enumerate(outs0):
            #    print(out0.shape)
            #    print(out1.shape)

            if self.gan_type == "lsgan":
                loss += torch.mean((out0 - 0.5) ** 2)
            elif self.gan_type == "nsgan":
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(
                    F.binary_cross_entropy(F.sigmoid(out0), all0)
                    + F.binary_cross_entropy(F.sigmoid(out1), all1)
                )
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


# Source: https://github.com/NVIDIA/pix2pixHD
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, params):
        super(MultiscaleDiscriminator, self).__init__()
        # self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
        #         use_sigmoid=False, num_D=3, getIntermFeat=False
        self.getIntermFeat = params["getIntermFeat"]
        self.n_layers = params["n_layer"]
        self.ndf = params["dim"]
        self.norm_layer = get_norm_layer(params["norm"])
        self.use_sigmoid = params["use_sigmoid"]
        self.getIntermFeat = params["getIntermFeat"]
        self.num_D = params["num_D"]
        self.gan_type = params["gan_type"]
        self.lambda_feat = params["lambda_feat"]

        for i in range(self.num_D):
            netD = NLayerDiscriminator(input_nc, params)
            if self.getIntermFeat:
                for j in range(self.n_layers + 2):
                    setattr(
                        self, "scale" + str(i) + "_layer" + str(j), getattr(netD, "model" + str(j))
                    )
            else:
                setattr(self, "layer" + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [
                    getattr(self, "scale" + str(num_D - 1 - i) + "_layer" + str(j))
                    for j in range(self.n_layers + 2)
                ]
            else:
                model = getattr(self, "layer" + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

    def calc_dis_loss(self, input_fake, input_real, comet_exp=None, mode=None):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0
        # print(len(outs0), len(outs1))

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            out0 = out0[-1]
            out1 = out1[-1]

            if self.gan_type == "lsgan":
                loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
            elif self.gan_type == "nsgan":
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(
                    F.binary_cross_entropy(F.sigmoid(out0), all0)
                    + F.binary_cross_entropy(F.sigmoid(out1), all1)
                )
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        return loss

    def calc_gen_loss(self, input_fake, input_real, comet_exp=None, mode=None):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0
        feat_weights = 4.0 / (self.n_layers + 1)
        D_weights = 1.0 / self.num_D
        self.criterionFeat = torch.nn.L1Loss()
        loss_G_GAN_Feat = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            out0_output = out0[-1]
            out1_output = out1[-1]

            if self.gan_type == "lsgan":
                loss += torch.mean((out0_output - 1) ** 2)  # LSGAN
            elif self.gan_type == "nsgan":
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

            if self.getIntermFeat:

                for j in range(len(out0) - 1):
                    loss_G_GAN_Feat += (
                        D_weights
                        * feat_weights
                        * self.criterionFeat(out0[j], out1[j].detach())
                        * self.lambda_feat
                    )
                if mode != None and type(mode) == str:
                    comet_exp.log_metric("loss_gen_feat_" + mode, loss_G_GAN_Feat.cpu().detach())
                loss += loss_G_GAN_Feat

            return loss


# Defines the PatchGAN discriminator with the specified arguments.
# Source: https://github.com/NVIDIA/pix2pixHD
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, params):
        # ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = params["getIntermFeat"]
        self.n_layers = params["n_layer"]
        self.ndf = params["dim"]
        self.norm_layer = get_norm_layer(params["norm"])
        self.use_sigmoid = params["use_sigmoid"]
        self.getIntermFeat = params["getIntermFeat"]

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [
            [
                nn.Conv2d(input_nc, self.ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        nf = self.ndf
        for n in range(1, self.n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                [
                    nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                    self.norm_layer(nf),
                    nn.LeakyReLU(0.2, True),
                ]
            ]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                self.norm_layer(nf),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if self.use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if self.getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, "model" + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, "model" + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)[-1]


##################################################################################
# Generator
##################################################################################


class VAEGen(nn.Module):
    # VAE architecture
    def __init__(self, input_dim, params):
        super(VAEGen, self).__init__()
        dim = params["dim"]
        n_downsample = params["n_downsample"]
        n_res = params["n_res"]
        activ = params["activ"]
        pad_type = params["pad_type"]

        # content encoder
        self.enc = ContentEncoder(
            n_downsample, n_res, input_dim, dim, "in", activ, pad_type=pad_type
        )
        self.dec = Decoder(
            n_downsample,
            n_res,
            self.enc.output_dim,
            input_dim,
            res_norm="in",
            activ=activ,
            pad_type=pad_type,
        )

    def forward(self, images):
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        hiddens = self.encode(images)
        if self.training == True:
            noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        hiddens = self.enc(images)
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images


##################################################################################
# Encoder and Decoders
##################################################################################


class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [
            Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)
        ]
        for i in range(2):
            self.model += [
                Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type,)
            ]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [
                Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
            ]
        self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [
            Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)
        ]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [
                Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type,)
            ]
            dim *= 2

        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(
        self, n_upsample, n_res, dim, output_dim, res_norm="adain", activ="relu", pad_type="zero",
    ):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [
                nn.Upsample(scale_factor=2),
                Conv2dBlock(
                    dim, dim // 2, 5, 1, 2, norm="ln", activation=activ, pad_type=pad_type,
                ),
            ]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [
            Conv2dBlock(
                dim, output_dim, 7, 1, 3, norm="none", activation="tanh", pad_type=pad_type,
            )
        ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm="in", activation="relu", pad_type="zero"):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm="none", activ="relu"):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [
            LinearBlock(dim, output_dim, norm="none", activation="none")
        ]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm="in", activation="relu", pad_type="zero"):
        super(ResBlock, self).__init__()

        model = []
        model += [
            Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)
        ]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation="none", pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
        padding=0,
        norm="none",
        activation="relu",
        pad_type="zero",
    ):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == "batch":
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == "in":
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == "ln":
            self.norm = LayerNorm(norm_dim)
        elif norm == "adain":
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == "none" or norm == "sn":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == "sn":
            self.conv = SpectralNorm(
                nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)
            )
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm="none", activation="relu"):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == "sn":
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == "batch":
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == "ln":
            self.norm = LayerNorm(norm_dim)
        elif norm == "none" or norm == "sn":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


##################################################################################
# VGG network definition
##################################################################################
from torchvision import models

# Source: https://github.com/NVIDIA/pix2pixHD
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# Source: https://github.com/NVIDIA/pix2pixHD
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda().eval()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


##################################################################################
# Normalization layers
##################################################################################


def get_norm_layer(norm_type="instance"):
    if not norm_type:
        print("norm_type is {}, defaulting to instance")
        norm_type = "instance"
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """

    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class SPADE(nn.Module):
    def __init__(self, param_free_norm_type, kernel_size, norm_nc, cond_nc):
        super().__init__()

        if param_free_norm_type == "instance":
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        # elif param_free_norm_type == "syncbatch":
        #     self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == "batch":
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError(
                "%s is not a recognized param-free norm type in SPADE" % param_free_norm_type
            )

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        self.norm_nc = norm_nc
        pw = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(cond_nc, nhidden, kernel_size=kernel_size, padding=pw), nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


##################################################################################
# SPADE
##################################################################################
class SPADEResnetBlock(nn.Module):
    def __init__(
        self, fin, fout, cond_nc, spade_use_spectral_norm, spade_param_free_norm, spade_kernel_size,
    ):
        super().__init__()
        # Attributes

        self.fin = fin
        self.fout = fout
        self.use_spectral_norm = spade_use_spectral_norm
        self.param_free_norm = spade_param_free_norm
        self.kernel_size = spade_kernel_size

        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if spade_use_spectral_norm:
            self.conv_0 = SpectralNorm(self.conv_0)
            self.conv_1 = SpectralNorm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = SpectralNorm(self.conv_s)

        self.norm_0 = SPADE(spade_param_free_norm, spade_kernel_size, fin, cond_nc)
        self.norm_1 = SPADE(spade_param_free_norm, spade_kernel_size, fmiddle, cond_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_param_free_norm, spade_kernel_size, fin, cond_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.activation(self.norm_0(x, seg)))
        dx = self.conv_1(self.activation(self.norm_1(dx, seg)))

        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def activation(self, x):
        return F.leaky_relu(x, 2e-1)

    def __str__(self):
        return strings.spaderesblock(self)


class SpadeDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        cond_nc,
        spade_n_up,
        spade_use_spectral_norm,
        spade_param_free_norm,
        spade_kernel_size,
    ):
        """Create a SPADE-based decoder, which forwards z and the conditioning
        tensors seg (in the original paper, conditioning is on a semantic map only).
        All along, z is conditioned on seg. First 3 SpadeResblocks (SRB) do not shrink
        the channel dimension, and an upsampling is applied after each. Therefore
        2 upsamplings at this point. Then, for each remaining upsamplings
        (w.r.t. spade_n_up), the SRB shrinks channels by 2. Before final conv to get 3
        channels, the number of channels is therefore:
            final_nc = channels(z) * 2 ** (spade_n_up - 2)
        Args:
            latent_dim (tuple): z's shape (only the number of channels matters)
            cond_nc (int): conditioning tensor's expected number of channels
            spade_n_up (int): Number of total upsamplings from z
            spade_use_spectral_norm (bool): use spectral normalization?
            spade_param_free_norm (str): norm to use before SPADE de-normalization
            spade_kernel_size (int): SPADE conv layers' kernel size
            fullspade (bool): Produces an image as a final  result
        Returns:
            [type]: [description]
        """
        super().__init__()

        self.z_nc = latent_dim
        self.spade_n_up = spade_n_up

        self.head_0 = SPADEResnetBlock(
            self.z_nc,
            self.z_nc,
            cond_nc,
            spade_use_spectral_norm,
            spade_param_free_norm,
            spade_kernel_size,
        )

        self.G_middle_0 = SPADEResnetBlock(
            self.z_nc,
            self.z_nc,
            cond_nc,
            spade_use_spectral_norm,
            spade_param_free_norm,
            spade_kernel_size,
        )
        self.G_middle_1 = SPADEResnetBlock(
            self.z_nc,
            self.z_nc,
            cond_nc,
            spade_use_spectral_norm,
            spade_param_free_norm,
            spade_kernel_size,
        )

        self.up_spades = nn.Sequential(
            *[
                SPADEResnetBlock(
                    self.z_nc // 2 ** i,
                    self.z_nc // 2 ** (i + 1),
                    cond_nc,
                    spade_use_spectral_norm,
                    spade_param_free_norm,
                    spade_kernel_size,
                )
                for i in range(spade_n_up - 2)
            ]
        )

        self.final_nc = self.z_nc // 2 ** (spade_n_up - 2)


        self.conv_img = nn.Conv2d(self.final_nc, 3, 3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2)

    def _apply(self, fn):
        # print("Applying SpadeDecoder", fn)
        super()._apply(fn)
        # self.head_0 = fn(self.head_0)
        # self.G_middle_0 = fn(self.G_middle_0)
        # self.G_middle_1 = fn(self.G_middle_1)
        # for i, up in enumerate(self.up_spades):
        #     self.up_spades[i] = fn(up)
        # self.conv_img = fn(self.conv_img)
        return self

    def forward(self, z, cond):
        y = self.head_0(z, cond)

        y = self.upsample(y)
        y = self.G_middle_0(y, cond)
        y = self.upsample(y)
        y = self.G_middle_1(y, cond)

        for i, up in enumerate(self.up_spades):
            y = self.upsample(y)
            y = up(y, cond)

        y = self.conv_img(F.leaky_relu(y, 2e-1))
        y = torch.tanh(y)
        return y

    def __str__(self):
        return strings.spadedecoder(self)


class SpadeGen(nn.Module):
    def __init__(self, input_dim, params):
        super(SpadeGen, self).__init__()
        dim = params["dim"]
        n_downsample = params["n_downsample"]
        n_res = params["n_res"]
        activ = params["activ"]
        pad_type = params["pad_type"]
        mlp_dim = params["mlp_dim"]

        # content encoder
        self.enc1_content = ContentEncoder(
            n_downsample, n_res, input_dim, dim, "in", activ, pad_type=pad_type
        )
        self.enc2_content = ContentEncoder(
            n_downsample, n_res, input_dim, dim, "in", activ, pad_type=pad_type
        )

        latent_dim = self.enc1_content.output_dim

        self.dec1 = SpadeDecoder(
            latent_dim=latent_dim,
            cond_nc=1,
            spade_n_up=n_downsample,
            spade_use_spectral_norm=True,
            spade_param_free_norm="instance",
            spade_kernel_size=3,
        )

        self.dec2 = SpadeDecoder(
            latent_dim=latent_dim,
            cond_nc=1,
            spade_n_up=n_downsample,
            spade_use_spectral_norm=True,
            spade_param_free_norm="instance",
            spade_kernel_size=3,
        )

    def forward(self, images, masks, encoder_name):
        # reconstruct an image
        content = self.encode(images, encoder_name)
        images_recon = self.decode(content, masks, encoder_name)
        return images_recon

    def encode(self, images, encoder_name):
        # encode an image to its content and style codes
        if encoder_name == 1:
            content = self.enc1_content(images)
        elif encoder_name == 2:
            content = self.enc2_content(images)
        else:
            print("wrong value for encoder_name, must be 0 or 1")
            return None
        return content

    def decode(self, content, mask, encoder_name):
        # decode content and style codes to an image
        if encoder_name == 1:
            images = self.dec1(content, mask)
        elif encoder_name == 2:
            images = self.dec2(content, mask)
        else:
            print("wrong value for encoder_name, must be 0 or 1")
            return None
        return images


class SpadeTestGen(nn.Module):
    def __init__(self, input_dim, size, batch_size, params):
        super(SpadeTestGen, self).__init__()
        dim = params["dim"]
        n_downsample = params["n_downsample"]
        n_res = params["n_res"]
        activ = params["activ"]
        pad_type = params["pad_type"]
        mlp_dim = params["mlp_dim"]

        self.latent_dim = dim
        self.batch_size = batch_size
        self.size = size

        # Get size of latent vector based on downsampling:
        self.dec = SpadeDecoder(
            latent_dim=self.latent_dim,
            cond_nc=4,
            spade_n_up=n_downsample,
            spade_use_spectral_norm=True,
            spade_param_free_norm="instance",
            spade_kernel_size=3,
        )

    def forward(self, z, cond):
        return self.dec(z, cond)


class SpadeAdaINGen(nn.Module):
    def __init__(self, input_dim, size, batch_size, params):
        super(SpadeAdaINGen, self).__init__()
        dim = params["dim"]
        n_downsample = params["n_downsample"]
        n_spade = params["n_spade"]
        n_res = params["n_res"]
        activ = params["activ"]
        pad_type = params["pad_type"]
        mlp_dim = params["mlp_dim"]
        n_down_adain = params["n_down_adain"]

        self.latent_dim = dim
        self.batch_size = batch_size

        # Get size of latent vector based on downsampling:
        self.dec1 = SpadeDecoder(
            latent_dim=self.latent_dim,
            cond_nc=4,
            spade_n_up=n_spade,
            spade_use_spectral_norm=True,
            spade_param_free_norm="instance",
            spade_kernel_size=3,
            fullspade=False,
        )

        nc = self.dec1.final_nc

        # n_upsample, n_res, dim, output_dim

        self.dec2 = Decoder(
            n_downsample - n_spade, n_res, nc, 3, res_norm="adain", activ=activ, pad_type=pad_type,
        )

        num_adain = self.get_num_adain_params(self.dec2)

        self.ParamGen = StyleEncoder(
            n_down_adain, 3, dim, num_adain, "in", activ, pad_type=pad_type
        )

    def forward(self, z, cond, style_im):
        y = self.dec1(z, cond)
        adain_params = self.ParamGen(style_im)
        self.assign_adain_params(adain_params, self.dec2)
        y = self.dec2(y)
        return y

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, : m.num_features]
                std = adain_params[:, m.num_features : 2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features :]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def get_adain_param(self, style):
        """
        output of mlp on style
        """
        adain_params = self.mlp(style)
        return adain_params


# This one has an encoder/decoder architecture
class SpadeAdaINGenII(nn.Module):
    def __init__(self, input_dim, size, batch_size, params):
        super(SpadeAdaINGenII, self).__init__()
        dim = params["dim"]
        n_downsample = params["n_downsample"]
        n_spade = params["n_spade"]
        n_res = params["n_res"]
        activ = params["activ"]
        pad_type = params["pad_type"]
        mlp_dim = params["mlp_dim"]
        n_down_adain = params["n_down_adain"]

        self.latent_dim = dim
        self.batch_size = batch_size

        self.enc = ContentEncoder(n_downsample, 0, input_dim, dim, "in", activ, pad_type=pad_type)

        latent_dim = self.enc.output_dim
        self.adain = Decoder(
            0, n_res, latent_dim, latent_dim, res_norm="adain", activ=activ, pad_type=pad_type,
        )

        self.dec = SpadeDecoder(
            latent_dim=latent_dim,
            cond_nc=1,
            spade_n_up=n_downsample,
            spade_use_spectral_norm=True,
            spade_param_free_norm="instance",
            spade_kernel_size=3,
        )

        num_adain = self.get_num_adain_params(self.adain)

        self.ParamGen = StyleEncoder(
            n_down_adain, 3, dim, num_adain, "in", activ, pad_type=pad_type
        )

    def forward(self, input, cond, style_im):
        y = self.enc(input)
        adain_params = self.ParamGen(style_im)
        self.assign_adain_params(adain_params, self.adain)
        y = self.adain(y)
        y = self.dec(y, cond)

        return y

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, : m.num_features]
                std = adain_params[:, m.num_features : 2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features :]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def get_adain_param(self, style):
        """
        output of mlp on style
        """
        adain_params = self.mlp(style)
        return adain_params

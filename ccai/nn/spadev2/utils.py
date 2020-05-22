"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms, models
from ccai.nn.spadev2.data  import ImageFilelist, ImageFolder
from PIL import Image
from torchvision.transforms import functional as F
import torch
import torch.nn as nn
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init
import time
from ccai.nn.spadev2.resnet import resnet34
import re
from addict import Dict

# Methods
# get_all_data_loaders          : primary data loader interface (load trainA, testA, trainB, testB)
# get_data_loader_list          : list-based data loader
# get_data_loader_folder        : folder-based data loader
# get_data_loader_mask_and_im   : masks and images lists-based data loader
# get_config                    : load yaml file
# eformat                       :
# write_2images                 : save output image
# prepare_sub_folder            : create checkpoints and images folders for saving outputs
# write_one_row_html            : write one row of the html file for output images
# write_html                    : create the html file.
# write_loss
# slerp
# get_slerp_interp
# get_model_list
# load_vgg16
# load_flood_classifier
# load_inception
# vgg_preprocess
# get_scheduler
# weights_init

# Worlds for the GAN will be called A and B


def get_all_data_loaders(conf):
    """primary data loader interface (load trainA, testA, trainB, testB)

    Arguments:
        conf {dict} -- configuration dictionary

    Returns:
        train_loader_a, train_loader_b, test_loader_a, test_loader_b
         -- data loaders for test and train sets in worlds A and B
    """
    batch_size = conf["batch_size"]
    num_workers = conf["num_workers"]
    if "new_size" in conf:
        new_size_a = new_size_b = conf["new_size"]
    else:
        new_size_a = conf["new_size_a"]
        new_size_b = conf["new_size_b"]
    height = conf["crop_image_height"]
    width = conf["crop_image_width"]

    if "data_root" in conf:
        train_loader_a = get_data_loader_folder(
            os.path.join(conf["data_root"], "trainA"),
            batch_size,
            True,
            new_size_a,
            height,
            width,
            num_workers,
            True,
        )
        test_loader_a = get_data_loader_folder(
            os.path.join(conf["data_root"], "testA"),
            batch_size,
            False,
            new_size_a,
            new_size_a,
            new_size_a,
            num_workers,
            True,
        )
        train_loader_b = get_data_loader_folder(
            os.path.join(conf["data_root"], "trainB"),
            batch_size,
            True,
            new_size_b,
            height,
            width,
            num_workers,
            True,
        )
        test_loader_b = get_data_loader_folder(
            os.path.join(conf["data_root"], "testB"),
            batch_size,
            False,
            new_size_b,
            new_size_b,
            new_size_b,
            num_workers,
            True,
        )
    else:
        train_loader_a = get_data_loader_list(
            conf["data_folder_train_a"],
            conf["data_list_train_a"],
            batch_size,
            True,
            new_size_a,
            height,
            width,
            num_workers,
            True,
        )
        test_loader_a = get_data_loader_list(
            conf["data_folder_test_a"],
            conf["data_list_test_a"],
            batch_size,
            False,
            new_size_a,
            new_size_a,
            new_size_a,
            num_workers,
            True,
        )
        train_loader_b = get_data_loader_list(
            conf["data_folder_train_b"],
            conf["data_list_train_b"],
            batch_size,
            True,
            new_size_b,
            height,
            width,
            num_workers,
            True,
        )
        test_loader_b = get_data_loader_list(
            conf["data_folder_test_b"],
            conf["data_list_test_b"],
            batch_size,
            False,
            new_size_b,
            new_size_b,
            new_size_b,
            num_workers,
            True,
        )
    return train_loader_a, train_loader_b, test_loader_a, test_loader_b


def seg_batch_transform(img_batch):
    N = img_batch.shape[0]
    for i in range(N):
        img_batch[i, :, :, :] = seg_transform()(img_batch[i, :, :, :])
    return img_batch


def seg_transform():
    """
    Transformations for segmentation model.
    The parameters for normalization are those corresponding to the ImageNet dataset
    """
    segmentation_transform = transforms.Compose(
        [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
    return segmentation_transform


def transform_torchVar():
    """Transformations for a Torch Tensor.
    """
    transfo = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transfo


def get_data_loader_list(
    root,
    file_list,
    batch_size,
    train,
    new_size=None,
    height=256,
    width=256,
    num_workers=4,
    crop=True,
):
    """ List-based data loader with transformations
     (horizontal flip, resizing, random crop, normalization are handled)

    Arguments:
        root {str} -- path root
        file_list {str list} -- list of the file names
        batch_size {int} --
        train {bool} -- training mode

    Keyword Arguments:
        new_size {int} -- parameter for resizing (default: {None})
        height {int} -- dimension for random cropping (default: {256})
        width {int} -- dimension for random cropping (default: {256})
        num_workers {int} -- number of workers (default: {4})
        crop {bool} -- crop(default: {True})

    Returns:
        loader -- data loader with transformed dataset
    """
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    transform_list = (
        [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    )
    transform_list = (
        [transforms.Resize((new_size, new_size))] + transform_list
        if new_size is not None
        else transform_list
    )
    transform_list = (
        [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    )
    transform = transforms.Compose(transform_list)
    dataset = ImageFilelist(root, file_list, transform=transform)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        num_workers=num_workers,
    )
    return loader


def default_txt_reader(flist):
    """
    Arguments:
        flist {str} -- path to text file with txt format: impath \n

    Returns:
        list of strings, each corresponding to one line of the input file
    """
    imlist = []
    print(flist)
    with open(flist, "r") as rf:
        for line in rf.readlines():
            impath = [line.strip()]

            # if "Screenshot" in line:
            #    print("**************")
            #    print(impath)
            # impath = line.strip().split()

            # if "Screenshot" in line:
            #    print(impath)
            #    print("***************")
            imlist.append(impath)
    return imlist


class MyDatasetRect(Dataset):
    """
    Dataset class for images and masks filenames inputs
    """

    def __init__(self, file_list, mask_list, rect_list, new_size, height, width):
        self.image_paths = default_txt_reader(file_list)
        if mask_list is not None:
            self.target_paths = default_txt_reader(mask_list)
            print("Segmentation mask will be used")
        else:
            self.target_paths = None
            print("No segmentation mask")
        if rect_list is not None:
            self.rect_paths = default_txt_reader(rect_list)
            print("Rectangles will be used")
        else:
            self.rect_paths = None
            print("No rectangles")

        self.new_size = new_size
        self.height = height
        self.width = width

    def transform(self, image, mask, rect):
        """Apply transformations to image and corresponding mask.
        Transformations applied are:
            random horizontal flipping, resizing, random cropping and normalizing
        Arguments:
            image {Image} -- Image
            mask {Image} -- Mask

        Returns:
            image, mask {Image, Image} -- transformed image and mask
        """

        flip = False
        # Random horizontal flipping
        if torch.rand(1) > 1.0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flip = True

        # print('debugging mask transform 2 size',mask.size)
        # Resize
        resize = transforms.Resize(size=(self.new_size, self.new_size))
        image = resize(image)
        to_tensor = transforms.ToTensor()
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.height, self.width))
        image = F.crop(image, i, j, h, w)

        if type(mask) is not torch.Tensor:
            # Resize mask
            if flip == True:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            # mask = mask.resize((image.width, image.height), Image.NEAREST)
            mask = resize(mask)
            mask = F.crop(mask, i, j, h, w)
            if np.max(mask) == 1:
                mask = to_tensor(mask) * 255
            else:
                mask = to_tensor(mask)

        # Make mask binary
        mask_thresh = (torch.max(mask) - torch.min(mask)) / 2.0
        mask = (mask > mask_thresh).float()

        # Transform to tensor

        image = to_tensor(image)

        # Process rectangle
        rect = F.crop(rect, i, j, h, w)
        rect = to_tensor(rect)

        # print('debugging mask transform 5 size',mask.size)
        # Normalize
        normalizer = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = normalizer(image)
        return image, mask, rect

    def __getitem__(self, index):
        """Get transformed image and mask at index index in the dataset

        Arguments:
            index {int} -- index at which to get image, mask pair

        Returns:
            image, mask pair
        """

        image = Image.open(self.image_paths[index][0]).convert("RGB")

        if self.target_paths is not None:
            mask = Image.open(self.target_paths[index][0])

        else:
            mask = torch.tensor([])

        if self.rect_paths is not None:
            rect = Image.open(self.rect_paths[index][0])
        else:
            rect = torch.tensor([])

        x, y, z = self.transform(image, mask, rect)
        y = y[0].unsqueeze(0)
        return x, y, z

    def __len__(self):
        """return dataset length

        Returns:
            int -- dataset length
        """
        return len(self.image_paths)


class MyDataset(Dataset):
    """
    Dataset class for images and masks filenames inputs
    """

    def __init__(self, file_list, mask_list, new_size, height, width):
        self.image_paths = default_txt_reader(file_list)
        if mask_list is not None:
            self.target_paths = default_txt_reader(mask_list)
            print("Segmentation mask will be used")
        else:
            self.target_paths = None
            print("No segmentation mask")
        self.new_size = new_size
        self.height = height
        self.width = width

    def transform(self, image, mask):
        """Apply transformations to image and corresponding mask.
        Transformations applied are:
            random horizontal flipping, resizing, random cropping and normalizing
        Arguments:
            image {Image} -- Image
            mask {Image} -- Mask

        Returns:
            image, mask {Image, Image} -- transformed image and mask
        """

        flip = False
        # Random horizontal flipping
        if torch.rand(1) > 1.0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flip = True

        # print('debugging mask transform 2 size',mask.size)
        # Resize
        resize = transforms.Resize(size=(self.new_size, self.new_size))
        image = resize(image)
        to_tensor = transforms.ToTensor()
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.height, self.width))
        image = F.crop(image, i, j, h, w)

        if type(mask) is not torch.Tensor:
            # Resize mask
            if flip == True:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            # mask = mask.resize((image.width, image.height), Image.NEAREST)
            mask = resize(mask)
            mask = F.crop(mask, i, j, h, w)
            if np.max(mask) == 1:
                mask = to_tensor(mask) * 255
            else:
                mask = to_tensor(mask)

        # Make mask binary
        mask_thresh = (torch.max(mask) - torch.min(mask)) / 2.0
        mask = (mask > mask_thresh).float()

        # Transform to tensor

        image = to_tensor(image)

        # print('debugging mask transform 5 size',mask.size)
        # Normalize
        normalizer = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = normalizer(image)
        return image, mask

    def __getitem__(self, index):
        """Get transformed image and mask at index index in the dataset

        Arguments:
            index {int} -- index at which to get image, mask pair

        Returns:
            image, mask pair
        """

        image = Image.open(self.image_paths[index][0]).convert("RGB")

        if self.target_paths is not None:
            mask = Image.open(self.target_paths[index][0])

        else:
            mask = torch.tensor([])

        x, y = self.transform(image, mask)
        y = y[0].unsqueeze(0)
        return x, y

    def __len__(self):
        """return dataset length

        Returns:
            int -- dataset length
        """
        return len(self.image_paths)


class DatasetInferenceFID(Dataset):
    """
    Dataset class for images and masks filenames inputs
    """

    def __init__(self, file_list_a, file_list_b, new_size):
        self.image_paths = default_txt_reader(file_list_a)
        self.target_paths = default_txt_reader(file_list_b)
        self.new_size = new_size

    def transform(self, image_a, image_b):
        """Apply transformations to image and corresponding mask.
        Transformations applied are:
            random horizontal flipping, resizing, random cropping and normalizing
        Arguments:
            image {Image} -- Image
            mask {Image} -- Mask

        Returns:
            image, mask {Image, Image} -- transformed image and mask
        """

        # Resize
        resize = transforms.Resize(size=(self.new_size, self.new_size))
        image_a = resize(image_a)
        image_b = resize(image_b)

        # Transform to tensor
        to_tensor = transforms.ToTensor()
        image_a = to_tensor(image_a)
        image_b = to_tensor(image_b)

        # Normalize
        normalizer = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image_a = normalizer(image_a)
        image_b = normalizer(image_a)
        return image_a, image_b

    def __getitem__(self, index):
        """Get transformed image and mask at index index in the dataset

        Arguments:
            index {int} -- index at which to get image, mask pair

        Returns:
            image, mask pair
        """
        image_a = Image.open(self.image_paths[index][0]).convert("RGB")
        image_b = Image.open(self.target_paths[index][0]).convert("RGB")
        x, y = self.transform(image_a, image_b)
        return x, y

    def __len__(self):
        """return dataset length

        Returns:
            int -- dataset length
        """
        return len(self.image_paths)


def get_fid_data_loader(file_list_a, file_list_b, batch_size, train, new_size=256, num_workers=4):
    """
    Masks and images lists-based data loader with transformations
    (horizontal flip, resizing, random crop, normalization are handled)

    Arguments:
        file_list_a {str list} -- list of images filenames domain A
        file_list_b {str list} -- list of images filenames domain B
        batch_size {int} -- batch size
        train {bool} -- training

    Keyword Arguments:
        new_size {int} -- parameter for resizing (default: {None})
        num_workers {int} -- number of workers (default: {4})

    Returns:
        loader -- data loader with transformed dataset
    """
    dataset = DatasetInferenceFID(file_list_a, file_list_b, new_size)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )
    return loader


class MyDatasetSynthetic(Dataset):
    """
    Dataset class for synthetic paired images and masks
    """

    def __init__(
        self,
        file_list_a,
        file_list_b,
        mask_list,
        semantic_a_list,
        semantic_b_list,
        new_size,
        height,
        width,
    ):
        self.image_paths = default_txt_reader(file_list_a)
        self.pair_paths = default_txt_reader(file_list_b)
        self.target_paths = default_txt_reader(mask_list)
        self.semantic_a = default_txt_reader(semantic_a_list)
        self.semantic_b = default_txt_reader(semantic_b_list)
        self.new_size = new_size
        self.height = height
        self.width = width

    def transform(self, image_a, image_b, mask, semantic_a, semantic_b):
        """Apply transformations to image and corresponding mask.
        Transformations applied are:
            random horizontal flipping, resizing, random cropping and normalizing
        Arguments:
            image_a {Image} -- Image
            image_b {Image} -- Image
            mask {Image} -- Mask
        Returns:
            image_a, image_b, mask {Image, Image, Image} -- transformed image_a, pair image_b and mask
        """
        # Random horizontal flipping
        if torch.rand(1) > 1:
            image_a = image_a.transpose(Image.FLIP_LEFT_RIGHT)
            image_b = image_b.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            semantic_a = semantic_a.transpose(Image.FLIP_LEFT_RIGHT)
            semantic_b = semantic_b.transpose(Image.FLIP_LEFT_RIGHT)

        # print('debugging mask transform 2 size',mask.size)
        # Resize
        resize = transforms.Resize(size=(self.new_size, self.new_size))
        image_a = resize(image_a)
        image_b = resize(image_b)
        # print('dim image after resize',image.size)

        # Resize mask
        # mask = mask.resize((image_b.width, image_b.height), Image.NEAREST)
        mask = resize(mask)
        semantic_a = semantic_a.resize((image_b.width, image_b.height), Image.NEAREST)
        semantic_b = semantic_b.resize((image_b.width, image_b.height), Image.NEAREST)

        # print('debugging mask transform 3 size',mask.size)
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image_b, output_size=(self.height, self.width)
        )
        image_a = F.crop(image_a, i, j, h, w)
        image_b = F.crop(image_b, i, j, h, w)

        mask = F.crop(mask, i, j, h, w)
        semantic_a = F.crop(semantic_a, i, j, h, w)
        semantic_b = F.crop(semantic_b, i, j, h, w)

        # print('debugging mask transform 4 size',mask.size)
        # Transform to tensor
        to_tensor = transforms.ToTensor()
        image_a = to_tensor(image_a)
        image_b = to_tensor(image_b)
        semantic_a = to_tensor(semantic_a) * 255  # to_tensor clip to 0:1
        semantic_b = to_tensor(semantic_b) * 255
        semantic_a = mapping(semantic_a)
        semantic_b = mapping(semantic_b)

        if np.max(mask) == 1:
            mask = to_tensor(mask) * 255

        else:
            mask = to_tensor(mask)
        mask[mask > 0.5] = 1
        mask[mask < 0.5] = 0

        # print('debugging mask transform 5 size',mask.size)
        # Normalize
        normalizer = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        image_a = normalizer(image_a)
        image_b = normalizer(image_b)
        # print(torch.unique(mask))
        # print(torch.unique(semantic_a))
        return image_a, image_b, mask, semantic_a, semantic_b

    def __getitem__(self, index):
        """Get transformed image and mask at index index in the dataset
        Arguments:
            index {int} -- index at which to get image, mask pair
        Returns:
            image_a, image_b, mask pair
        """
        image_a = Image.open(self.image_paths[index][0]).convert("RGB")
        image_b = Image.open(self.pair_paths[index][0]).convert("RGB")
        mask = Image.open(self.target_paths[index][0]).convert("L")
        semantic_a = Image.open(self.semantic_a[index][0]).convert("L")
        semantic_b = Image.open(self.semantic_b[index][0]).convert("L")

        # PALETIZED HERE
        x, y, z, sa, sb = self.transform(image_a, image_b, mask, semantic_a, semantic_b)
        return x, y, z, sa, sb

    def __len__(self):
        """return dataset length
        Returns:
            int -- dataset length
        """
        return len(self.image_paths)


def get_synthetic_data_loader(
    file_list_a,
    file_list_b,
    mask_list,
    sem_list_a,
    sem_list_b,
    batch_size,
    train,
    new_size=256,
    height=256,
    width=256,
    num_workers=4,
    crop=True,
):
    """
    Masks and images lists-based data loader with transformations
    (horizontal flip, resizing, random crop, normalization are handled)

    Arguments:
        file_list_a {str list} -- list of images filenames domain A
        file_list_b {str list} -- list of images filenames domain B
        mask_list {str list} -- list of masks filenames
        batch_size {int} -- batch size
        train {bool} -- training

    Keyword Arguments:
        new_size {int} -- parameter for resizing (default: {None})
        height {int} -- dimension for random cropping (default: {256})
        width {int} -- dimension for random cropping (default: {256})
        num_workers {int} -- number of workers (default: {4})
        crop {bool} -- crop(default: {True})

    Returns:
        loader -- data loader with transformed dataset
    """
    dataset = MyDatasetSynthetic(
        file_list_a, file_list_b, mask_list, sem_list_a, sem_list_b, new_size, height, width,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        num_workers=num_workers,
    )
    return loader


def get_data_loader_mask_and_im(
    file_list,
    mask_list,
    batch_size,
    train,
    new_size=None,
    height=256,
    width=256,
    num_workers=4,
    crop=True,
):
    """
    Masks and images lists-based data loader with transformations
    (horizontal flip, resizing, random crop, normalization are handled)

    Arguments:
        file_list {str list} -- list of images filenames
        mask_list {str list} -- list of masks filenames
        batch_size {int} -- batch size
        train {bool} -- training

    Keyword Arguments:
        new_size {int} -- parameter for resizing (default: {None})
        height {int} -- dimension for random cropping (default: {256})
        width {int} -- dimension for random cropping (default: {256})
        num_workers {int} -- number of workers (default: {4})
        crop {bool} -- crop(default: {True})

    Returns:
        loader -- data loader with transformed dataset
    """
    dataset = MyDataset(file_list, mask_list, new_size, height, width)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        num_workers=num_workers,
    )
    return loader


def get_data_loader_mask_and_im_and_rect(
    file_list,
    mask_list,
    rect_list,
    batch_size,
    train,
    new_size=None,
    height=256,
    width=256,
    num_workers=4,
    crop=True,
):
    """
    Masks and images lists-based data loader with transformations
    (horizontal flip, resizing, random crop, normalization are handled)

    Arguments:
        file_list {str list} -- list of images filenames
        mask_list {str list} -- list of masks filenames
        batch_size {int} -- batch size
        train {bool} -- training

    Keyword Arguments:
        new_size {int} -- parameter for resizing (default: {None})
        height {int} -- dimension for random cropping (default: {256})
        width {int} -- dimension for random cropping (default: {256})
        num_workers {int} -- number of workers (default: {4})
        crop {bool} -- crop(default: {True})

    Returns:
        loader -- data loader with transformed dataset
    """
    dataset = MyDatasetRect(file_list, mask_list, rect_list, new_size, height, width)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        num_workers=num_workers,
    )
    return loader


def get_data_loader_folder(
    input_folder, batch_size, train, new_size=None, height=256, width=256, num_workers=4, crop=True,
):
    """
    Folder-based data loader with transformations
     (horizontal flip, resizing, random crop, normalization are handled)

    Arguments:
        input_folder {str} -- path to folder with input images
        batch_size {int} -- batch size
        train {bool} -- training

    Keyword Arguments:
       new_size {int} -- parameter for resizing (default: {None})
        height {int} -- dimension for random cropping (default: {256})
        width {int} -- dimension for random cropping (default: {256})
        num_workers {int} -- number of workers (default: {4})
        crop {bool} -- crop(default: {True})

    Returns:
        loader -- data loader with transformed dataset

    Returns:
        [type] -- [description]
    """
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    transform_list = (
        [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    )
    transform_list = (
        [transforms.Resize((new_size, new_size))] + transform_list
        if new_size is not None
        else transform_list
    )
    transform_list = (
        [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    )
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        num_workers=num_workers,
    )
    return loader


def get_config(config):
    """Parse config yaml file

    Arguments:
        config {str} -- path to yaml config file

    Returns:
        dict -- parsed yaml file
    """
    with open(config, "r") as stream:
        return yaml.safe_load(stream)


def eformat(f, prec):
    s = "%.*e" % (prec, f)
    mantissa, exp = s.split("e")
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%d" % (mantissa, int(exp))


def __write_images(image_outputs, display_image_num, file_name):
    """Save output image
    Arguments:
        image_outputs {Tensor list} -- list of output images
        display_image_num {int} -- number of images to be displayed
        file_name {str} -- name of the file where to save the images
    """
    image_outputs = [
        images.expand(-1, 3, -1, -1) for images in image_outputs
    ]  # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(
        image_tensor.data, nrow=display_image_num, padding=0, normalize=True
    )
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix, comet_exp=None):
    """Write images from both worlds a and b of the cycle  A-B-A as jpg
    Arguments:
        image_outputs {Tensor list} -- list of images, the first half being outputs in B,
                                        the second half being outputs in A
        display_image_num {int} -- number of images to be displayed
        image_directory {str} --
        postfix {str} -- postfix to filename

    Keyword Arguments:
        comet_exp {Comet experience} --  (default: {None})
    """
    n = len(image_outputs)
    __write_images(
        image_outputs[0 : n // 2],
        display_image_num,
        "%s/gen_a2b_%s.jpg" % (image_directory, postfix),
    )
    __write_images(
        image_outputs[n // 2 : n],
        display_image_num,
        "%s/gen_b2a_%s.jpg" % (image_directory, postfix),
    )
    if comet_exp is not None:
        comet_exp.log_image("%s/gen_a2b_%s.jpg" % (image_directory, postfix))
        comet_exp.log_image("%s/gen_b2a_%s.jpg" % (image_directory, postfix))


def prepare_sub_folder(output_directory):
    """Create images and checkpoints subfolders in output directory

    Arguments:
        output_directory {str} -- output directory

    Returns:
        checkpoint_directory, image_directory-- checkpoints and images directories
    """
    image_directory = os.path.join(output_directory, "images")
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, "checkpoints")
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_loss(iterations, trainer, train_writer):
    members = [
        attr
        for attr in dir(trainer)
        if not callable(getattr(trainer, attr))
        and not attr.startswith("__")
        and ("loss" in attr or "grad" in attr or "nwd" in attr)
    ]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def slerp(val, low, high):
    """
    Spherical linear interpolation (slerp)
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    Arguments:
        val {float} -- mean in Gaussian prior
        low {float} -- smallest value in the interpolation
        high {float} -- highest value in the interpolation
    Returns:
        slerp value
    """
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def get_slerp_interp(nb_latents, nb_interp, z_dim):
    """
    modified from: PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot
    https://github.com/ptrblck/prog_gans_pytorch_inference
    """

    latent_interps = np.empty(shape=(0, z_dim), dtype=np.float32)
    for _ in range(nb_latents):
        low = np.random.randn(z_dim)
        high = np.random.randn(z_dim)  # low + np.random.randn(512) * 0.7
        interp_vals = np.linspace(0, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals], dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))

    return latent_interps[:, :, np.newaxis, np.newaxis]


# Get model list for resume
def get_model_list(dirname, key):
    """get last model in dirname, whose name contain key

    Arguments:
        dirname {str} -- directory name
        key {str} -- "key" in the model name

    Returns:
        last_model_name {str} -- last model name
    """
    if os.path.exists(dirname) is False:
        return None
    gen_models = [
        os.path.join(dirname, f)
        for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f
    ]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def load_vgg16(model_dir):
    raise NotImplementedError(
        "This function relies on torch.utils.serialization.load_lua which is deprecated"
    )


def load_flood_classifier(ckpt_path):
    """ Load flood classifier based on a pretrained resnet18 network.

    Arguments:
        ckpt_path {str} -- path to checkpoint

    Returns:
        model -- flood classifier model
    """
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(ckpt_path))
    return model


class Resnet34_8s(nn.Module):
    def __init__(self, num_classes=1000):
        super(Resnet34_8s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_8s = resnet34(
            fully_conv=True, pretrained=True, output_stride=8, remove_avg_pool_layer=True,
        )

        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_8s.fc = nn.Conv2d(resnet34_8s.inplanes, num_classes, 1)

        self.resnet34_8s = resnet34_8s

        self._normal_initialization(self.resnet34_8s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        input_spatial_dim = x.size()[2:]

        if feature_alignment:
            x = adjust_input_image_size_for_proper_feature_alignment(x, output_stride=8)

        x = self.resnet34_8s(x)

        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        # x = nn.functional.interpolate(input=x, size=input_spatial_dim, mode="bilinear")
        print("#######################################################################################")

        return x


def load_segmentation_model(ckpt_path, classes):
    """load Resnet34 segmentation model with output stride 8 from checkpoint

    Arguments:
        ckpt_path {str} -- checkpoint path

    Returns:
        model -- segmentation model
    """
    model = Resnet34_8s(num_classes=classes).to("cuda")
    model.load_state_dict(torch.load(ckpt_path))
    return model


# Define the helper function
def decode_segmap(image, nc=19):
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.
    Arguments:
        image {array} -- segmented image
        (array of image size containing classat each pixel)
    Returns:
        array of size 3*nc -- A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((19, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(nc):
        idx = image == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def load_inception(model_path):
    """Load Inception model

    Arguments:
        model_path {str} -- model path

    Returns:
        model -- Inception model
    """
    state_dict = torch.load(model_path)
    model = inception_v3(pretrained=False, transform_input=True)
    model.aux_logits = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, state_dict["fc.weight"].size(0))
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False
    return model


def vgg_preprocess(batch):
    """Preprocess batch to use VGG model
    """
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean))  # subtract mean
    return batch


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    """Returns a learning rate scheduler such that the learning rate of each parameter group is set to the initial
    lr decayed by hyperparameter gamma every step_size epochs when a learning rate policy is specified in the hyperparameters.
    When iterations=-1, sets initial lr as lr.
    Arguments:
        optimizer {Optimizer} -- Wrapped optimizer
        hyperparameters {} -- Hyperparameters parsed from config yaml file

    Keyword Arguments:
        iterations {int} -- index of the last epoch (default: {-1})
    """
    if "lr_policy" not in hyperparameters or hyperparameters["lr_policy"] == "constant":
        scheduler = None  # constant scheduler
    elif hyperparameters["lr_policy"] == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=hyperparameters["step_size"],
            gamma=hyperparameters["gamma"],
            last_epoch=iterations,
        )
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", hyperparameters["lr_policy"]
        )
    return scheduler


def weights_init(init_type="gaussian"):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find("Conv") == 0 or classname.find("Linear") == 0) and hasattr(m, "weight"):
            # print m.__class__.__name__
            if init_type == "gaussian":
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == "default":
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def pytorch03_to_pytorch04(state_dict_base, trainer_name):
    def __conversion_core(state_dict_base, trainer_name):
        state_dict = state_dict_base.copy()
        if trainer_name == "MUNIT":
            for key, value in state_dict_base.items():
                if key.endswith(
                    (
                        "enc_content.model.0.norm.running_mean",
                        "enc_content.model.0.norm.running_var",
                        "enc_content.model.1.norm.running_mean",
                        "enc_content.model.1.norm.running_var",
                        "enc_content.model.2.norm.running_mean",
                        "enc_content.model.2.norm.running_var",
                        "enc_content.model.3.model.0.model.1.norm.running_mean",
                        "enc_content.model.3.model.0.model.1.norm.running_var",
                        "enc_content.model.3.model.0.model.0.norm.running_mean",
                        "enc_content.model.3.model.0.model.0.norm.running_var",
                        "enc_content.model.3.model.1.model.1.norm.running_mean",
                        "enc_content.model.3.model.1.model.1.norm.running_var",
                        "enc_content.model.3.model.1.model.0.norm.running_mean",
                        "enc_content.model.3.model.1.model.0.norm.running_var",
                        "enc_content.model.3.model.2.model.1.norm.running_mean",
                        "enc_content.model.3.model.2.model.1.norm.running_var",
                        "enc_content.model.3.model.2.model.0.norm.running_mean",
                        "enc_content.model.3.model.2.model.0.norm.running_var",
                        "enc_content.model.3.model.3.model.1.norm.running_mean",
                        "enc_content.model.3.model.3.model.1.norm.running_var",
                        "enc_content.model.3.model.3.model.0.norm.running_mean",
                        "enc_content.model.3.model.3.model.0.norm.running_var",
                    )
                ):
                    del state_dict[key]
        else:

            def __conversion_core(state_dict_base):
                state_dict = state_dict_base.copy()
                for key, value in state_dict_base.items():
                    if key.endswith(
                        (
                            "enc.model.0.norm.running_mean",
                            "enc.model.0.norm.running_var",
                            "enc.model.1.norm.running_mean",
                            "enc.model.1.norm.running_var",
                            "enc.model.2.norm.running_mean",
                            "enc.model.2.norm.running_var",
                            "enc.model.3.model.0.model.1.norm.running_mean",
                            "enc.model.3.model.0.model.1.norm.running_var",
                            "enc.model.3.model.0.model.0.norm.running_mean",
                            "enc.model.3.model.0.model.0.norm.running_var",
                            "enc.model.3.model.1.model.1.norm.running_mean",
                            "enc.model.3.model.1.model.1.norm.running_var",
                            "enc.model.3.model.1.model.0.norm.running_mean",
                            "enc.model.3.model.1.model.0.norm.running_var",
                            "enc.model.3.model.2.model.1.norm.running_mean",
                            "enc.model.3.model.2.model.1.norm.running_var",
                            "enc.model.3.model.2.model.0.norm.running_mean",
                            "enc.model.3.model.2.model.0.norm.running_var",
                            "enc.model.3.model.3.model.1.norm.running_mean",
                            "enc.model.3.model.3.model.1.norm.running_var",
                            "enc.model.3.model.3.model.0.norm.running_mean",
                            "enc.model.3.model.3.model.0.norm.running_var",
                            "dec.model.0.model.0.model.1.norm.running_mean",
                            "dec.model.0.model.0.model.1.norm.running_var",
                            "dec.model.0.model.0.model.0.norm.running_mean",
                            "dec.model.0.model.0.model.0.norm.running_var",
                            "dec.model.0.model.1.model.1.norm.running_mean",
                            "dec.model.0.model.1.model.1.norm.running_var",
                            "dec.model.0.model.1.model.0.norm.running_mean",
                            "dec.model.0.model.1.model.0.norm.running_var",
                            "dec.model.0.model.2.model.1.norm.running_mean",
                            "dec.model.0.model.2.model.1.norm.running_var",
                            "dec.model.0.model.2.model.0.norm.running_mean",
                            "dec.model.0.model.2.model.0.norm.running_var",
                            "dec.model.0.model.3.model.1.norm.running_mean",
                            "dec.model.0.model.3.model.1.norm.running_var",
                            "dec.model.0.model.3.model.0.norm.running_mean",
                            "dec.model.0.model.3.model.0.norm.running_var",
                        )
                    ):
                        del state_dict[key]

        return state_dict

    state_dict = dict()
    state_dict["a"] = __conversion_core(state_dict_base["a"], trainer_name)
    state_dict["b"] = __conversion_core(state_dict_base["b"], trainer_name)
    return state_dict


# Domain adversarial loss (define a classifier on top of the feature extracted)
def conv_block(in_channels, out_channels):
    """returns a block Convolution - batch normalization - ReLU - Pooling

    Arguments:
        in_channels {int} -- Number of channels in the input image
        out_channels {int} -- Number of channels produced by the convolution

    Returns:
        block -- Convolution - batch normalization - ReLU - Pooling
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding

    Arguments:
        in_planes {int} -- Number of channels in the input image
        out_planes {int} -- Number of channels produced by the convolution

    Keyword Arguments:
        stride {int or tuple, optional} -- Stride of the convolution. Default: 1 (default: {1})
        groups {int, optional} -- Number of blocked connections from input channels to output channels.tion] (default: {1})
        dilation {int or tuple, optional} -- Spacing between kernel elements (default: {1})

    Returns:
        output layer of 3x3 convolution with padding
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution
    Arguments:
        in_planes {int} -- Number of channels in the input image
        out_planes {int} -- Number of channels produced by the convolution

    Keyword Arguments:
        stride {int or tuple, optional} -- Stride of the convolution. Default: 1 (default: {1})
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride
        self.downsample = downsample
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(conv1x1(inplanes, planes, stride), norm_layer(planes))

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def merge_classes(output):

    merged = torch.zeros(output.shape[0], 10, output.shape[2], output.shape[2])
    dic = {
        9: [14, 15, 16],
        8: [13, 17, 18],
        7: [11, 12],
        6: [10],
        5: [9],
        4: [8],
        3: [5, 6, 7],
        2: [2, 3, 4],
        1: [0, 1],
        0: [],
    }

    for key in dic:
        d = dic[key]
        if len(d) == 0:
            continue

        merged[:, key] = output[:, d].sum(dim=1)

    return merged


def mapping(im):
    im[im == 255] = 8
    im[im == 200] = 7
    im[im == 178] = 6
    im[im == 149] = 5
    im[im == 133] = 4
    im[im == 76] = 3
    im[im == 55] = 2
    im[im == 29] = 1
    im[im == 0] = 0
    return im


# Define the encoded
class domainClassifier(nn.Module):
    def __init__(self, input_dim, dim):
        super(domainClassifier, self).__init__()

        self.max_pool1 = nn.MaxPool2d(2)
        self.BasicBlock1 = BasicBlock(input_dim, 128, True)
        self.max_pool2 = nn.MaxPool2d(2)
        self.BasicBlock2 = BasicBlock(128, 64, True)
        self.avg_pool = nn.AvgPool2d((8, 8))
        self.fc = nn.Linear(64, 1)
        self.output_dim = dim

    def forward(self, x):
        max_pooled1 = self.max_pool1(x)
        res_block1 = self.BasicBlock1(max_pooled1)
        max_pooled2 = self.max_pool2(res_block1)
        res_block2 = self.BasicBlock2(max_pooled2)
        avg_pool = self.avg_pool(res_block2)
        fc_output = self.fc(avg_pool.squeeze())
        # print(fc_output.shape)
        # logits = nn.functional.softmax(fc_output)
        # return logits
        return fc_output


def flatten_opts(opts):
    """Flattens a multi-level addict.Dict or native dictionnary into a single
    level native dict with string keys representing the keys sequence to reach
    a value in the original argument.

    d = addict.Dict()
    d.a.b.c = 2
    d.a.b.d = 3
    d.a.e = 4
    d.f = 5
    flatten_opts(d)
    >>> {
        "a.b.c": 2,
        "a.b.d": 3,
        "a.e": 4,
        "f": 5,
    }

    Args:
        opts (addict.Dict or dict): addict dictionnary to flatten

    Returns:
        dict: flattened dictionnary
    """
    values_list = []

    def p(d, prefix="", vals=[]):
        for k, v in d.items():
            if isinstance(v, dict):
                p(v, prefix + k + ".", vals)
            elif isinstance(v, list):
                if isinstance(v[0], dict):
                    for i, m in enumerate(v):
                        p(m, prefix + k + "." + str(i) + ".", vals)
                else:
                    vals.append((prefix + k, str(v)))
            else:
                vals.append((prefix + k, v))

    p(opts, vals=values_list)
    return dict(values_list)


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def load_opts(path=None, default=None):
    """Loads a configuration Dict from 2 files:
    1. default files with shared values across runs and users
    2. an overriding file with run- and user-specific values
    Args:
        path (pathlib.Path): where to find the overriding configuration
            default (pathlib.Path, optional): Where to find the default opts.
            Defaults to None. In which case it is assumed to be a default config
            which needs processing such as setting default values for lambdas and gen
            fields
    Returns:
        addict.Dict: options dictionnary, with overwritten default values
    """
    if default is None:
        default_opts = Dict()
    else:
        with open(default, "r") as f:
            default_opts = Dict(yaml.safe_load(f))

    with open(path, "r") as f:
        overriding_opts = Dict(yaml.safe_load(f))

    default_opts.update(overriding_opts)

    return default_opts


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    # batch = batch.div_(255.0)
    return (batch - mean) / std


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

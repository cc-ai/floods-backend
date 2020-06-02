import torch, os, tqdm, cv2
import matplotlib.pyplot as plt
from scipy.misc import imresize
from flask import jsonify
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2 as cv
import math
from ccai.config import CONFIG
from ccai.nn.model.segmentation import MyDataset


def cuda_check(MODEL):
    if torch.cuda.is_available():
        MODEL_STATE_DICT = torch.load(CONFIG.MODEL_CHECKPOINT_FILE)
    else:
        MODEL_STATE_DICT = torch.load(CONFIG.MODEL_CHECKPOINT_FILE, map_location={"cuda:0": "cpu"})

    MODEL.gen.load_state_dict(MODEL_STATE_DICT["2"])

    if torch.cuda.is_available():
        MODEL.cuda()

    MODEL.eval()


def model_validation(ROUTE_MODEL, VALID_MODELS):
    if ROUTE_MODEL.lower() not in VALID_MODELS:
        response = jsonify({"error": "Invalid model", "valid_models": VALID_MODELS})
        response.status_code = 400
        return response


def model_launch(MODEL, MODEL_NEW_SIZE, MASK_MODEL, temp_dir, path_to_gsv_image):

    path_to_flooded_image = model_spade(
        MODEL, MODEL_NEW_SIZE, MASK_MODEL, temp_dir, path_to_gsv_image
    )

    return path_to_flooded_image


def model_spade(MODEL, MODEL_NEW_SIZE, MASK_MODEL, temp_dir, path_to_gsv_image):
    with torch.no_grad():
        # Define the transform to infer with the generator
        transform = transforms.Compose(
            [
                # transforms.Resize((new_size, new_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        mask_transform = transforms.Compose([transforms.ToTensor()])

        # Define image path
        path_xa = path_to_gsv_image

        # Mask stuff
        model_deeplab(temp_dir, MASK_MODEL, MODEL_NEW_SIZE)
        mask = Image.open(temp_dir + "/gsv_0.png")
        mask = Variable(mask_transform(mask).cuda())

        # Make mask binary
        mask_thresh = (torch.max(mask) - torch.min(mask)) / 2.0
        mask = (mask > mask_thresh).float()
        mask = mask[0].unsqueeze(0).unsqueeze(0)

        # Load and transform the non_flooded image
        x_a = Variable(transform(Image.open(path_xa).convert("RGB")).unsqueeze(0).cuda())

        # MASK SMOOTHING
        mask = mask.squeeze().detach().cpu().numpy()
        mask = cv.normalize(
            mask, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F
        )
        mask = mask.astype(dtype=np.uint8)
        ret, thresh = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Find largest contour
        max_area = 0
        max_idx = 0
        for j, cnt in enumerate(contours):
            if cv.contourArea(contours[j]) > max_area:  # just a condition
                max_idx = j
                max_area = cv.contourArea(contours[j])
        hyp_length = math.sqrt(mask.shape[-2] ** 2 + mask.shape[-1] ** 2)
        cnt = contours[max_idx]
        smooth_mask = np.zeros(mask.shape)
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                dist = cv.pointPolygonTest(cnt, (y, x), True)
                norm_dist = dist / hyp_length
                if norm_dist < 0:
                    norm_dist = -norm_dist
                    mask_value = int(255 * math.exp(-400 * norm_dist))
                    smooth_mask[x, y] = mask_value
        smooth_mask = smooth_mask + mask
        smooth_mask = torch.tensor(smooth_mask, device="cuda").float()
        smooth_mask = smooth_mask.unsqueeze(0) / 255.0
        mask = smooth_mask
        # ----------------------------------------------------------

        latent_size1 = MODEL_NEW_SIZE // (2 ** CONFIG.model_config["gen"]["n_downsample"])

        latent_size2 = mask.shape[-1] // (2 ** CONFIG.model_config["gen"]["n_downsample"])

        z = (
            torch.empty(1, CONFIG.model_config["gen"]["dim"], latent_size1, latent_size2)
            .normal_(mean=0, std=1.0)
            .cuda()
        )
        x_a_masked = x_a * (1.0 - mask)

        x_ab = MODEL.gen(z, x_a_masked)

        # Denormalize .Normalize(0.5,0.5,0.5)...
        outputs = (x_ab + 1) / 2.0

        path_to_flooded_image = os.path.join(temp_dir, "output" + "{:03d}.jpg".format(0))
        vutils.save_image(outputs.data, path_to_flooded_image, padding=0, normalize=True)

    return path_to_flooded_image


def model_deeplab(temp_dir, MASK_MODEL, MODEL_NEW_SIZE):
    size_mask_1 = 600
    size_mask = (int(size_mask_1), int(size_mask_1))
    temp_dir = temp_dir + "/"
    path_to_gsv_image = os.path.join(temp_dir, "gsv_0.jpg")
    valid_transform = transforms.Compose(
        [
            transforms.Resize(size_mask),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    val_ds = MyDataset(temp_dir, transform=valid_transform)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i_batch, images in tqdm.tqdm(enumerate(val_dl)):
            imgs = images.to("cuda")
            out_batch = MASK_MODEL(imgs.float()).cpu()
            batch_size_ = len(out_batch)
            for j in range(batch_size_):
                out1 = out_batch[j, :, :, :]
                res = out1.squeeze(0).max(0)
                original_image = plt.imread(path_to_gsv_image)
                original_image = imresize(original_image, size=size_mask)
                patches = []
                mask_flat = np.zeros(size_mask)
                for p in [0, 1, 9]:
                    mask_flat[np.where(res[1] == p)] = 255
                # Sasha: to help with the multi-channel jpg masks
                filename = path_to_gsv_image
                extension_length = len(filename.split(".")[-1])
                filename_png = filename[: -extension_length - 1] + ".png"
                cv2.imwrite(filename_png, mask_flat.astype(int))
                # cv2.imwrite(dir_mask+list_paths[it_*batch_size+j],mask_flat.astype(int))

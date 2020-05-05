import torch, os, tqdm, cv2
import matplotlib.pyplot as plt
from scipy.misc import imresize
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import numpy as np
from ccai.config import CONFIG
from ccai.nn.spade.segmentation import MyDataset

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

def model_launch(MODEL, ROUTE_MODEL, MODEL_NEW_SIZE, MASK_MODEL, temp_dir, path_to_gsv_image):
    if ROUTE_MODEL is "munit":
        path_to_flooded_image = model_munit(MODEL, MODEL_NEW_SIZE, temp_dir, path_to_gsv_image)

    elif ROUTE_MODEL is "spade":
        path_to_flooded_image = model_spade(MODEL, MODEL_NEW_SIZE, MASK_MODEL, temp_dir, path_to_gsv_image)

    else:
        path_to_gsv_image = model_spade(MODEL, MODEL_NEW_SIZE, temp_dir, path_to_gsv_image)
        path_to_flooded_image = model_munit(MODEL, MODEL_NEW_SIZE, temp_dir, path_to_gsv_image)

    return path_to_flooded_image

def model_munit(MODEL, MODEL_NEW_SIZE, temp_dir, path_to_gsv_image):
    with torch.no_grad():

        transform = transforms.Compose(
            [
                transforms.Resize(MODEL_NEW_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        image_transformed = transform(
            Image.open(path_to_gsv_image).convert("RGB")
        ).unsqueeze(0)

        if torch.cuda.is_available():
            image_transformed = image_transformed.cuda()

        x_a = image_transformed
        c_xa_b, _ = MODEL.gen.encode(x_a, 1)
        content = c_xa_b
        style_data = np.load(CONFIG.MODEL_STYLE_FILE)
        style = torch.tensor(style_data.reshape(1, 16, 1, 1), dtype=torch.float)
        if torch.cuda.is_available():
            style = style.cuda()

        outputs = MODEL.gen.decode(content, style, 2)
        outputs = (outputs + 1) / 2.0

        path_to_flooded_image = os.path.join(temp_dir, "output" + "{:03d}.jpg".format(0))
        vutils.save_image(outputs.data, path_to_flooded_image, padding=0, normalize=True)

    return path_to_flooded_image

def model_spade(MODEL, MODEL_NEW_SIZE, MASK_MODEL, temp_dir, path_to_gsv_image):
    with torch.no_grad():

        transform = transforms.Compose(
            [
                transforms.Resize((MODEL_NEW_SIZE, MODEL_NEW_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        mask_transform = transforms.Compose(
            [transforms.Resize((MODEL_NEW_SIZE, MODEL_NEW_SIZE)), transforms.ToTensor(), ]
        )

        path_xa = path_to_gsv_image

        # mask = Image.open(CONFIG.MODEL_MASK_FILE)

        #DeepLab
        model_deeplab(temp_dir, MASK_MODEL,MODEL_NEW_SIZE)
        mask = Image.open(temp_dir + "/gsv_0.png")

        # process
        mask = Variable(mask_transform(mask).cuda())
        mask_thresh = (torch.max(mask) - torch.min(mask)) / 2.0
        mask = (mask > mask_thresh).float()
        mask = mask[0].unsqueeze(0).unsqueeze(0)

        # Load and transform the non_flooded image
        x_a = Variable(transform(Image.open(path_xa).convert("RGB")).unsqueeze(0).cuda())
        x_a_augment = torch.cat([x_a, mask], dim=1)
        c_a = MODEL.gen.encode(x_a_augment, 1)

        # Perform cross domain translation
        x_ab = MODEL.gen.decode(c_a, mask, 2)

        # Denormalize .Normalize(0.5,0.5,0.5)...
        outputs = (x_ab + 1) / 2.0

        # Define output path
        path_to_flooded_image = os.path.join(temp_dir, "output" + "{:03d}.jpg".format(0))
        vutils.save_image(outputs.data, path_to_flooded_image, padding=0, normalize=True)

        return path_to_flooded_image

def model_deeplab(temp_dir, MASK_MODEL, MODEL_NEW_SIZE):
    size_mask_1 = 600
    size_mask = (int(size_mask_1), int(size_mask_1))
    temp_dir = temp_dir +"/"
    path_to_gsv_image = os.path.join(temp_dir, "gsv_0.jpg")
    valid_transform = transforms.Compose(
                    [
                         transforms.Resize(size_mask),
                         transforms.ToTensor(),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])

    # valid_transform = transforms.Compose(
    #     [transforms.Resize((MODEL_NEW_SIZE, MODEL_NEW_SIZE)), transforms.ToTensor(), ]
    # )

    val_ds=MyDataset(temp_dir,transform = valid_transform)
    val_dl=torch.utils.data.DataLoader(val_ds,batch_size=1,shuffle=False)

    with torch.no_grad():
        for i_batch, images in tqdm.tqdm(enumerate(val_dl)):
            imgs =images.to('cuda')
            out_batch= MASK_MODEL(imgs.float()).cpu()
            batch_size_ = len(out_batch)
            for j in range(batch_size_):
              out1=out_batch[j,:,:,:]
              res=out1.squeeze(0).max(0)
              original_image=plt.imread(path_to_gsv_image)
              original_image = imresize(original_image,size=size_mask)
              patches =[]
              mask_flat=np.zeros(size_mask)
              for p in [0,1,9]:
                mask_flat[np.where(res[1]==p)]=255
              # Sasha: to help with the multi-channel jpg masks
              filename = path_to_gsv_image
              extension_length = len(filename.split('.')[-1])
              filename_png = filename[:-extension_length-1]+'.png'
              cv2.imwrite(filename_png,mask_flat.astype(int))
              #cv2.imwrite(dir_mask+list_paths[it_*batch_size+j],mask_flat.astype(int))




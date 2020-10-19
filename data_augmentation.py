import os
import cv2
from tqdm import tqdm
from glob import glob
from albumentations import CenterCrop, RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip

def load_data(path):
    images = sorted(glob(os.path.join(path, "images_43_norm/*")))
    masks = sorted(glob(os.path.join(path, "masks_43/*")))
    return images, masks

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def augment_data(images, masks, save_path1, save_path2, augment=True):
    H = 1024  #256
    W = 1024  #256

    for x, y in tqdm(zip(images, masks), total=len(images)):
        name = x.split("/")[-1].split(".")
        """ Extracting the name and extension of the image and the mask. """
        image_name = name[0]
        image_extn = name[1]

        name = y.split("/")[-1].split(".")
        mask_name = name[0]
        mask_extn = name[1]

        """ Reading image and mask. """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        """ Augmentation """
        if augment == True:
            aug = CenterCrop(H, W, p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = RandomRotate90(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = GridDistortion(p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            save_images = [x, x1, x2, x3, x4, x5]
            save_masks =  [y, y1, y2, y3, y4, y5]

        else:
            save_images = [x]
            save_masks = [y]

        """ Saving the image and mask. """
        idx = 0
        for i, m in zip(save_images, save_masks):
            i = cv2.resize(i, (W, H))
            m = cv2.resize(m, (W, H))

            if len(images) == 1:
                tmp_img_name = f"{image_name}.{image_extn}"
                tmp_mask_name = f"{mask_name}.{mask_extn}"
            else:
                tmp_img_name = f"{image_name}_{idx}.{image_extn}"
                tmp_mask_name = f"{mask_name}_{idx}.{mask_extn}"

            num= tmp_img_name.split(".")[0]
            n = int(num.split("_")[0])
            p = int(num.split("_")[1])
            cifra1= 6*(n-1)+p+1
            tmp_img_name = f"{cifra1}.{image_extn}" #para que 1_2 sea 3 y asÃ­
            

            num= tmp_mask_name.split(".")[0]
            a = int(num.split("_")[0])
            b = int(num.split("_")[1])
            
            cifra2= 6*(a-1)+b+1
            tmp_mask_name = f"{cifra2}.{mask_extn}"

            image_path = os.path.join(save_path1, tmp_img_name)
            mask_path = os.path.join(save_path2, tmp_mask_name)
            

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)


            idx += 1
        

if __name__ == "__main__":
    """ Loading original images and masks. """
    path = "BreCaHAD/"
    images, masks = load_data(path)
    print(f"Original Images: {len(images)} - Original Masks: {len(masks)}")

    """ Creating folders. """
    create_dir("Data/images")
    create_dir("Data/masks")

    """ Applying data augmentation. """
    augment_data(images, masks, "Data/images","Data/masks", augment=True)

    """ Loading augmented images and masks. """
    images, masks = load_data("Data/")
    print(f"Augmented Images: {len(images)} - Augmented Masks: {len(masks)}")

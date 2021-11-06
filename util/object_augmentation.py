from dataset import *
import matplotlib.pyplot as plt
from random import *
import cv2
import os

dataset_path = "/opt/ml/segmentation/input/data"
train_path = dataset_path + '/train.json'

# "Background":0, "General trash":1, "Paper":2, "Paper pack":3, "Metal":4, "Glass":5, "Plastic":6, "Styrofoam":7, "Plastic bag":8, "Battery":9, "Clothing":10)

train_dataset = CustomDataset(dataset_path, train_path, mode='train', transform=None)

cut_data_idx = 10  # 잘라붙여 증가시킬 class
cut_dataset = []
other_dataset = []
for i in range(len(train_dataset)):
    if cut_data_idx in set(train_dataset[i][1].flatten()):
        cut_dataset.append(train_dataset[i])
    else:
        other_dataset.append(train_dataset[i])

print(f"count : {len(cut_dataset)}")
for cut_data in cut_dataset:
    print(set(cut_data[1].flatten()))

# train_loader, val_loader = loadDataLoader("/opt/ml/segmentation/input/data")

def merge_image(insert_image, insert_mask, class_id, base_image=None, base_mask=None):
    """
    Args:
        insert_image: 추출하려고 하는 이미지
        insert_mask: 추출하려고 하는 이미지의 마스크
        class_id: 목적 class
        base_image: 배경이 되는 이미지
        base_mask: 배경이 되는 이미지의 마스크
    """
    tmp_img = np.ones((512,512,3), dtype=np.float32) * 255
    if type(base_image) is type(None):
        base_image = tmp_img.copy()
    tmp_img[:,:,0] = np.where(insert_mask == class_id, insert_image[:,:,2], base_image[:,:,2])  # R or B
    tmp_img[:,:,1] = np.where(insert_mask == class_id, insert_image[:,:,1], base_image[:,:,1])  # G
    tmp_img[:,:,2] = np.where(insert_mask == class_id, insert_image[:,:,0], base_image[:,:,0])  # B or R
    if type(base_mask) is type(None):
        base_mask = np.zeros((512,512), dtype=np.uint8)
    tmp_mask = np.where(insert_mask == class_id, insert_mask, base_mask)  # mask
    return tmp_img, tmp_mask.astype(np.uint8)

image1 = cut_dataset[1][0]  # 5, 6
mask1 = cut_dataset[1][1]
image2 = other_dataset[0][0]
mask2 = other_dataset[0][1]

tmp_img, tmp_mask = merge_image(image1, mask1, cut_data_idx)
fig, axes = plt.subplots(2, 2, figsize=(6, 6), dpi=150)
axes[0, 0].imshow(image1)
axes[0, 0].axis("off")
axes[0, 1].imshow(mask1)
axes[0, 1].axis("off")
axes[1, 0].imshow(tmp_img)
axes[1, 0].axis("off")
axes[1, 1].imshow(tmp_mask)
axes[1, 1].axis("off")

tmp_img, tmp_mask = merge_image(image1, mask1, cut_data_idx, base_image=image2, base_mask=mask2)
fig, axes = plt.subplots(2, 2, figsize=(6, 6), dpi=150)
axes[0, 0].imshow(image2)
axes[0, 0].axis("off")
axes[0, 1].imshow(mask2)
axes[0, 1].axis("off")
axes[1, 0].imshow(tmp_img)
axes[1, 0].axis("off")
axes[1, 1].imshow(tmp_mask)
axes[1, 1].axis("off")

new_images = []
new_masks = []
max_idx = len(other_dataset)
for cut_image_idx in range(len(cut_dataset)):
    for i in range(5):
        base_image_idx = randint(0, max_idx)
        tmp_img, tmp_mask = merge_image(
            cut_dataset[cut_image_idx][0],
            cut_dataset[cut_image_idx][1],
            cut_data_idx,
            base_image=other_dataset[base_image_idx][0],
            base_mask=other_dataset[base_image_idx][1]
        )
        new_images.append(tmp_img)
        new_masks.append(tmp_mask)

fig, axes = plt.subplots(2, 2, figsize=(6, 6), dpi=150)
axes[0, 0].imshow(new_images[0])
axes[0, 0].axis("off")
axes[0, 1].imshow(new_masks[0])
axes[0, 1].axis("off")
axes[1, 0].imshow(new_images[12])
axes[1, 0].axis("off")
axes[1, 1].imshow(new_masks[12])
axes[1, 1].axis("off")

save_dir = "/opt/ml/segmentation/input/custom_data/image/clothes_train5_rgb/"
mask_dir = "/opt/ml/segmentation/input/custom_data/mask/clothes_train5_rgb/"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
if not os.path.isdir(mask_dir):
    os.makedirs(mask_dir)

for image_idx, image in enumerate(new_images):
    file_dir = save_dir+f"{image_idx:04}.jpg"
    cv2.imwrite(file_dir, image*255)

for image_idx, image in enumerate(new_masks):
    file_dir = mask_dir+f"{image_idx:04}.png"
    cv2.imwrite(file_dir, image)

# import json
# with open('train.json') as f:
#     json_object = json.load(f)

# for mask_idx, mask in enumerate(new_masks):
#     tmp_image = {
#                 "license": 0,
#                 "url": None,
#                 "file_name": f"image/{mask_idx:04}.jpg",
#                 "height": 512,
#                 "width": 512,
#                 "date_captured": None,
#                 "id": 0
#             }
#     tmp_segmentation = mask

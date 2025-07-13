import os
import shutil
import random
from tqdm import tqdm

# Set seed for reproducibility
random.seed(42)

# Input and output directories
input_dir = "downloaded_faces"
output_dir = "downloaded_faces_split"
splits = ['training_images', 'validation_images', 'testing_images']
ratios = [0.7, 0.15, 0.15]  # train/val/test

# Create folders
for split in splits:
    for class_name in os.listdir(input_dir):
        os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

# Split data
for class_name in tqdm(os.listdir(input_dir), desc="Processing classes"):
    class_dir = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    images = os.listdir(class_dir)
    random.shuffle(images)

    train_split = int(ratios[0] * len(images))
    val_split = int(ratios[1] * len(images))

    train_imgs = images[:train_split]
    val_imgs = images[train_split:train_split + val_split]
    test_imgs = images[train_split + val_split:]

    for img_list, split in zip([train_imgs, val_imgs, test_imgs], splits):
        for img in img_list:
            src = os.path.join(class_dir, img)
            dst = os.path.join(output_dir, split, class_name, img)
            shutil.copyfile(src, dst)

print("Done splitting into train/val/test.")

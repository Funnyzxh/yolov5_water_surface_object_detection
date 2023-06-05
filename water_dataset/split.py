import random
import glob
import os
import shutil


def copyfiles(fil,folder):
    basename = os.path.basename(fil)
    filename = os.path.splitext(basename)[0]

    # copy image
    src = fil
    dest = os.path.join(image_dir, folder,f"{filename}.jpg")
    shutil.copyfile(src, dest)

    # copy annotations
    src = os.path.join(label_dir, f"{filename}.txt")
    dest = os.path.join(label_dir, folder, f"{filename}.txt")
    if os.path.exists(src):
        shutil.copyfile(src, dest)


label_dir = "labels"
image_dir = "images"
lower_limit = 0
files = glob.glob(os.path.join(image_dir, '*.jpg'))

random.shuffle(files)

folders = {"train": 0.8, "val": 0.2}
check_sum = sum([folders[x] for x in folders])

assert check_sum == 1.0, "Split proportion is not equal to 1.0"

for folder in folders:
    image_folder = os.path.join(image_dir, folder)
    label_folder = os.path.join(label_dir, folder)
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)
    if not os.path.exists(label_folder):
        os.mkdir(label_folder)
    limit = round(len(files) * folders[folder])
    for fil in files[lower_limit:lower_limit + limit]:
        copyfiles(fil, folder)
    lower_limit = lower_limit + limit
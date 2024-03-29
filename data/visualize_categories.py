"""
https://matplotlib.org/stable/gallery/statistics/histogram_multihist.html
"""

import os
import cv2
import json
import pathlib
import pickle
import numpy as np
from pylabel import importer
from ultralytics.data.converter import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def get_one_image(images):
    # img_list = []
    # for img in images:
    #     img_list.append(cv2.imread(img))
    img_list = images
    max_width = []
    max_height = 0
    padding = 200
    for img in img_list:
        max_width.append(img.shape[0])
        max_height += img.shape[1]
    w = np.max(max_width)
    h = max_height + padding

    # create a new array with a size large enough to contain all the images
    final_image = np.zeros((h, w, 3), dtype=np.uint8)

    current_y = 0  # keep track of where your current image was last placed in the y coordinate
    for image in img_list:
        # add an image to the final array and increment the y coordinate
        final_image[current_y:image.shape[0] + current_y, :image.shape[1], :] = image
        current_y += image.shape[0]

    return final_image

def get_one_image_2(img_list):
    max_width = 0
    total_height = 200  # padding
    for img in img_list:
        if img.shape[1] > max_width:
            max_width = img.shape[1]
        total_height += img.shape[0]

    # create a new array with a size large enough to contain all the images
    final_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)

    current_y = 0  # keep track of where your current image was last placed in the y coordinate
    for image in img_list:
        # add an image to the final array and increment the y coordinate
        image = np.hstack((image, np.zeros((image.shape[0], max_width - image.shape[1], 3))))
        final_image[current_y:current_y + image.shape[0], :, :] = image
        current_y += image.shape[0]
    return final_image


np.random.seed(19680801)

n_bins = 10
x = np.random.randn(1000, 3)

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)

colors = ['red', 'tan', 'lime']
ax0.hist(x, n_bins, density=True, histtype='bar', color=colors, label=colors)
ax0.legend(prop={'size': 10})
ax0.set_title('bars with legend')

ax1.hist(x, n_bins, density=True, histtype='bar', stacked=True)
ax1.set_title('stacked bar')

ax2.hist(x, n_bins, histtype='step', stacked=True, fill=False)
ax2.set_title('stack step (unfilled)')

# Make a multiple-histogram of data-sets with different length.
x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
ax3.hist(x_multi, n_bins, histtype='bar')
ax3.set_title('different sample sizes')

fig.tight_layout()
plt.show()

PATH_TO_ANNOTATIONS = "/home/vladimir/datasets/TACO/data/annotations/annotations.json"
PATH_TO_IMAGES = "/home/vladimir/datasets/TACO/data/images"
OUTPUT_DIR = "/home/vladimir/datasets/TACO/data/parsing/categories"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import the dataset into the pylable schema 
# dataset = importer.ImportCoco(PATH_TO_ANNOTATIONS, path_to_images=PATH_TO_IMAGES, name="TACO_coco")
# dataset.df.head(5)
# df = dataset.df
with open(PATH_TO_ANNOTATIONS) as f:
    data = json.load(f)

# Create dict of category indices
catIndices = dict([(x["id"], x["name"]) for x in data["categories"]])
# Create image dict
images = {f'{x["id"]:d}': x for x in data["images"]}
# Create image-annotations dict
imgToAnns = defaultdict(list)
for ann in data["annotations"]:
    imgToAnns[ann["image_id"]].append(ann)
# Create objects dict
objectDict = dict([(x["id"], list()) for x in data["categories"]])

# Go through images and extract each object
# print(f"Processsing...", end="")
# for imgId, anns in imgToAnns.items():
#     # print(f"Processing image {imgId}...", end="")
#     img = cv2.imread(os.path.join(PATH_TO_IMAGES, images[str(imgId)]["file_name"]))
#     for ann in anns:
#         # print(f" cat-{ann['category_id'] }", end="")
#         print(f".", end="")
#         x, y, w, h = [int(x) for x in ann["bbox"]]
#         if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
#             print(f"\nImage {imgId} cat-{ann['category_id']} cropId = {len(objectDict[ann['category_id']])}\n")
#         w = min(max(0, w), img.shape[1])
#         x = min(max(0, x), img.shape[1] - w)
#         h = min(max(0, h), img.shape[0])
#         y = min(max(0, y), img.shape[0] - h)
#         crop = img[y:y+h, x:x+w]
#         objectDict[ann["category_id"]].append(crop)
#     # print("\n")
#     # if imgId == 10:
#     #     break

# with open(str(Path(OUTPUT_DIR).parent / "objectDict.pkl"), 'wb') as f:
#     pickle.dump(f, objectDict)
with open(str(Path(OUTPUT_DIR).parent / "objectDict.pkl"), 'rb') as f:
    objectDict = pickle.load(f)

# Save crops
if False:
    print(f"\n")
    for catId, catObjects in objectDict.items():
        print(f"Processing category {catId}...", end="")
        resDir = os.path.join(OUTPUT_DIR, f"{str(catId)} - {catIndices[catId]}")
        if len(catObjects):
            os.makedirs(resDir, exist_ok=True)
            for objId, obj in enumerate(catObjects):
                print(f" obj-{objId} ", end="")
                cv2.imwrite(os.path.join(resDir, str(objId) + ".jpg"), obj)
        print("\n")

print("Done!")

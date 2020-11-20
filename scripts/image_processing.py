import cv2
import os

# Iterate over all image files
directory = r'../assets/datasets/masks_correct/00000'
new_img_dir = r'../assets/datasets/masks_correct/resized'
counter = 0
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, (128, 128))
    save_to = os.path.join(new_img_dir, f"{str(counter)}.jpg")
    print(cv2.imwrite(save_to, resized))
    counter += 1


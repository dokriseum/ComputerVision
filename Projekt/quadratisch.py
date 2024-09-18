#! /usr/bin/env python3

import cv2
import os
import numpy as np


def make_square_by_letterboxing(image, desired_size=256):
    old_size = image.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    resized = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im

def process_images(input_dir, output_dir, desired_size=256):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                if image is not None:
                    square_image = make_square_by_letterboxing(image, desired_size)
                    relative_path = os.path.relpath(root, input_dir)
                    output_subdir = os.path.join(output_dir, relative_path)
                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)
                    output_path = os.path.join(output_subdir, file)
                    cv2.imwrite(output_path, square_image)
                else:
                    print(f"Skipping file (could not read): {image_path}")



input_directory = '/Users/dokriseum/Projects/CE-Master_Computer_Vision/cv-project-training/dataset'
output_directory = '/Users/dokriseum/Projects/CE-Master_Computer_Vision/cv-project-training/dataset_quadratisch'
process_images(input_directory, output_directory, desired_size=2048)

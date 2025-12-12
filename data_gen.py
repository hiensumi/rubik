import os
import cv2 as cv
import numpy as np

'''
This is the program that generates images for training the model by:
    1. Take backgrounds from ./dataset/images/backgrounds
    2. Take cube face images from ./dataset/images/rubik_image
    3. Randomly put cube face images on each of the backgrounds and save them to ./dataset/images/combined, meanwhile save data to rubik_coord.csv with format:
        image_name, x1, y1, x2, y2, x3, y3, x4, y4
        where (x1, y1), (x2, y2), (x3, y3), (x4, y4) are the coordinates of the 4 corners of the cube face in clockwise order starting from top-left corner.
'''

def generate_data(num_images=1000):
    import pandas as pd
    import random

    backgrounds_path = './image-dataset/images/backgrounds'
    rubik_images_path = './image-dataset/images/rubik_image'
    combined_path = './image-dataset/combined'
    os.makedirs(combined_path, exist_ok=True)

    backgrounds = [os.path.join(backgrounds_path, f) for f in os.listdir(backgrounds_path)]
    rubik_images = [os.path.join(rubik_images_path, f) for f in os.listdir(rubik_images_path)]

    data = []

    for i in range(num_images):
        bg_path = random.choice(backgrounds)
        rubik_path = random.choice(rubik_images)

        bg = cv.imread(bg_path)
        rubik = cv.imread(rubik_path)
        
        if random.random() < 0.4:
            cv.imwrite(os.path.join(combined_path, f'combined_{i:04d}.png'), bg)
            data.append([f'combined_{i:04d}.png', -1, -1, -1, -1, -1, -1, -1, -1])
            continue
            
        h_bg, w_bg, _ = bg.shape
        h_rubik, w_rubik, _ = rubik.shape

        scale = random.uniform(0.5, 1.5)
        rubik_resized = cv.resize(rubik, (int(w_rubik * scale), int(h_rubik * scale)))
        h_rubik_resized, w_rubik_resized, _ = rubik_resized.shape

        x_offset = random.randint(0, w_bg - w_rubik_resized)
        y_offset = random.randint(0, h_bg - h_rubik_resized)

        bg[y_offset:y_offset + h_rubik_resized, x_offset:x_offset + w_rubik_resized] = rubik_resized

        image_name = f'combined_{i:04d}.png'
        cv.imwrite(os.path.join(combined_path, image_name), bg)

        x1, y1 = x_offset, y_offset
        x2, y2 = x_offset + w_rubik_resized, y_offset
        x3, y3 = x_offset + w_rubik_resized, y_offset + h_rubik_resized
        x4, y4 = x_offset, y_offset + h_rubik_resized

        data.append([image_name, x1, y1, x2, y2, x3, y3, x4, y4])

    df = pd.DataFrame(data, columns=['image_name', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])
    df.to_csv('./image-dataset/rubik_coord.csv', index=False)

if __name__ == '__main__':
    generate_data(10000)
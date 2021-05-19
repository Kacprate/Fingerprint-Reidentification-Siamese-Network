import cv2
import fingerprint_enhancer		
import os
import tqdm

images = os.listdir('./data/light_data/dataset/train_data')

i = False
for image_name in tqdm.tqdm(images):
    img = cv2.imread('./data/light_data/dataset/train_data/' + image_name, 0)
    try:
        out = fingerprint_enhancer.enhance_Fingerprint(img)
        cv2.imwrite('./data/enhanced/light_data_enhanced/' + image_name, out)
    except:
        pass
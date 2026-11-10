import random
import cv2
import numpy as np
random.seed(42) 
for name in ['bull', 'leon', 'shelly', 'spike', '8-bit', 'bibi', 'bo', 'emz', 'frank', 'miko']:
    number = 21
    for i in range(1, 21):
        img = cv2.imread(f'dataset/{name}/{name}_{i:03d}.jpg')
        for k in range(9):
            angle = random.uniform(-25, 25)
            scale = random.uniform(0.85, 1.15)
            M = cv2.getRotationMatrix2D((128, 128), angle, scale)
            M[0, 2] += random.uniform(-0.08, 0.08) * 256
            M[1, 2] += random.uniform(-0.08, 0.08) * 256
            new = cv2.warpAffine(img, M, (256, 256), borderMode=cv2.BORDER_REPLICATE)
            if random.random() < 0.5:
                new = cv2.flip(new, 1)
            contrast = random.uniform(0.75, 1.25)
            brightness = random.uniform(-0.25, 0.25) * 127.5
            new = cv2.convertScaleAbs(new, alpha=contrast, beta=brightness)
            hsv = cv2.cvtColor(new, cv2.COLOR_BGR2HSV).astype(int)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-4, 4)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.85, 1.15), 0, 255)
            new = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            if random.random() < 0.2:
                new = cv2.GaussianBlur(new, (5, 5), 0)
            cv2.imwrite(f'dataset/{name}/{name}_{number:03d}.jpg', new, [cv2.IMWRITE_JPEG_QUALITY, 95])
            number += 1
    print(name, 'nice krasava')
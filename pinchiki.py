import os
import cv2
import numpy as np
for name in ['bull', 'leon', 'shelly', 'spike']:
    os.makedirs(f'dataset_masked/{name}', exist_ok=True)
    for i in range(1, 21):
        file = f'{name}_{i:03d}.jpg'
        img = cv2.imread(f'dataset/{name}/{file}')
        x, y, w, h = cv2.selectROI(file, img)
        cv2.destroyAllWindows()
        if w > 0: 
            mask = np.zeros(img.shape[:2], np.uint8)
            mask[y - 6:y + h + 6, x - 6:x + w + 6] = 255
            img = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)
        cv2.imwrite(f'dataset_masked/{name}/{file}', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
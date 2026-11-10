import os
import cv2

popups = ['emz_001', 'miko_001', 'miko_002']

for name in ['8-bit', 'bibi', 'bo', 'emz', 'frank', 'miko']:
    files = sorted(os.listdir(f'dataset/{name}'))
    for i in range(20):
        img = cv2.imread(f'dataset/{name}/{files[i]}')
        new_name = f'{name}_{i + 1:03d}'
        if new_name in popups:
            x, y, size = 295, 110, 470
        else:
            x, y, size = 430, 63, 504
        square = img[y:y + size, x:x + size]
        square = cv2.resize(square, (256, 256), interpolation=cv2.INTER_AREA)
        os.remove(f'dataset/{name}/{files[i]}')
        cv2.imwrite(f'dataset/{name}/{new_name}.jpg', square, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(name, 'cropped')

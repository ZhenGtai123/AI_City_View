import numpy as np
import cv2
import os

output_dir = 'output_1'
folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))][:30]

# Sky is class 0 in the semantic config
for folder in folders:
    sem_path = os.path.join(output_dir, folder, 'semantic_map.png')
    if os.path.exists(sem_path):
        sem = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
        sky_pct = np.sum(sem == 0) / sem.size * 100
        if sky_pct > 10:
            print(f'{folder}: sky={sky_pct:.1f}%')

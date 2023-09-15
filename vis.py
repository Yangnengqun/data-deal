import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

path = '/home/data/datasets/Map_V11_DB10/ADE20K/pid_mask/ADE_val_00000903.png'
image = Image.open(path)
image_num = np.array(image)
plt.imshow(image_num,cmap ='gray')
plt.show()

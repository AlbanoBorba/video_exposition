from skimage import io
import numpy as np
import os

path = '../results/experiment_refactory_load_image/val_images/25k/'
videos_tuple = []
# get all image path
for root, dirs, files in os.walk(path):
    for f in files:
        if f.endswith('.png'):
            path = os.path.join(root, f)
            videos_tuple.append({'path': path.split('/')[-1], 'image': 'image'})

images = {}

# split by sample
for t in videos_tuple:
    key = t['path'].split('_')[0]
    
    try:
        images[key].append(t)
    except:
        images[key] = [t]

frames = []
# order by frame
for _, value in images.items():
    value.sort(key=lambda x: x['path'])
    frames.append((x['image'] for x in value))
W
# horizontal concat
grid = (np.concatenate(f, axis=1) for f in frames)

# vertical concat
grid = np.concatenate(i, axis=0)

# save image
io.imsave('path'+'grid.png', grid)

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
            videos_tuple.append({'path': path.split('/')[-1], 'image': io.imread(path)})

images = {}

# split by sample
for t in videos_tuple:
    key = t['path'].split('_')[0]
    
    try:
        images[key].append(t)
    except:
        images[key] = [t]

# order by frame
for i in images:
    print(i)
    i = i.sort(key=lambda x: x['path'])
    i = (x['image'] for x in i)

# horizontal concat
grid = (np.concatenate(i, axis=1) for i in images)

# vertical concat
grid = np.concatenate(i, axis=0)

# save image
io.imsave('path'+'grid.png', grid)

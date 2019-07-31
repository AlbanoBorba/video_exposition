from skimage import io
import numpy as np

if __name__ == '__main__':
	path = '../results/experiment_refactory_load_image/val_images/25k/'

    videos_tuple = []

    # get all image path
	for root, dirs, files in os.walk(path):
		for f in files:
            if f.endswith('.png'):
                path = os.path.join(root,f)
				paths.append({'path': path, 'image': io.imread(path)})

    images = {}

    # split by sample
    for t in video_tuple:
        key = t['path'].split('_')[0]
        images[key] = t

    # order by frame
    for i in image:
        i = i.sort(key = lambda x : x['path'])
        i = (x['image'] for x in i)

    # horizontal concat
    grid = (np.concatenate(i, axis=1) for i in image)

    # vertical concat
    grid = np.concatenate(i, axis=0)

    # save image
    io.imsave('path'+'grid.png',grid)

    

import os
import numpy as np
from skimage import io, transform
import cv2


def read_video(video_path):
	cap = cv2.VideoCapture(video_path)

	while True:
		ret, frame = cap.read()
		
		if ret:
			avg = avg + frame

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			break
		else:
			break
	
	cap.release()
	cv2.destroyAllWindows()

def read_image(path):
	img = io.imread(path)
	medians.append(np.median(img))

global avg = np.zeros([1280, 720, 3])

if __name__ == '__main__':	
	path = '/media/albano/external'
	#path = './'

	for root, dirs, files in os.walk(path):
		for f in files:
			if f.endswith('.mov') and f.startswith('n_'):
				read_video(os.path.join(root,f))




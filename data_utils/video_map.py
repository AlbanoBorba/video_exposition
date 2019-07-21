import os
import numpy as np
from skimage import io, transform
import cv2
import matplotlib.pyplot as plt

def read_video(video_path, frames=40):
	cap = cv2.VideoCapture(video_path)

	global file
	count = 0
	x = []

	while True:
		ret, frame = cap.read()
		
		if ret:
			x.append(np.average(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
	
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

			count += 1
			if count == frames: break

		else:
			break
	
	file.write(str(np.average(x))+'\t'+str(np.std(x))+'\n')

	cap.release()
	cv2.destroyAllWindows()

def read_image(path):
	img = io.imread(path)
	medians.append(np.median(img))

file = open("histogram_distrib/all_distrib_40f.txt", "w")

if __name__ == '__main__':	
	path = '/media/albano/external'
	#path = './'

	for root, dirs, files in os.walk(path):
		for f in files:
			if f.endswith('.mov'):
				read_video(os.path.join(root,f))
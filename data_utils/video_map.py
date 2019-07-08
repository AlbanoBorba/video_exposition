import os
import numpy as np
from skimage import io, transform
import cv2
import matplotlib.pyplot as plt

def read_video(video_path):
	cap = cv2.VideoCapture(video_path)

	global avg
	global count
	global file

	while True:
		ret, frame = cap.read()
		
		if ret:
			
			#print(count)
			count = count + 1
			x = np.average(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
			file.write(x)
			avg.append(x)
			#avg = avg + frame.reshape((1280,720,3))
	
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

avg = []
count = 0
file = open("day_distrib.txt", "w")

if __name__ == '__main__':	
	path = '/media/albano/external'
	#path = './'

	for root, dirs, files in os.walk(path):
		for f in files:
			if f.endswith('.mov') and f.startswith('d_'):
				print(f)
				read_video(os.path.join(root,f))

	#r = [x for x in range(count)]

	print(avg)
	print(count)




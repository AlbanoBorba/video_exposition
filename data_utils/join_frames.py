import os
import cv2

def joinFrames(path, name, ext='jpg', fps=19.0):

    files = [file for file in sorted(os.listdir(path)) if file.split('.')[-1] == ext]

    frames = [cv2.imread(path+'/'+file) for file in files]

    fourcc = 0 # no compression
    #fourcc = 0x7634706d # mp4
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

    width = frames[0].shape[1]
    height = frames[0].shape[0]
    fps = fps
    
    out_writer = cv2.VideoWriter(path+'/'+name, fourcc, fps, (width, height))

    for frame in frames:
        out_writer.write(frame)

    out_writer.release()

if __name__ == '__main__':
    joinFrames('./bdd_test/00', 'out.avi')
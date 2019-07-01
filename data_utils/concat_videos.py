import os
import cv2
import numpy as np

def getPath(path):
    #path = './a/b'

    path = path.split('/')
    folder =  path.pop(-1)
    path = '/'.join(path)
    if path != '':
        path = path + '/'

    return path

def concat(paths, out_path, out_ext='avi'):
    # Read videos
    caps = []
    for path in paths:
        caps.append(cv2.VideoCapture(path))
    
    # Count videos
    w = h = 1
    if len(caps) == 2:
        w += 1
    elif len(caps) == 3:
        w += 2
    elif len(caps) == 4:
        w += 1
        h += 1
    elif len(caps) == 6:
        w += 2
        h += 2
    else:
        print("Invalid number of videos")

    # Set fourcc
    fourcc = 0

    # Get video properties
    fps = 10.0
    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH)) * w 
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT)) * h

    # Get all frames
    frames = []
    for cap in caps:
        print('Video')
        aux = []
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                aux.append(frame)
            else:
                break
        cap.release()
        frames.append(aux)

    # Concatenate
    concat = []
    if w == 2:
        # concat 1 e 2
        aux = []
        for a,b in zip(frames[0], frames[1]):
            aux.append(np.concatenate((a,b), axis=1))
            
        if h == 2:
            # concat 3 e 4
            aux2 = []
            for a,b in zip(frames[2], frames[3]):
                aux2.append(np.concatenate((a,b), axis=1))

            # concat (1,2) e (3,4)
            for a,b in zip(aux, aux2):
                concat.append(np.concatenate((a,b), axis=0))
        else:
            concat = aux

    elif w == 3:
        # concat 1, 2 e 3
        aux = []
        for a,b,c in zip(frames[0], frames[1], frames[2]):
            aux.append(np.concatenate((a,b,c), axis=1))

        if h == 3:
            # concat 4, 5 e 6
            aux2 = []
            for a,b,c in zip(frames[3], frames[4], frames[5]):
                aux2.append(np.concatenate((a,b,c), axis=1))

            # concat (1,2,3) e (4,5,6)
            for a,b in zip(aux, aux2):
                concat.append(np.concatenate((a,b), axis=0))
        else:
            concat = aux

    # Writer object
    #path = getPath(path[0])
    out = cv2.VideoWriter('{}/concat.{}'.format(out_path, out_ext), fourcc, fps, (width, height))
    
    # Write video
    for c in concat:
        out.write(c)

    out.release()

if __name__ == '__main__':
    #path = './bdd_mini/train/08'
    #concat([path+'/08_under/out.avi',path+'/08_under_ucan_under/out.avi', path+'/08_over/out.avi', path+'/08_over_ucan_over/out.avi'], path)

    #path = './bdd_mini/test/05'
    #concat([path+'/out.avi',path+'/05_ucan_over/out.avi'], path)

    path = './bdd_mini/test/tinhosa'
    concat([path+'/out.avi', path+'/tinhosa_ucan_under/out.avi'], path)

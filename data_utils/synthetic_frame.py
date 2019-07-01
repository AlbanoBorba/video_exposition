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

    return path, folder

def under(rgb, percent):
    rgb = rgb/255.
    rgb = saturate(rgb, [np.percentile(rgb, percent),1.])*255
    return (rgb).astype(np.uint8)      

def over(rgb, percent):
    rgb = rgb/255.
    rgb = saturate(rgb, [0.,np.percentile(rgb, percent)])*255
    return (rgb).astype(np.uint8)

def saturate(rgb, threshold = [0, .6]):
    rgb_clipped = np.clip(rgb, threshold[0], threshold[1])
    rgb_clipped = rgb_clipped - np.min(rgb_clipped)
    rgb_clipped = rgb_clipped / np.max(rgb_clipped)

    return rgb_clipped

def process(path, threshold=25, exposure='over under', out_ext='jpg'):
    # get name files in path
    files = [file for file in sorted(os.listdir(path)) if file.split('.')[-1] == out_ext]

    # get np frames by opencv
    frames = [cv2.imread(path+'/'+file) for file in files]

    # get under and over exposures for each frame
    if 'under' in exposure:
        under_results = [under(frame, threshold) for frame in frames]
    if 'over' in exposure:
        over_results = [over(frame, 100-threshold) for frame in frames]

    # set save path
    path, folder = getPath(path)  
    
    if 'under' in exposure:
        save_path = path+'/'+folder+'_under'
        os.mkdir(save_path)
        
        # save new frames
        for result in under_results:
            file_name = str(under_results.index(result))+'.'+out_ext
            cv2.imwrite(save_path+'/'+file_name,result)
    
    if 'over' in exposure:
        save_path = path+'/'+folder+'_over'
        os.mkdir(save_path)

        # save new frames
        for result in over_results:
            file_name = str(over_results.index(result))+'.'+out_ext
            cv2.imwrite(save_path+'/'+file_name,result)

if __name__ == '__main__':
    path = "./bdd_test/00"
    
    process(path)
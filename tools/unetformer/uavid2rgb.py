import numpy as np


def uavid2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [128, 0, 0]           # building
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 64, 128]        # road
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 128, 0]           # tree
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [128, 128, 0]         # vegetation
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [64, 0, 128]          # moving car
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [192, 0, 192]         # static car
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [64, 64, 0]           # human
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [0, 0, 0]             # cluster

    # mask_rgb[np.all(mask_convert == 8, axis=0)] = [0, 0, 128]             # sky
    # mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb


def custom2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 0, 0]             # cluster       black
    
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 0, 0]           # building      red
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [192, 192, 192]       # road        grey  
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [192, 0, 192]         # car           light violet
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 128, 0]           # tree          green
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [128, 128, 0]         # vegetation    dark green
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 255, 0]         # human         yellow
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [135, 206, 250]       # sky           light blue
    mask_rgb[np.all(mask_convert == 8, axis=0)] = [0, 0, 128]           # water         blue

    mask_rgb[np.all(mask_convert == 9, axis=0)] = [252,230,201]          # ground       egg
    mask_rgb[np.all(mask_convert == 10, axis=0)] = [128, 64, 128]        # mountain     dark violet

    # mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb


def custom2rgb_point(mask):
    size = mask.shape[0]
    mask_rgb = np.zeros(shape=(size,3), dtype=np.uint8)

    mask_convert = mask

    mask_rgb[mask_convert == 0] = [0, 0, 0]             # cluster       black
    
    mask_rgb[mask_convert ==1] = [128, 0, 0]           # building      red
    mask_rgb[mask_convert ==2] = [192, 192, 192]       # road        grey  
    mask_rgb[mask_convert ==3] = [192, 0, 192]         # car           light violet
    mask_rgb[mask_convert ==4] = [0, 128, 0]           # tree          green
    mask_rgb[mask_convert ==5] = [128, 128, 0]         # vegetation    dark green
    mask_rgb[mask_convert ==6] = [255, 255, 0]         # human         yellow
    mask_rgb[mask_convert ==7] = [135, 206, 250]       # sky           light blue
    mask_rgb[mask_convert ==8] = [0, 0, 128]           # water         blue

    mask_rgb[mask_convert ==9] = [252,230,201]          # ground       egg
    mask_rgb[mask_convert ==10] = [128, 64, 128]        # mountain     dark violet

    # mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb


def rgb2custom(rgb):
    h, w = rgb.shape[0], rgb.shape[1]
    mask = np.zeros(shape=(h, w), dtype=np.uint8)
    mask_convert = rgb
    mask[np.all(mask_convert == [0, 0, 0], axis=2)] = 0             # cluster       black
    mask[np.all(mask_convert == [128, 0, 0], axis=2)] = 1           # building      red
    mask[np.all(mask_convert == [192, 192, 192], axis=2)] = 2       # road          grey  
    mask[np.all(mask_convert == [192, 0, 192], axis=2)] = 3         # car           light violet
    mask[np.all(mask_convert == [0, 128, 0], axis=2)] = 4           # tree          green
    mask[np.all(mask_convert == [128, 128, 0], axis=2)] = 5         # vegetation    dark green
    mask[np.all(mask_convert == [255, 255, 0], axis=2)] = 6         # human         yellow
    mask[np.all(mask_convert == [135, 206, 250], axis=2)] = 7       # sky           light blue
    mask[np.all(mask_convert == [0, 0, 128], axis=2)] = 8           # water         blue
    mask[np.all(mask_convert == [252,230,201], axis=2)] = 9         # ground        egg
    mask[np.all(mask_convert == [128, 64, 128], axis=2)] = 10       # mountain      dark violet

    # mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask

def remapping(mask):
    mask[mask==5] = 2               # vegetation -> road
    mask[mask==6] = 0               # human  /
    mask[mask==7] = 0               # sky    /
    mask[mask==8] = 0               # water  /
    mask[mask==9] = 2               # ground -> road
    mask[mask==10] = 0              # mountain   /
    return mask
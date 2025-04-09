import numpy as np

# originally from ocrolib (here also with alpha support):
def pil2array(im,alpha=0):
    if im.mode=="L":
        a = np.frombuffer(im.tobytes(),'B')
        a.shape = im.height, im.width
        return a
    if im.mode=="LA":
        a = np.frombuffer(im.tobytes(),'B')
        a.shape = im.height, im.width, 2
        if not alpha: a = a[:,:,0]
        return a
    if im.mode=="RGB":
        a = np.frombuffer(im.tobytes(),'B')
        a.shape = im.height, im.width, 3
        return a
    if im.mode=="RGBA":
        a = np.frombuffer(im.tobytes(),'B')
        a.shape = im.height, im.width, 4
        if not alpha: a = a[:,:,:3]
        return a
    # fallback to Pillow grayscale conversion:
    return pil2array(im.convert("L"))

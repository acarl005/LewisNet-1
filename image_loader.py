import numpy as np
import glob
from PIL import Image

def load_images(paths):
    """Take a glob pattern for images and load into a 4-D numpy array, converted to grayscale"""
    if type(paths) == str:
        file_names = glob.glob(paths)
    elif type(paths) == list:
        file_names = []
        for path in paths:
            for file_name in glob.glob(path):
                file_names.append(file_name)
    else:
        raise ValueError("cannot load: " + str(paths))

    # convert("L") converts them to grayscale
    list_of_image_2d_arrays = [np.array(Image.open(name).convert("L")) for name in file_names]
    to_3d_array_of_images = np.array(list_of_image_2d_arrays)
    to_4d_array_of_images = to_3d_array_of_images[:,:,:,np.newaxis]
    return to_4d_array_of_images


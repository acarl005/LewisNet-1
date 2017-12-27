from PIL import Image
import glob
from os.path import basename, splitext

# split each page into "tiles" of 150x150
TILE_SIZE = 150

image_paths = glob.glob("img-pg/*.png")


for image_path in image_paths:
    im = Image.open(image_path)
    width, height = im.size
    tile_count = 0
    base_image_name = splitext(basename(image_path))[0]
    for i in range(0, height // TILE_SIZE):
        for j in range(0, width // TILE_SIZE):
            piece_i_start = i * TILE_SIZE
            piece_j_start = j * TILE_SIZE
            piece = im.crop((piece_j_start, piece_i_start, piece_j_start + TILE_SIZE, piece_i_start + TILE_SIZE))
            tile_count += 1
            piece.save("tiles/" + base_image_name + "-" + str(tile_count) + ".png")



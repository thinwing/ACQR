from PIL import Image as img
from pathlib import Path
from pdf2image import convert_from_path as cov

def convert(path=str, image_path=str):
    # path contains the name of figure
    # im = img.open(path)
    # fig = im.convert('RGB')
    # path_new = path.replace('pdf', 'eps')
    # fig.save(path_new, lossless=True)

    images = cov(path, 300)
    for _, image in enumerate(images):
        image.save(image_path, fmt='png')
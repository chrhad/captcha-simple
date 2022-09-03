from PIL import Image
import argparse
import sys

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Image RGB extractor")
    argparser.add_argument("input_file", help="input file in .jpg format")
    args = argparser.parse_args()
    
    fname = args.input_file
    im = Image.open(fname)
    height, width = im.size[1], im.size[0]
    print(height, width)
    pix = im.load()  # Get the RGBA value of the pixel of an image
    for y in range(height):
        print([sum(pix[x,y]) / 765 for x in range(width)])

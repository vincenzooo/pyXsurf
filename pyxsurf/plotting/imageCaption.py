from PIL import ImageFont
from PIL import ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

"""A module that creates a caption as image."""

def autofont(novo,csize,minfont=10):
    """Given a text novo (as list of lines), determine the font size that fits an image of size csize."""
    linefraction=0.1 #vertical margin for lines as a fraction of line height
    
    fontsize=minfont
    font = ImageFont.truetype("arial.ttf", fontsize)
    longestline=novo[np.argmax([len(n) for n in novo])]
    while font.getsize(longestline)[0] < csize[0]:
        # iterate until the text size is just smaller than the criteria
        fontsize += 1
        font = ImageFont.truetype("arial.ttf", fontsize)
    # optionally de-increment to be sure it is less than criteria
    fontsize -= 1 
    font = ImageFont.truetype("arial.ttf", fontsize)       
    
    #decrease font size if it doesn't fit 
    while font.getsize(longestline)[1]*len(novo)*(1+linefraction) > csize[1]:
        # iterate until the text size is just larger than the criteria
        fontsize -= 1
        font = ImageFont.truetype("arial.ttf", fontsize)
    # optionally de-increment to be sure it is less than criteria
    fontsize -= 1 
    font = ImageFont.truetype("arial.ttf", fontsize)
    return font
    
def render_caption(novo,csize,minfont=10):
    """return a rendered image of the text novo (as list of lines) determining the font size that fits the box. csize is the size of the returned image (x,y) in pixels."""

    offset = 10  #margin in pixels between linesheight   
    font=autofont(novo,csize,minfont)
    capim=Image.new("RGB",csize)
    d = ImageDraw.Draw(capim)
    #d = ImageDraw.Draw(newim)
    for line in novo:
        d.text((0,offset),line,(255,255,255),font=font)
        offset += font.getsize(line)[1]
    del d
    
    return capim
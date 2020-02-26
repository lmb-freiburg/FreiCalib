import cv2
import numpy as np


def put_text_centered(img, text, loc, **kwargs):
    textsize = cv2.getTextSize(text, kwargs['fontFace'], kwargs['fontScale'], kwargs['thickness'])[0]
    textX = int(round(loc[0] - textsize[0]/ 2.0))
    textY = int(round(loc[1] + textsize[1]/ 2.0))
    cv2.putText(img, text, (textX, textY), **kwargs)
    return img

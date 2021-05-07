import cv2
import numpy as np
import random


def isInRect(r, p):
    return (r[0] < p[0] and r[1] < p[1] and r[2] > p[0] and r[3] > p[1])

def 

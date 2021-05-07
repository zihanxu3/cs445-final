# Final project for CS 445 - Computational Photography 
# by Zihan Xu and Jiarui Zhang 
# Referenced: https://learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/

import numpy as np
import cv2
import sys
import os
import math
from PIL import Image

def morph_triangle(img1, img2, output, face1_trig, face2_trig, intermediate_trig, alpha):

    r1 = cv2.boundingRect(np.float32(face1_trig))
    r2 = cv2.boundingRect(np.float32(face2_trig))
    r = cv2.boundingRect(np.float32(intermediate_trig))

    offset1 = [(face1_trig[:][0] - r1[0]), (face1_trig[:][1] - r1[1])]
    offset2 = [(face2_trig[:][0] - r2[0]), (face2_trig[:][1] - r2[1])]
    offset_inter = [(intermediate_trig[:][0] - r[0]), (intermediate_trig[:][1] - r[1])]

    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]


    warpMat1 = cv2.getAffineTransform( np.float32(offset1), np.float32(offset_inter) )
    warpMat2 = cv2.getAffineTransform( np.float32(offset2), np.float32(offset_inter) )

    warped1 = cv2.warpAffine( img1Rect, warpMat1, (r2[2], r2[3]), None,
     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    warped2 = cv2.warpAffine( img1Rect, warpMat2, (r2[2], r2[3]), None,
     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    
    blended = (1.0 - alpha) * warped1 + alpha * warped2

    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    output[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = output[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * blended







def morph_two_images(img1, img2, face_points1, face_points2, triangulation):
    duration = 1
    frame_rate = 30
    frame_count = duration * frame_rate
    img1 = np.array(img1, dtype="float32")
    img2 = np.array(img2, dtype="float32")

    output = []
    for i in range(frame_count):
        alpha = (i + 1) / frame_count
        # compute the weighted average of two frames
        intermediate = np.zeros(face_points1.shape)
        intermediate[:, 0] = (1 - alpha) * face_points1[:, 0] + alpha * face_points2[:, 0]
        intermediate[:, 1] = (1 - alpha) * face_points1[:, 1] + alpha * face_points2[:, 1]

        morphed_frame = np.zeros(img1.shape, dtype="float32")

        for tri in triangulation:
            # indices of triangle -- applicable to keypoints and intermediate as well
            x = tri[0]
            y = tri[1]
            z = tri[2]

            face1_trig = [face_points1[x], face_points1[y], face_points1[z]]
            face2_trig = [face_points2[x], face_points2[y], face_points2[z]]
            intermediate_trig = [intermediate[x], intermediate[y], intermediate[z]]
            
            morph_triangle(img1, img2, morphed_frame, face1_trig, face2_trig, intermediate_trig, alpha)

        output.append(morphed_frame)
    
    return output
        



#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import cv2,os
import random
import numpy as np


def random_color():
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)

    return (r,g,b)

def make_shape(save_path,iter=100):
    if(not os.path.exists(save_path)):
        os.makedirs(save_path)

    for i in range(iter):
        s = 128;r=random.randint(40,60)
        img = np.zeros((s,s,3),dtype=np.uint8)
        d18 = 18.0*np.pi/180
        d54 = 54.0*np.pi/180
        a = (int(s/2),int(s/2 - r))
        b = (int(s/2+r*np.cos(d18)),int(s/2-r*np.sin(d18)))
        e = (int(s/2-r*np.cos(d18)),int(s/2-r*np.sin(d18)))

        c = (int(s/2+r*np.cos(d54)),int(s/2+r*np.sin(d54)))
        d = (int(s/2-r*np.cos(d54)),int(s/2+r*np.sin(d54)))

        #pentagon
        color = random_color()
        cv2.line(img,a,b,color,2)
        cv2.line(img,b,c,color,2)
        cv2.line(img,c,d,color,2)
        cv2.line(img,d,e,color,2)
        cv2.line(img,e,a,color,2)
        cv2.imwrite('{0}/00_{1}.jpg'.format(save_path,i),img)

        #start
        img1 = np.zeros_like(img)
        color = random_color()
        cv2.line(img1,a,d,color,2)
        cv2.line(img1,d,b,color,2)
        cv2.line(img1,b,e,color,2)
        cv2.line(img1,e,c,color,2)
        cv2.line(img1,c,a,color,2)
        cv2.imwrite('{0}/01_{1}.jpg'.format(save_path,i),img1)

        #circle
        img2 = np.zeros_like(img)
        color = random_color()
        cv2.circle(img2,(int(s/2),int(s/2)),r,color,2)
        cv2.imwrite('{0}/02_{1}.jpg'.format(save_path,i),img2)

        #ellipse
        img3 = np.zeros_like(img)
        color = random_color()
        cv2.ellipse(img3,(int(s/2),int(s/2)),(r-10,r+10),0,0,360,color)
        cv2.imwrite('{0}/03_{1}.jpg'.format(save_path,i),img3)
    return

if __name__ == '__main__':
    make_shape('test_set',10)

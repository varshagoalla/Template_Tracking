
import os, sys, cv2, copy
import numpy as np
from lk_trackers import *

def IOU(box_1,box_2):
    ((x,y),w,h) = box_1
    ((t_x,t_y),t_w,t_h) = box_2
    width = max(min(x+w,t_x+t_w)-max(x,t_x), 0)
    height = max(min(y+h,t_y+t_h)-max(y,t_y), 0)
    intersection = width*height
    union = w*h + t_w*t_h - intersection
    
    return intersection/union

def main():
    img_folder_path = sys.argv[1]
    grd_truths_path = sys.argv[2]
    images = [img_folder_path + "/" + s for s in sorted(os.listdir(img_folder_path))]
    grd_truths = [[int(s) for s in line.strip("\n").split(",")] for line in open(grd_truths_path,'r').readlines()]

    #Get template
    template = cv2.imread(images[0])
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    #template = cv2.GaussianBlur(template, (5, 5), 5) 
    [t_x,t_y,t_w,t_h] = grd_truths[0]
    box = np.array([[t_x, t_y], [t_x+t_w,t_y+t_h]])
    tl = np.array([t_x, t_y, 1])
    br = np.array([t_x+t_w,t_y+t_h, 1])
    tr = np.array([t_x+t_w,t_y,1])
    bl = np.array([t_x,t_y+t_h,1])

    mIOU = 0
    threshold = 0.001
    
    for i in range(1,len(images)):
        image = cv2.imread(images[i])
        [g_x,g_y,g_w,g_h] = grd_truths[i]
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        
        p = np.zeros(6)
        p=LK(image_gray, template_gray, p,box, threshold,"affine",100)    
        M = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])

        #p=np.zeros(2)
        #p=LK(image_gray, template_gray, p,box, threshold,"translate")  
        #M = np.array([[1,0, p[0]], [0, 1, p[1]]])
    
        new_tl = (M @ tl).astype(int)
        new_br = (M @ br).astype(int)
        new_tr = (M @ tr).astype(int)
        new_bl = (M @ bl).astype(int)
        max_x = max(new_tl[0],new_br[0],new_tr[0],new_bl[0])
        min_x = min(new_tl[0],new_br[0],new_tr[0],new_bl[0])
        max_y = max(new_tl[1],new_br[1],new_tr[1],new_bl[1])
        min_y = min(new_tl[1],new_br[1],new_tr[1],new_bl[1])


        iou = IOU(((new_tl[0],new_tl[1]),new_br[0]-new_tl[0],new_br[1]-new_tl[1]),((g_x,g_y),g_w,g_h))
        mIOU += iou

        #out = cv2.rectangle(image, (min_x,min_y), (max_x,max_y), (255, 255, 0), 1)
        out = cv2.rectangle(image, tuple(new_tl),tuple(new_br), (255, 0, 0), 2)
        cv2.imshow("template",out)
        cv2.waitKey(1)

        #template_gray = image_gray
        #box = np.array([[new_tl[0],new_tl[1]],[new_br[0],new_br[1]]])
    mIOU = mIOU/(len(images)-1)   
    print(mIOU)
    pass

main()
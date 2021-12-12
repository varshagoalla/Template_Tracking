
import os, sys, cv2, copy
import numpy as np
from lk_trackers import *




def main():
    
    
    
    
    template = cv2.resize(cv2.imread(sys.argv[1]),(640,480))
    #template = cv2.GaussianBlur(template, (5, 5), 5) 
    t_x = int(int(sys.argv[2])*640/1280)
    t_y = int(int(sys.argv[3])*480/720)
    t_w = int(int(sys.argv[4])*640/1280)
    t_h = int(int(sys.argv[5])*480/720)
    box = np.array([[t_x,t_y], [t_x+t_w,t_y+t_h]])
    tl = np.array([t_x, t_y, 1])
    br = np.array([t_x+t_w,t_y+t_h, 1])
    tr = np.array([t_x+t_w,t_y,1])
    bl = np.array([t_x,t_y+t_h,1])
    
    #out = cv2.rectangle(template, (tl[0],tl[1]),(br[0],br[1]), (255, 0, 0), 2)
    #cv2.imshow("template",out)
    #cv2.waitKey(0)
    mIOU = 0
    threshold = 0.001
    num_layers = 1
    vid = cv2.VideoCapture(-1)
    while True:
        ret, image = vid.read()
        method = cv2.TM_CCORR_NORMED
        result = cv2.matchTemplate(image,template[t_y:t_y+t_h,t_x:t_x+t_w], method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        (x,y) = max_loc
        out = cv2.rectangle(image, (x, y),(x+t_w, y+t_h), (255, 0, 0), 2)
        cv2.imshow("template",out)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #template_gray = image_gray
        #box = np.array([[new_tl[0],new_tl[1]],[new_br[0],new_br[1]]])
   
    pass

main()

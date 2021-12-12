
import os, sys, cv2
import numpy as np


def IOU(box_1,box_2):
    ((x,y),w,h) = box_1
    ((t_x,t_y),t_w,t_h) = box_2
    width = max(min(x+w,t_x+t_w)-max(x,t_x), 0)
    height = max(min(y+h,t_y+t_h)-max(y,t_y), 0)
    intersection = width*height
    union = w*h + t_w*t_h - intersection
    #print("intersection %d union %d"%(intersection,union))
    #print(intersection/union)
    return intersection/union

def ssd(img_1,img_2):
    return np.sum((img_1[:,:,0:3]-img_2[:,:,0:3])**2)

def ncc(img_1,img_2):
    num = np.sum((img_1-np.mean(img_1))*(img_2-np.mean(img_2)))
    den = np.sqrt(np.sum((img_1-np.mean(img_1))**2)*np.sum((img_2-np.mean(img_2))**2))
    return num/den

def match_template(image,template,method):
    rows,cols,ch = image.shape
    min_val = 1000000000000
    max_val = -100000000000
    max_start = None
    min_start = None
    h,w,ch = template.shape
    for x in range(0,cols-w+1):
        for y in range(0,rows-h+1):
            if method=="ssd":
                val = ssd(image[y:y+h,x:x+w],template)
                if val<min_val:
                    min_val = val
                    min_start = (x,y)
            else:
                val = ncc(image[y:y+h,x:x+w],template)
                if val>max_val:
                    max_val = val
                    max_start = (x,y)
    if method=="ssd":        
        return min_start,w,h
    return max_start,w,h




def main():
    img_folder_path = sys.argv[1]
    grd_truths_path = sys.argv[2]
    images = [img_folder_path + "/" + s for s in sorted(os.listdir(img_folder_path))]
    grd_truths = [[int(s) for s in line.strip("\n").split(",")] for line in open(grd_truths_path,'r').readlines()]

    #Get template
    image = cv2.imread(images[0])
    [t_x,t_y,t_w,t_h] = grd_truths[0]
    template = image[t_y:t_y+t_h,t_x:t_x+t_w]

    mIOU = 0
    for i in range(1,len(images)):
        image = cv2.imread(images[i])
        [g_x,g_y,g_w,g_h] = grd_truths[i]
        if sys.argv[3]=="ncc":
            method = cv2.TM_CCORR_NORMED
        else:
            method = cv2.TM_SQDIFF
        result = cv2.matchTemplate(image,template, method)
        #TM_SQDIFF
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if method==cv2.TM_CCORR_NORMED:
            (x,y) = max_loc
        else:
            (x,y) = min_loc
        iou = IOU(((x,y),t_w,t_h),((g_x,g_y),g_w,g_h))
        mIOU += iou
        out = cv2.rectangle(image, (x, y),(x+t_w, y+t_h), (255, 0, 0), 2)
        cv2.imshow("template",out)
        cv2.waitKey(1)
    mIOU = mIOU/(len(images)-1)   
    print(mIOU)
    """
    #part2
    if part == "2":
        frame1 = image
        (x,y),w,h = ((t_x,t_y),t_w,t_h)
        for i in range(1,len(images)):
            frame2 = cv2.imread(images[i])
            [g_x,g_y,g_w,g_h] = grd_truths[i]
            ((x,y),w,h)= affine_LK(frame2,frame1,))
            #Image1 = Image.open(images[i-1]).convert('L')
            #Image2 = Image.open(images[i]).convert('L')
            #t=0.3
            LK_OpticalFlow(Image1, Image2)
            iou = IOU(((x,y),w,h),((g_x,g_y),g_w,g_h))
            mIOU += iou
            out = cv2.rectangle(frame2, (round(x), round(y)),(round(x+w), round(y+h)), (255, 0, 0), 2)
            frame1 = frame2
            cv2.imshow("template",out)
            cv2.waitKey(1)
        mIOU = mIOU/(len(images)-1)   
        print(mIOU)
        pass
"""
main()
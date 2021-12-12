import cv2
import numpy as np 
import math
import copy

def jacobian(cols, rows, transformation):
    if transformation=="affine":
        x = np.array(range(cols))
        y = np.array(range(rows))
        x, y = np.meshgrid(x, y) 
        ones = np.ones((rows,cols))
        zeros = np.zeros((rows,cols))
        J = np.stack((np.stack((x, zeros, y, zeros, ones, zeros), axis=2),np.stack((zeros, x, zeros, y, zeros, ones), axis=2)), axis=2)
    elif transformation=="translate":
        ones = np.ones((rows,cols))
        zeros = np.zeros((rows, cols))
        J = np.stack((np.stack((ones, zeros), axis=2), np.stack((zeros, ones), axis=2)), axis=2)
    return J
    

def LK(image, template, p, box, t, transform, max_itr=100):
        template = template[box[0][1]:box[1][1],box[0][0]:box[1][0]]
        t_rows, t_cols = template.shape
        rows,cols = image.shape
        p_ = p
        i = 0
        dp_val = np.inf
        while i<=max_itr and t<=dp_val:
            M = np.array([[1,0,p_[0]],[0,1,p_[1]]])
            if transform=="affine":
                M = np.array([[1+p_[0], p_[2], p_[4]], [p_[1], 1+p_[3], p_[5]]])
            warped = cv2.warpAffine(image, M, (cols,rows),flags=cv2.INTER_CUBIC)
            warped = warped[box[0][1]:box[1][1],box[0][0]:box[1][0]]
            difference = template.astype(int) - warped.astype(int)
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
            grad_x_warped = cv2.warpAffine(grad_x, M, (cols,rows),flags=cv2.INTER_CUBIC+cv2.WARP_INVERSE_MAP)
            grad_x_warped = grad_x_warped[box[0][1]:box[1][1],box[0][0]:box[1][0]]
            grad_y_warped = cv2.warpAffine(grad_y, M, (cols,rows),flags=cv2.INTER_CUBIC+cv2.WARP_INVERSE_MAP)
            grad_y_warped = grad_y_warped[box[0][1]:box[1][1],box[0][0]:box[1][0]]
            J = jacobian(t_cols, t_rows, "translate")
            if transform == "affine":
                J = jacobian(t_cols, t_rows, "affine")      
            grad = np.expand_dims(np.stack((grad_x_warped, grad_y_warped), axis=2),axis=2)
            steepest_descents = np.matmul(grad, J)
            steepest_descents_T = np.transpose(steepest_descents, (0, 1, 3, 2))
            H = np.matmul(steepest_descents_T, steepest_descents).sum((0,1))
            difference = difference.reshape((t_rows,t_cols, 1, 1))
            dp = np.matmul(np.linalg.pinv(H), (steepest_descents_T * difference).sum((0,1))).reshape((-1))
            p_ += dp
            dp_val = np.linalg.norm(dp)
            i += 1
        return p_





def pyr_LK(image, template, box, num_layers, threshold, transformation, iterations=100):
    template_copy = copy.deepcopy(template)
    image_copy = copy.deepcopy(image)
    down_factor = 1/2**num_layers
    box_ = (box*down_factor).astype(int)
    for i in range(num_layers):
        image_copy = cv2.pyrDown(image)
        template_copy = cv2.pyrDown(template_copy)
    p = np.zeros(2)
    if transformation=="affine":
        p = np.zeros(6)
    box_temp = box_    
    for i in range(num_layers+1):
        p = LK(image_copy, template_copy, p, box_temp,  threshold,transformation,iterations)
        image_copy = cv2.pyrUp(image_copy)
        template_copy = cv2.pyrUp(template_copy)
        box_temp = (box_temp * 2).astype(int)
    return p

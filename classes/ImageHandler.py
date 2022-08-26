import cv2
import numpy as np

class ImageHandler:
    
    @staticmethod
    def crop(np_image, points):
        return np_image[points[0]:points[1], points[2]:points[3]]

    @staticmethod
    def write_to_file(filename, np_image):
        return cv2.imwrite(filename, np_image)

    @staticmethod
    def draw_vehicle_shape(np_image, points, color=(255,0,0), thickness=1):
        top_left = (points[2], points[0])
        bottom_right = (points[3], points[1])
        cv2.rectangle(np_image, top_left, bottom_right, color, thickness=thickness)

    @staticmethod
    def draw_losangle(np_image, points, color=(1.,1.,1.), thickness=1):
        for i in range(4):
            pt1 = tuple(points[:,i].astype(int).tolist())
            pt2 = tuple(points[:,(i+1)%4].astype(int).tolist())
            cv2.line(np_image,pt1,pt2,color,thickness)

    @staticmethod
    def write2img(np_image,points,strg,txt_color=(0,0,0),bg_color=(255,255,255),font_size=1):
        wh_img = np.array(np_image.shape[1::-1])
        
        font = cv2.FONT_HERSHEY_SIMPLEX

        wh_text,v = cv2.getTextSize(strg, font, font_size, 3)
        rpoints = points / np.array(wh_img, dtype=float).reshape(2,1)
        
        bl_corner = rpoints.min(1) * wh_img
        tl_corner = np.array([bl_corner[0],bl_corner[1]-wh_text[1]])/wh_img
        br_corner = np.array([bl_corner[0]+wh_text[0],bl_corner[1]])/wh_img
        bl_corner /= wh_img

        if (tl_corner < 0.).any():
            delta = 0. - np.minimum(tl_corner,0.)
        elif (br_corner > 1.).any():
            delta = 1. - np.maximum(br_corner,1.)
        else:
            delta = 0.

        tl_corner += delta
        br_corner += delta
        bl_corner += delta

        tpl = lambda x: tuple((x*wh_img).astype(int).tolist())

        cv2.rectangle(np_image, tpl(tl_corner), tpl(br_corner), bg_color, -1)	
        cv2.putText(np_image,strg,tpl(bl_corner),font,font_size,txt_color,3) 

    
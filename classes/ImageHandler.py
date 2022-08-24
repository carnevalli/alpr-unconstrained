import cv2

class ImageHandler:
    
    @staticmethod
    def crop(np_image, points):
        return np_image[points[0]:points[1], points[2]:points[3]]

    @staticmethod
    def write_to_file(filename, np_image):
        return cv2.imwrite(filename, np_image)

    
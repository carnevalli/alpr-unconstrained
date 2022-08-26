import cv2
from src.keras_utils import detect_lp
from src.utils import im2single

class LicensePlateDetector:
    def __init__(self, wpod_net_model=None, threshold=.5, 
        max_image_size=1600, max_results=None,
        bw_threshold=None):
        self.wpod_net_model = wpod_net_model
        self.threshold = threshold
        self.max_image_size = max_image_size + (max_image_size%(2**4))
        self.max_results = max_results
        self.bw_threshold = bw_threshold

    
    def detect(self, np_image):
        print('Searching for license plates using WPOD-NET. Threshold: %.0f' % (self.threshold * 100))
        max_img_dimension = max(np_image.shape[:2])
        side = min(max_img_dimension, self.max_image_size)
        bound_dim = min(side + (side%(2**4)), self.max_image_size)
        print(max(np_image.shape[:2]), min(np_image.shape[:2]))
        print("\t\tBound dim: %d, side: %f" % (bound_dim, side))

        lp_labels, lp_images, _ = detect_lp(self.wpod_net_model,im2single(np_image),bound_dim,2**4,(360,120),self.threshold)

        print('\t\tPossible LP found: %d' % len(lp_images))

        if self.max_results:
            lp_images = lp_images[:self.max_results]

        lps_output = []
        if len(lp_images):
            for j in range(len(lp_images)):
                lp_image = lp_images[j] * 255.
                lp_image = cv2.cvtColor(lp_image, cv2.COLOR_BGR2GRAY)
                if self.bw_threshold:
                    (thresh, lp_image) = cv2.threshold(lp_image, self.bw_threshold, 255, cv2.THRESH_BINARY)
                # lp_image = cv2.cvtColor(lp_image, cv2.COLOR_GRAY2BGR)

                lps_output.append({
                    "image": lp_image,
                    "points": lp_labels[j].pts
                })
        return lps_output
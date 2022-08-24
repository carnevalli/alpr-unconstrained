from darknet.python.darknet import detect
from src.label import dknet_label_conversion
from src.utils import nms

class LicensePlateOCR:
    def __init__(self, 
        ocr_net=None, ocr_meta=None,
        threshold=.4):
        self.ocr_net = ocr_net
        self.ocr_meta = ocr_meta
        self.threshold = threshold

    def detect(self, img_path):
        print('Performing OCR. Threshold: %.0f' % ( self.threshold * 100))

        R,(width,height) = detect(self.ocr_net, self.ocr_meta, img_path ,thresh=self.threshold, nms=None)

        lp_str = ''
        if len(R):

            L = dknet_label_conversion(R,width,height)
            L = nms(L,.45)

            L.sort(key=lambda x: x.tl()[0])
            lp_str = ''.join([chr(l.cl()) for l in L])

        return lp_str

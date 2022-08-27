import os
import time
import torch
import cv2
from classes.LicensePlateOCR import LicensePlateOCR
from classes.LicensePlateTransformation import LicensePlateTransformation
from classes.OutputProcessor import OutputProcessor
import darknet.python.darknet as darknet
from uuid import uuid4
from flask import Flask, render_template, request, make_response
from werkzeug.exceptions import BadRequest
from classes.LicensePlateDetector import LicensePlateDetector
from classes.VehicleDetector import VehicleDetector
from classes.ImageHandler import ImageHandler
from src.keras_utils import load_model


app = Flask(__name__)

files_dir = 'files/'
valid_exts = ['png', 'jpg', 'webp']

# YoloV5 vehicle detection settings
yolov5_model = None
# WPOD-NET LP Detection setting
wpod_net_model = None
# OCR Settings
ocr_weights = b'data/ocr/ocr-net.weights'
ocr_netcfg  = b'data/ocr/ocr-net.cfg'
ocr_dataset = b'data/ocr/ocr-net.data'
ocr_net = None
ocr_meta = None


@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/run", methods=['POST'])
def run():
    img = extract_img(request)
    img_uid = str(uuid4())
    img_ext = get_img_extension(img.filename)
    img_path =  files_dir + img_uid + '/'
    img_full_path = img_path + 'base.jpg'

    os.makedirs(img_path)
    os.makedirs(img_path + 'output')
    img.save(img_full_path)

    np_image = cv2.imread(img_full_path)

    vehicles = vehicle_detection(img_uid, np_image)

    for i, vehicle in enumerate(vehicles):
        v = ImageHandler.crop(np_image, vehicle['points'])
        vehicle_lps = []

        lps = lp_detection(img_uid, v)

        for j, lp in enumerate(lps):
            ImageHandler.write_to_file(img_path + '/output/v_%d_lp_%d.png' % (i, j), lp['image'])
            lp_str = lp_ocr(img_path + '/output/v_%d_lp_%d.png' % (i, j))

            if len(lp_str.strip()) > 0:
                vehicle_lps.append((lp_str, lp['points']))

        vehicles[i]['lps'] = vehicle_lps

    validation_regex = load_regex_from_file('regex.tsv')
    output = generate_outputs(img_uid, np_image, vehicles, img_path, validation_regex)
    
    return '<pre>' + str(vehicles) + '</pre>'

def vehicle_detection(img_uid, np_image):
    base_dir = files_dir + img_uid
    output_dir = base_dir + '/output'
    detector = VehicleDetector(model=yolov5_model, coco_categories_of_interest=['bus'])
    vehicles = detector.detect(np_image)

    return vehicles

def lp_detection(img_uid, np_image):
    detector = LicensePlateDetector(wpod_net_model=wpod_net_model)
    lps = detector.detect(np_image)
    return lps

def lp_ocr(img_path):
    detector = LicensePlateOCR(ocr_net=ocr_net, ocr_meta=ocr_meta)
    lp = detector.detect(img_path)
    return lp

def generate_outputs(img_uid, np_image, vehicles, img_path, validation_regex):
    processor = OutputProcessor(img_uid, generate_demo=True, generate_vehicles=True, validation_regex_list=validation_regex)
    return processor.process(np_image, vehicles)

def load_regex_from_file(path):
    return LicensePlateTransformation.loadRegexPatterns(path)

def get_img_extension(f):
    ext = os.path.splitext(f)[-1]
    return ext

def extract_img(request):
    if 'img' not in request.files:
        raise BadRequest('Missing file parameter')

    img = request.files['img']

    if img.filename == '':
        raise BadRequest('Given file is invalid')

    if get_img_extension(img.filename).replace('.', '') not in valid_exts:
        raise BadRequest('Not supported image extension. Supported: ' + ', '.join(valid_exts))

    return img


if __name__ == '__main__':
    print('Loading YoloV5...')
    yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    wpod_net_model = load_model('data/lp-detector/wpod-net_update1.h5')
    ocr_net  = darknet.load_net(ocr_netcfg, ocr_weights, 0)
    ocr_meta = darknet.load_meta(ocr_dataset)
    app.run(port=3001, debug=True)
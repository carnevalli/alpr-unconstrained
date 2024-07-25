import os
import time
import torch
import cv2
import json
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

params = {
    'max_vehicles' : 1,
    'max_lps' : 1,
    'regex' : [],
    'vehicle_threshold': .5,
    'vehicles_order': 'area',
    'coco_categories': 'car,truck,bus',
    'whole_image_fallback': False,
    'lp_threshold': .5,
    'lp_max_image_size': 1600,
    'lp_bw_threshold': 127,
    'ocr_threshold': .5,
    'suppress_transformations': False,
    'generate_demo': False,
    'demo_filename': 'demo.png',
    'generate_vehicles': False
}

default_params = {
    'max_vehicles' : 1,
    'max_lps' : 1,
    'regex' : '[]',
    'vehicle_threshold': .5,
    'vehicles_order': 'area',
    'coco_categories': ['car', 'truck', 'bus'],
    'whole_image_fallback': False,
    'lp_threshold': .5,
    'lp_max_image_size': 1600,
    'lp_bw_threshold': 127,
    'ocr_threshold': .4,
    'suppress_transformations': False,
    'generate_demo': False,
    'demo_filename': 'demo.png',
    'generate_vehicles': False
}

warnings = []

@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/run", methods=['POST'])
def run():
    img = extract_img(request)
    
    for k in params.keys():
        if request.form.get(k):
            params[k] = request.form.get(k)

    MAX_VEHICLES_MIN = 1
    MAX_VEHICLES_MAX = 5
    MAX_LPS_MIN = 1
    MAX_LPS_MAX = 5
    ZERO_ONE_RANGE_MIN = 0
    ZERO_ONE_RANGE_MAX = 1
    VEHICLES_ORDER_VALUES = ['area', 'confidence']
    LP_MAX_IMAGE_SIZE_MIN = 480
    LP_MAX_IMAGE_SIZE_MAX = 2000
    LP_BW_THRESHOLD_MIN = 0
    LP_BW_THRESHOLD_MAX = 255

    for k in params.keys():
        v = params[k]
        if k == 'max_vehicles':
            params[k] = v = int(v)
            if v < MAX_VEHICLES_MIN or v > MAX_VEHICLES_MAX:
                params[k] = default_params[k]
                add_range_warning(k, MAX_VEHICLES_MIN, MAX_VEHICLES_MAX)

        elif k == 'max_lps':
            params[k] = v = int(v)
            if v < MAX_LPS_MIN or v > MAX_LPS_MAX:
                params[k] = default_params[k]
                add_range_warning(k, MAX_LPS_MIN, MAX_LPS_MAX)

        elif k == 'regex':
            try:
                j = json.loads(v)
            except:
                warnings.append('Invalid JSON supplied to "%s" parameter. Using default: %s' % (k, str(default_params[k])))

        elif k in ['lp_threshold', 'vehicle_threshold', 'ocr_threshold']:
            params[k] = v = float(v)
            if v < ZERO_ONE_RANGE_MIN or v > ZERO_ONE_RANGE_MAX:
                params[k] = default_params[k]
                add_range_warning(k, ZERO_ONE_RANGE_MAX, ZERO_ONE_RANGE_MAX)

        elif k == 'vehicles_order':
            if v not in VEHICLES_ORDER_VALUES:
                params[k] = default_params[k]
                warnings.append('Invalid value supplied for "%s" parameter. Should be one of: %s. Using default: %s' % (k, str(VEHICLES_ORDER_VALUES, default_params[k])))

        elif k == 'coco_categories':
            params[k] = v.split(',')
            if len(params[k]) == 0:
                params[k] = default_params[k]
                warnings.append('No value supplied for %s parameter. Using default value instead: %s' % (k, str(default_params[k])))

        elif k in ['whole_image_fallback', 'suppress_transformations', 'generate_demo', 'generate_vehicles']:
            if v not in ['0', '1']:
                params[k] = default_params[k]
                add_boolean_warning(k)
            else:
                params[k] = True if v == '1' else False

        elif k == 'lp_max_image_size':
            params[k] = v = int(v)
            if v < LP_MAX_IMAGE_SIZE_MIN or v > LP_MAX_IMAGE_SIZE_MAX:
                params[k] = default_params[k]
                add_range_warning(k, LP_MAX_IMAGE_SIZE_MIN, LP_MAX_IMAGE_SIZE_MAX)

        elif k == 'lp_bw_threshold':
            params[k] = v = int(v)
            if v < LP_BW_THRESHOLD_MIN or v > LP_BW_THRESHOLD_MAX:
                params[k] = default_params[k]
                add_range_warning(k, LP_BW_THRESHOLD_MIN, LP_BW_THRESHOLD_MAX)
        elif k == 'demo_filename':
            if len(v) < 5:
                params[k] = default_params[k]
                warnings.append('Invalid value supplied for "%s" parameters. Should be a valid filename with a valid image extension. Using default: "%s"', (k, default_params[k]))





    return params


    # img_uid = str(uuid4())
    # img_ext = get_img_extension(img.filename)
    # img_path =  files_dir + img_uid + '/'
    # img_full_path = img_path + 'base.jpg'

    # os.makedirs(img_path)
    # os.makedirs(img_path + 'output')
    # img.save(img_full_path)

    # np_image = cv2.imread(img_full_path)

    # vehicles = vehicle_detection(img_uid, np_image)

    # for i, vehicle in enumerate(vehicles):
    #     v = ImageHandler.crop(np_image, vehicle['points'])
    #     vehicle_lps = []

    #     lps = lp_detection(img_uid, v)

    #     for j, lp in enumerate(lps):
    #         ImageHandler.write_to_file(img_path + '/output/v_%d_lp_%d.png' % (i, j), lp['image'])
    #         lp_str = lp_ocr(img_path + '/output/v_%d_lp_%d.png' % (i, j))

    #         if len(lp_str.strip()) > 0:
    #             vehicle_lps.append((lp_str, lp['points']))

    #     vehicles[i]['lps'] = vehicle_lps

    # validation_regex = load_regex_from_file('regex.tsv')
    # output = generate_outputs(img_uid, np_image, vehicles, img_path, validation_regex)

    # parsed = json.dumps(output)
    
    # return '<pre>' + parsed + '</pre>'

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

def add_range_warning(key, min, max):
    warnings.append('Invalid value supplied for paramater "%s". Should be between %d and %d. Using default: %d' %
        (key, min, max, default_params[key]))

def add_boolean_warning(key):
    warnings.append('Invalid value supplied for parameter "%s". Should be 0 or 1. Using default: %d' % (key, (1 if default_params[key] else 0)))

if __name__ == '__main__':
    print('Loading YoloV5...')
    yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    wpod_net_model = load_model('data/lp-detector/wpod-net_update1.h5')
    ocr_net  = darknet.load_net(ocr_netcfg, ocr_weights, 0)
    ocr_meta = darknet.load_meta(ocr_dataset)
    app.run(host="0.0.0.0", port=3001, debug=True)
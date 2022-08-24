import os
import time
import torch
import cv2
from uuid import uuid4
from flask import Flask, render_template, request, make_response
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
from classes.VehicleDetector import VehicleDetector
from classes.ImageHandler import ImageHandler
from numpy import asarray

app = Flask(__name__)

files_dir = 'files/'
valid_exts = ['png', 'jpg', 'webp']

# YoloV5 vehicle detection settings
yolov5_model = None


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

    np_img = cv2.imread(img_full_path)

    vehicles = vehicle_detection(img_uid, np_img)

    return '<pre>' + str(vehicles) + '</pre>'

def vehicle_detection(img_uid, np_img):
    base_dir = files_dir + img_uid
    output_dir = base_dir + '/output'
    detector = VehicleDetector(model=yolov5_model, coco_categories_of_interest=['bus'])
    vehicles = detector.detect(np_img)

    for i, v in enumerate(vehicles):
        crop = ImageHandler.crop(np_img, v['points'])
        ImageHandler.write_to_file(files_dir + img_uid + '/output/' + 'v_%d.png' % i, crop)

    return vehicles

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
    app.run(port=3001, debug=True)
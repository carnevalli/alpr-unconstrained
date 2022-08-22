import sys
from turtle import shapesize
import cv2
import numpy as np
import re
import json

from glob						import glob
from os.path 					import splitext, basename, isfile
from src.utils 					import crop_region, image_files_from_folder, loadRegexPatterns
from src.drawing_utils			import draw_label, draw_losangle, write2img
from src.label 					import lread, Label, readShapes
from transform					import findsimilar

from pdb import set_trace as pause


YELLOW = (  0,255,255)
RED    = (  0,  0,255)

input_dir = sys.argv[1]
output_dir = sys.argv[2]

validation_regex_path = ''
suppress_transformations = False
generate_demo = False

if len(sys.argv) >= 4:
	validation_regex_path = sys.argv[3]

if len(sys.argv) >= 5:
	suppress_transformations = sys.argv[4] == '1'

if len(sys.argv) >= 6:
	generate_demo = sys.argv[5] == '1'

img_files = image_files_from_folder(input_dir)

regex_patterns = loadRegexPatterns(validation_regex_path)

for img_file in img_files:

	bname = splitext(basename(img_file))[0]

	original_image = None

	if generate_demo:
		original_image = cv2.imread(img_file)

	detected_cars_labels = '%s/%s_cars.txt' % (output_dir,bname)

	vehicle_labels = lread(detected_cars_labels)

	report = {
		"name" : bname,
		"img_file" : img_file,
		"vehicles": []
	}

	sys.stdout.write('%s' % bname)

	if vehicle_labels:

		for i,vehicle_label in enumerate(vehicle_labels):

			
			vehicle_report = {
				'class' : vehicle_label.cl(),
				"conf" : vehicle_label.prob(),
				'img': '%s/%s_%d_car.jpg' % (output_dir,bname,i),
				'coords' : {
					'tlx' : vehicle_label.tl()[0],
					'tly' : vehicle_label.tl()[1],
					'brx' : vehicle_label.br()[0],
					'bry' : vehicle_label.br()[1],
				},
				'lps': []
			}

			if generate_demo:
				draw_label(original_image,vehicle_label,color=YELLOW,thickness=3)

			lp_labels_str = sorted(glob('%s/%s_%d_car_*_lp_str.txt' % (output_dir,bname,i)))

			for lp_label_str in lp_labels_str:
				lp_shapes_file = lp_label_str.replace('_str', '')
				if isfile(lp_label_str.replace('_str', '')):
					if isfile(lp_label_str):
						
						lp_str = ''

						# LP from OCR
						with open(lp_label_str,'r') as f:
							lp_str = f.read().strip()

						# transformation to find valid similar LP strings
						lp_similar = []
						if not suppress_transformations:
							lp_similar = findsimilar(lp_str, regex_patterns)

						matches = []
						if regex_patterns:
							for pattern_id, pattern in regex_patterns:
								ms = re.findall(pattern, lp_str, flags=re.IGNORECASE)

								for m in ms: 
									matches.append((pattern_id, m))

						lp_shapes = readShapes(lp_shapes_file)

						vehicle_report['lps'].append({
							"img": lp_shapes_file.replace('.txt', '.jpg'),
							"pts" : lp_shapes[0].pts.tolist(),
							"ocr" : lp_str,
							"matches" : matches,
							"similar" : lp_similar,
						})

						if generate_demo:
							pts = lp_shapes[0].pts * vehicle_label.wh().reshape(2,1) + vehicle_label.tl().reshape(2,1)
							ptspx = pts * np.array(original_image.shape[1::-1], dtype=float).reshape(2,1)

							draw_losangle(original_image,ptspx,RED,3)
						
							lp_label = Label(0,tl=pts.min(1),br=pts.max(1))
							write2img(original_image, lp_label, lp_str)

						sys.stdout.write(',%s' % lp_str)

			report["vehicles"].append(vehicle_report)
	if generate_demo:
		cv2.imwrite('%s/%s_output.png' % (output_dir,bname), original_image)
	sys.stdout.write('\n')
	with open('%s/%s_report.json' % (output_dir,bname), 'wt') as out_file:
		json.dump(report, out_file, indent=4)



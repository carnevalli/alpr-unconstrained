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

if len(sys.argv) >= 4:
	validation_regex_path = sys.argv[3]

if len(sys.argv) >= 5:
	suppress_transformations = sys.argv[4] == '1'

img_files = image_files_from_folder(input_dir)

regex_patterns = loadRegexPatterns(validation_regex_path)

for img_file in img_files:

	bname = splitext(basename(img_file))[0]

	I = cv2.imread(img_file)

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
				'coords' : {
					'class' : vehicle_label.cl(),
					'tlx' : vehicle_label.tl()[0],
					'tly' : vehicle_label.tl()[1],
					'brx' : vehicle_label.br()[0],
					'bry' : vehicle_label.br()[1],
					"conf" : vehicle_label.prob()
				},
				"img": '%s/%s_%d_car.jpg' % (output_dir,bname,i),
				'lps': []
			}

			draw_label(I,vehicle_label,color=YELLOW,thickness=3)

			lp_labels_str = glob('%s/%s_%d_car_*_lp_str.txt' % (output_dir,bname,i))

			for j in range(len(lp_labels_str)):
				lp_shapes_file = lp_labels_str[j].replace('_str', '')
				if isfile(lp_labels_str[j].replace('_str', '')):
					if isfile(lp_labels_str[j]):
						
						lp_str = ''

						# LP from OCR
						with open(lp_labels_str[j],'r') as f:
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
							
							if len(matches) == 0:
								continue

						lp_shapes = readShapes(lp_shapes_file)
						pts = lp_shapes[0].pts * vehicle_label.wh().reshape(2,1) + vehicle_label.tl().reshape(2,1)
						ptspx = pts * np.array(I.shape[1::-1], dtype=float).reshape(2,1)
						draw_losangle(I,ptspx,RED,3)
						
						
						lp_label = Label(0,tl=pts.min(1),br=pts.max(1))
						write2img(I, lp_label, lp_str)

						sys.stdout.write(',%s' % lp_str)

						vehicle_report['lps'].append({
							"img": '%s/%s_%d_car_%d_lp.jpg' % (output_dir,bname, i, j),
							"pts" : lp_shapes[0].pts.tolist(),
							"ocr" : lp_str,
							"matches" : matches,
							"similar" : lp_similar,
						})

			report["vehicles"].append(vehicle_report)

	cv2.imwrite('%s/%s_output.png' % (output_dir,bname),I)
	sys.stdout.write('\n')
	with open('%s/%s_report.json' % (output_dir,bname), 'wt') as out_file:
		json.dump(report, out_file, indent=4)



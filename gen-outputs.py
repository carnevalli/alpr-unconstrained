import sys
import cv2
import numpy as np
import re

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

	car_labels = lread(detected_cars_labels)

	sys.stdout.write('%s' % bname)

	if car_labels:

		for i,car_label in enumerate(car_labels):

			draw_label(I,car_label,color=YELLOW,thickness=3)

			lp_labels = glob('%s/%s_%d_car_*_lp.txt' % (output_dir,bname,i))
			lp_labels_str = glob('%s/%s_%d_car_*_lp_str.txt' % (output_dir,bname,i))

			for i in range(len(lp_labels)):
				if isfile(lp_labels[i]):
					if isfile(lp_labels_str[i]):
						
						lp_str = ''

						# LP from OCR
						with open(lp_labels_str[i],'r') as f:
							lp_str = f.read().strip()

						# transformation to find valid similar LP strings
						lp_similar = []
						if not suppress_transformations:
							lp_similar = findsimilar(lp_str, regex_patterns)

						if regex_patterns:
							matches = []
							for pattern_id, pattern in regex_patterns:
								m = re.findall(pattern, lp_str, flags=re.IGNORECASE)

								if len(m) > 0:
									matches.append((pattern_id, lp_str))
							
							if len(matches) == 0:
								continue

						lp_shapes = readShapes(lp_labels[i])
						pts = lp_shapes[0].pts * car_label.wh().reshape(2,1) + car_label.tl().reshape(2,1)
						ptspx = pts * np.array(I.shape[1::-1], dtype=float).reshape(2,1)
						draw_losangle(I,ptspx,RED,3)
						
						
						lp_label = Label(0,tl=pts.min(1),br=pts.max(1))
						write2img(I, lp_label, lp_str)

						sys.stdout.write(',%s' % lp_str)

	cv2.imwrite('%s/%s_output.png' % (output_dir,bname),I)
	sys.stdout.write('\n')



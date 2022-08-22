import sys
import cv2
import numpy as np
import traceback

import darknet.python.darknet as dn

from src.label 				import Label, lwrite
from os.path 				import splitext, basename, isdir
from os 					import makedirs
from src.utils 				import crop_region, image_files_from_folder
from darknet.python.darknet import detect


if __name__ == '__main__':

	try:
	
		input_dir  = sys.argv[1]
		output_dir = sys.argv[2]

		vehicle_threshold = .5
		max_vehicles = 0
		vehicles_order = 'area'
		coco_categories_of_interest = ['car', 'bus']
		whole_image_fallback = False

		if len(sys.argv) >= 4:
			vehicle_threshold = float(sys.argv[3]) / 100

		if len(sys.argv) >= 5:
			coco_categories_of_interest = sys.argv[4].split(",")

		if len(sys.argv) >= 6:
			max_vehicles = int(sys.argv[5])

		if len(sys.argv) >= 7:
			vehicles_order = sys.argv[6]

		if len(sys.argv) >= 8:
			whole_image_fallback = True if sys.argv[7] == '1' else False

		vehicle_weights = b'data/vehicle-detector/yolo-voc.weights'
		vehicle_netcfg  = b'data/vehicle-detector/yolo-voc.cfg'
		vehicle_dataset = b'data/vehicle-detector/voc.data'

		vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
		vehicle_meta = dn.load_meta(vehicle_dataset)

		imgs_paths = image_files_from_folder(input_dir)
		imgs_paths.sort()

		if not isdir(output_dir):
			makedirs(output_dir)

		print('Searching for vehicles using YOLO. Threshold: %.0f' % (vehicle_threshold * 100))
		print('Categories of interest: %d [%s]' % (len(coco_categories_of_interest), ','.join(coco_categories_of_interest)))

		for i,img_path in enumerate(imgs_paths):

			print('\tScanning %s' % img_path)

			bname = basename(splitext(img_path)[0])

			detection_results, image_sizes = detect(vehicle_net, vehicle_meta, img_path ,thresh=vehicle_threshold)
			
			vehicles = [r for r in detection_results if r[0] in coco_categories_of_interest]
			found_vehicles_labels = [v[0] for v in vehicles]

			print('\t\t%d vehicles found: %s' % (len(vehicles), ', '.join(found_vehicles_labels)))

			labels = []
			if len(vehicles):

				original_image = cv2.imread(img_path)
				image_sizes = np.array(image_sizes,dtype=float)

				crops = []

				for i,r in enumerate(vehicles):

					label, confidence, coords = r
					crops.append(
						{
							"label": label,
							"coords": coords,
							"confidence" : confidence
						}
					)

				crops.sort(key=lambda c : c['coords'][2] * c['coords'][3] if vehicles_order == 'area' else c['confidence'], reverse=True)
				crops = crops[: len(crops) if max_vehicles == 0 or max_vehicles >= len(crops) else max_vehicles]

				for i,r in enumerate(crops):

					area_x, area_y, area_width, area_height = (np.array(r['coords'])/np.concatenate( (image_sizes, image_sizes) )).tolist()
					top_left = np.array([area_x - area_width / 2., area_y - area_height / 2.])
					bottom_right = np.array([area_x + area_width / 2., area_y + area_height /2.])
					
					textual_label = Label(r['label'], top_left, bottom_right, r['confidence'])
					vehicle_label = crop_region(original_image, textual_label)

					labels.append(textual_label)

					cv2.imwrite('%s/%s_%d_car.png' % (output_dir,bname,i), vehicle_label)
			else:
				if whole_image_fallback:
					original_image = cv2.imread(img_path)

					height, width, channels = original_image.shape

					textual_label = Label('fallback', np.array([0, 0]), np.array([1, 1]), 0)
					vehicle_label = crop_region(original_image, textual_label)

					labels.append(textual_label)
					cv2.imwrite('%s/%s_%d_car.png' % (output_dir,bname, 0), vehicle_label)
			
			if labels:
				lwrite('%s/%s_cars.txt' % (output_dir,bname),labels)
	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
	
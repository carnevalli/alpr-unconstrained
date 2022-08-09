import sys
import cv2
import numpy as np
import traceback
import torch

from src.label 				import Label, lwrite
from os.path 				import splitext, basename, isdir
from os 					import makedirs
from src.utils 				import crop_region, image_files_from_folder


if __name__ == '__main__':

	try:
	
		input_dir  = sys.argv[1]
		output_dir = sys.argv[2]

		vehicle_threshold = .5
		max_vehicles = 0
		max_vehicles_order_by = 'area'
		coco_categories_of_interest = ['car', 'bus']

		if len(sys.argv) >= 4:
			vehicle_threshold = float(sys.argv[3]) / 100

		if len(sys.argv) >= 5:
			coco_categories_of_interest = sys.argv[4].split(",")

		if len(sys.argv) >= 6:
			max_vehicles = int(sys.argv[5])

		if len(sys.argv) >= 7:
			max_vehicles_order_by = sys.argv[6]

		model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

		imgs_paths = image_files_from_folder(input_dir)
		imgs_paths.sort()

		if not isdir(output_dir):
			makedirs(output_dir)

		print('Searching for vehicles using YOLOv5. Threshold: %.0f' % (vehicle_threshold * 100))
		print('Categories of interest: %d [%s]' % (len(coco_categories_of_interest), ','.join(coco_categories_of_interest)))

		for i,img_path in enumerate(imgs_paths):

			print('\tScanning %s' % img_path)

			bname = basename(splitext(img_path)[0])

			detection_results = model(img_path, size=640).pandas().xyxy[0]
			vehicles = detection_results.loc[detection_results['name'].isin(coco_categories_of_interest) & detection_results['confidence'] > 0].copy()

			vehicles['area'] =  (vehicles['xmax'] - vehicles['xmin']) * (vehicles['ymax'] - vehicles['ymin'])
			vehicles.sort_values('confidence' if max_vehicles_order_by == 'confidence' else 'area' , ascending=False, inplace=True)

			if max_vehicles > 0:
				vehicles = vehicles[:max_vehicles]

			found_vehicles_labels = vehicles['name'].values
			print('\t\t%d vehicles found: %s' % (len(vehicles), ', '.join(found_vehicles_labels)))

			if len(vehicles):
				original_image = cv2.imread(img_path)

				height, width, channels = original_image.shape
				labels = []

				vehicles['top_left_x'] = vehicles['xmin'] / width
				vehicles['top_left_y'] = vehicles['ymin'] / height

				vehicles['bottom_right_x'] = vehicles['xmax'] / width
				vehicles['bottom_right_y'] = vehicles['ymax'] / height

				for i, r in vehicles.iterrows():
					
					top_left = np.array([r['top_left_x'], r['top_left_y']])
					bottom_right = np.array([r['bottom_right_x'], r['bottom_right_y']])

					textual_label = Label(0, top_left, bottom_right)
					vehicle_label = crop_region(original_image, textual_label)

					labels.append(textual_label)

					cv2.imwrite('%s/%s_%d_car.png' % (output_dir,bname,i), vehicle_label)

				lwrite('%s/%s_cars.txt' % (output_dir,bname),labels)

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
	
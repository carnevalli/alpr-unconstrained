import sys, os
import cv2
import traceback

from src.keras_utils 			import load_model
from glob 						import glob
from os.path 					import splitext, basename
from src.utils 					import im2single
from src.keras_utils 			import load_model, detect_lp
from src.label 					import Shape, writeShapes


def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))


if __name__ == '__main__':

	try:
		
		input_dir  = sys.argv[1]
		output_dir = input_dir

		wpod_net_path = sys.argv[2]
		wpod_net = load_model(wpod_net_path)

		lp_threshold = .5

		if len(sys.argv) >= 4:
			lp_threshold = float(sys.argv[3]) / 100

		imgs_paths = glob('%s/*car.png' % input_dir)

		print('Searching for license plates using WPOD-NET. Threshold: %.0f' % (lp_threshold * 100))

		for i,img_path in enumerate(imgs_paths):

			print('\t Processing %s' % img_path)

			bname = splitext(basename(img_path))[0]
			vehicle_image = cv2.imread(img_path)

			ratio = float(max(vehicle_image.shape[:2]))/min(vehicle_image.shape[:2])
			side  = int(ratio*288.)
			bound_dim = min(side + (side%(2**4)),608)
			print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

			lp_labels, lp_images, _ = detect_lp(wpod_net,im2single(vehicle_image),bound_dim,2**4,(240,80),lp_threshold)

			print('\t\tPossible LP found: %d' % len(lp_images))

			if len(lp_images):
				for j in range(len(lp_images)):
					lp_image = lp_images[j]
					lp_image = cv2.cvtColor(lp_image, cv2.COLOR_BGR2GRAY)
					lp_image = cv2.cvtColor(lp_image, cv2.COLOR_GRAY2BGR)

					s = Shape(lp_labels[j].pts)

					cv2.imwrite('%s/%s_%d_lp.png' % (output_dir,bname,j),lp_image*255.)
					writeShapes('%s/%s_%d_lp.txt' % (output_dir,bname,j),[s])

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)



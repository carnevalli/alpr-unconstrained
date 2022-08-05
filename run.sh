#!/bin/bash

check_file() 
{
	if [ ! -f "$1" ]
	then
		return 0
	else
		return 1
	fi
}

check_dir() 
{
	if [ ! -d "$1" ]
	then
		return 0
	else
		return 1
	fi
}


# Check if Darknet is compiled
check_file "darknet/libdarknet.so"
retval=$?
if [ $retval -eq 0 ]
then
	echo "Darknet is not compiled! Go to 'darknet' directory and 'make'!"
	exit 1
fi

lp_model="data/lp-detector/wpod-net_update1.h5"
input_dir=''
output_dir=''
csv_file=''
vehiclep_files=0
vehicle_detection_threshold=50
lp_detection_threshold=50
ocr_detection_threshold=40
coco_categories="car,bus"
yolo_version=5
vehicle_only=0
lp_only=0
ocr_only=0
max_vehicles=0
max_vehicles_order_by="area"

# Check # of arguments
usage() {
	echo ""
	echo " Usage:"
	echo ""
	echo "   bash $0 -i input/dir -o output/dir -c csv_file.csv [-h] [-l path/to/model]:"
	echo ""
	echo "   -i, --input-dir   Input dir path (containing JPG or PNG images)"
	echo "   -o, --output-dir   Output dir path"
	echo "   -c, --csv-file   Output CSV file path"
	echo "   -l, --lp-model   Path to Keras LP detector model (default = $lp_model)"
	echo "   -k, --keep-files   Keep temporary files in outpuvehicleath"
	echo "   --vehicle-threshold  Vehicle detection threshold (default: $vehicle_detection_threshold, min: 1, max: 100)"
	echo "   --lp-threshold  LP detection threshold (default: $lp_detection_threshold, min: 1, max: 100)"
	echo "   --ocr-threshold  LP OCR detection threshold (default: $ocr_detection_threshold, min: 1, max: 100)"
	echo "   --max-vehicles Limits the number of detected vehicles ordering by descending image area occupied by each vehicle. (default: no limit)"
	echo "   --max-vehicles-order-by Defines the criteria used to filter max_vehicles option. Options: area, confidence"
	echo "   --coco-categories Comma-separated set of categories for object detection from cocodataset.org. (default: $coco_categories)"
	echo "   --yolo-voc Use YOLOv2-voc vehicle detection model instead of YOLOv5s-coco"
	echo "   --vehicle-only Stops image processing after vehicle detection stage"
	echo "   --lp-only Stops the image processing after license plate detection stage."
	echo "   --ocr-only Stops the image processing after OCR stage."
	echo "   -h, --help   Print this help information"
	echo ""
	exit 1
}

POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input-dir)
      input_dir="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--output-dir)
      output_dir="$2"
      shift # past argument
      shift # past value
      ;;
	-c|--csv-dir)
      csv_file="$2"
      shift # past argument
      shift # past value
      ;;
	-l|--lp-model)
      lp_model="$2"
      shift # past argument
      shift # past value
      ;;
    -k|--keep-files)
      keep_files=1
      shift # past argument
      ;;
	--vehicle-threshold)
	  vehicle_detection_threshold="$2"
	  shift
	  ;;
	--lp-threshold)
	  lp_detection_threshold="$2"
	  shift
	  ;;
	--ocr-threshold)
	  ocr_detection_threshold="$2"
	  shift
	  ;;
	--max-vehicles)
	  max_vehicles="$2"
	  shift
	  ;;
	--max-vehicles-order-by)
	  max_vehicles_order_by="$2"
	  shift
	  ;;
	--coco-categories)
	  coco_categories="$2"
	  shift
	  ;;
	--yolo-voc)
	  yolo_version=2
	  shift
	  ;;
	--vehicle-only)
	  vehicle_only=1
	  shift
	  ;;
	--lp-only)
	  lp_only=1
	  shift
	  ;;
	--ocr-only)
	  ocr_only=1
	  shift
	  ;;
	-h|--help)
      usage
      shift # past argument
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [ -z "$input_dir"  ]; then echo "Input dir not set."; usage; exit 1; fi
if [ -z "$output_dir" ]; then echo "Ouput dir not set."; usage; exit 1; fi
if [ -z "$csv_file"   ]; then echo "CSV file not set." ; usage; exit 1; fi
if [ $vehicle_detection_threshold -lt 1 ] || [ $vehicle_detection_threshold -gt 100 ]; then echo "Vehicle detection threshold must be between 1 and 100" ; usage; exit 1; fi
if [ $lp_detection_threshold -lt 1 ] || [ $lp_detection_threshold -gt 100 ]; then echo "LP detection threshold must be between 1 and 100" ; usage; exit 1; fi
if [ $ocr_detection_threshold -lt 1 ] || [ $ocr_detection_threshold -gt 100 ]; then echo "OCR detection threshold must be between 1 and 100" ; usage; exit 1; fi
if [ $yolo_version -ne 2 ] && [ $yolo_version -ne 5 ]; then echo "Invalid YOLO version. Please provide some of these available version numbers: [3,5]"; usage; exit 1; fi
if [ $max_vehicles_order_by != "area" ] && [ $max_vehicles_order_by != "confidence" ]; then echo "Invalid max_vehicles_order_by option. Valid options: area, confidence"; usage; exit 1; fi

# Check if input dir exists
check_dir $input_dir
retval=$?
if [ $retval -eq 0 ]
then
	echo "Input directory ($input_dir) does not exist"
	exit 1
fi

# Check if output dir exists, if not, create it
check_dir $output_dir
retval=$?
if [ $retval -eq 0 ]
then
	mkdir -p $output_dir
fi

# End if any error occur
set -e

# Detect vehicles
python vehicle-detection-v$yolo_version.py $input_dir $output_dir $vehicle_detection_threshold $coco_categories $max_vehicles $max_vehicles_order_by

if [ $vehicle_only -eq 0 ]
then
	# Detect license plates
	python license-plate-detection.py $output_dir $lp_model $lp_detection_threshold

	if [ $lp_only -eq 0 ]
	then
		# OCR
		python license-plate-ocr.py $output_dir $ocr_detection_threshold

		if [ $ocr_only -eq 0 ]
		then
			# Draw output and generate list
			python gen-outputs.py $input_dir $output_dir > $csv_file
		fi
	fi
fi

# Clean files temporary files
if [ $keep_files -eq 0 ]
then 
	rm $output_dir/*_lp.png 2> /dev/null
	rm $output_dir/*car.png 2> /dev/null
	rm $output_dir/*_cars.txt 2> /dev/null
	rm $output_dir/*_lp.txt 2> /dev/null
	rm $output_dir/*_str.txt 2> /dev/null
fi
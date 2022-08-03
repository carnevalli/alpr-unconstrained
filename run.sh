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
keep_files=0
car_detection_threshold=50
lp_detection_threshold=50
ocr_detection_threshold=40
coco_categories="car,bus"


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
	echo "   -k, --keep-files   Keep temporary files in output path"
	echo "   --car-threshold  Car detection threshold (default: $car_detection_threshold, min: 1, max: 100)"
	echo "   --lp-threshold  LP detection threshold (default: $lp_detection_threshold, min: 1, max: 100)"
	echo "   --ocr-threshold  LP OCR detection threshold (default: $ocr_detection_threshold, min: 1, max: 100)"
	echo "   --coco-categories Comma-separated set of categories for object detection from cocodataset.org. (default: $coco_categories)"
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
	--car-threshold)
	  car_detection_threshold="$2"
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
	--coco-categories)
	  coco_categories="$2"
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
if [ $car_detection_threshold -lt 1 ] || [ $car_detection_threshold -gt 100 ]; then echo "Car detection threshold must be between 1 and 100" ; usage; exit 1; fi

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
python vehicle-detection.py $input_dir $output_dir $car_detection_threshold $coco_categories

# Detect license plates
python license-plate-detection.py $output_dir $lp_model $lp_detection_threshold

# OCR
python license-plate-ocr.py $output_dir $ocr_detection_threshold

# Draw output and generate list
python gen-outputs.py $input_dir $output_dir > $csv_file

# Clean files temporary files
if [ $keep_files -eq 0 ]
then 
	rm $output_dir/*_lp.png
	rm $output_dir/*car.png
	rm $output_dir/*_cars.txt
	rm $output_dir/*_lp.txt
	rm $output_dir/*_str.txt
fi
from math import ceil, floor
import numpy as np

class VehicleDetector:

    def __init__(self, model=None,
        vehicle_threshold=.5,
        max_vehicles=None, vehicles_order='area', 
        coco_categories_of_interest=['bus', 'car', 'truck'],
        whole_image_fallback=False):
        
        self.model = model
        self.vehicle_threshold = vehicle_threshold
        self.max_vehicles = max_vehicles
        self.vehicles_order = vehicles_order
        self.coco_categories_of_interest = coco_categories_of_interest
        self.whole_image_fallback = whole_image_fallback

    def extract_vehicle_points(self, image, vehicle):
        left_column = floor(vehicle['xmin'])
        top_row = floor(vehicle['ymin'])
        right_column = ceil(vehicle['xmax'])
        bottom_row = ceil(vehicle['ymax'])

        return np.array([top_row, bottom_row, left_column, right_column])

    def calculate_relative_points(self, points, shape):
        return points / np.array([shape[0], shape[0], shape[1], shape[1]])

    def generate_label(self, category, shape, points, confidence):
        return {
            "category": category,
            "points": points,
            "rpoints": self.calculate_relative_points(points, shape),
            "confidence": confidence
        }

    def detect(self, np_image):
        print('Searching for vehicles using YOLOv5. Threshold: %.0f' % (self.vehicle_threshold * 100))
        print('Categories of interest: %d [%s]' % 
            (len(self.coco_categories_of_interest), ','.join(self.coco_categories_of_interest)))

        detection_results = self.model(np_image, size=640).pandas().xyxy[0]
        vehicles = detection_results.loc[detection_results['name'].isin(self.coco_categories_of_interest) & detection_results['confidence'] > 0].copy()

        vehicles['area'] =  (vehicles['xmax'] - vehicles['xmin']) * (vehicles['ymax'] - vehicles['ymin'])
        vehicles.sort_values('confidence' if self.vehicles_order == 'confidence' else 'area' , ascending=False, inplace=True)

        if self.max_vehicles:
            vehicles = vehicles[:self.max_vehicles]

        found_vehicles_labels = vehicles['name'].values
        print('\t\t%d vehicles found: %s' % (len(vehicles), ', '.join(found_vehicles_labels)))

        labels = []
        if len(vehicles):
            sequence = 0

            for _, v in vehicles.iterrows():
                points = self.extract_vehicle_points(np_image, v)
                labels.append(self.generate_label(v['name'], np_image.shape, points, v['confidence']))
        else:
            if self.whole_image_fallback:
                points = np.array([0, np_image.shape[0], 0, np_image.shape[1]])
                labels.append(self.generate_label('fallback', points, 0.))
        
        return labels

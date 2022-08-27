import numpy as np
import re
from classes.LicensePlateTransformation import LicensePlateTransformation

from classes.ImageHandler import ImageHandler

class OutputProcessor:
    def __init__(self, img_uid, 
        files_path="files/",
        output_folder="output/",
        validation_regex_list=None,
        suppress_transformations=False, 
        generate_demo=False, 
        demo_filename="demo.png",
        generate_vehicles=False):

        self.img_uid = img_uid
        self.files_path = files_path
        self.output_folder = output_folder
        self.validation_regex_list = validation_regex_list
        self.suppress_transformations = suppress_transformations
        self.generate_demo = generate_demo
        self.demo_filename = demo_filename
        self.generate_vehicles = generate_vehicles

    @staticmethod
    def get_lp_points(points, vehicle_image_shape, top_left_points):
        v_shape = np.array(vehicle_image_shape).reshape(2,1)
        tl_shape = np.array(top_left_points).reshape(2,1)
        return points * v_shape + tl_shape

    def get_output_path(self):
        return self.files_path + self.img_uid +  '/' + self.output_folder

    def process(self, np_image, vehicles):
        for i, vehicle in enumerate(vehicles):
            if self.generate_vehicles:
                crop = ImageHandler.crop(np_image, vehicle['points'])
                ImageHandler.write_to_file( self.get_output_path() + 'v_%d.png' % i, crop)

            if self.generate_demo:
                np_vehicle = ImageHandler.crop(np_image, vehicle['points'])
                ImageHandler.draw_vehicle_shape(np_image, vehicle['points'], (0,255,255), thickness=3)
                vehicle_lps = []

                for lp in vehicle['lps']:
                    pts = OutputProcessor.get_lp_points(lp[1], (np_vehicle.shape[1], np_vehicle.shape[0]), (vehicle['points'][2], vehicle['points'][0]))
                    ImageHandler.draw_losangle(np_image, pts, (0,0,255), 3)
                    ImageHandler.write2img(np_image, pts, lp[0], font_size=2)

                    if not self.suppress_transformations and self.validation_regex_list:
                        vehicles[i]['similar'] = LicensePlateTransformation.findsimilar(lp[0], self.validation_regex_list)

                        matches = []
                        if self.validation_regex_list:
                            for pattern_id, pattern in self.validation_regex_list:
                                ms = re.findall(pattern, lp[0], flags=re.IGNORECASE)

                                for m in ms: 
                                    matches.append((pattern_id, m))
                            vehicles[i]['matches'] = matches


        if self.generate_demo:
            ImageHandler.write_to_file(self.get_output_path() + self.demo_filename, np_image)

        return vehicles
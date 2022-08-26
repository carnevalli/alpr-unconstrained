import numpy as np

class OutputProcessor:
    def __init__(self, validation_regex_path=None,
        suppress_transformations=False, generate_demo=False):
        self.validation_regex_path = validation_regex_path
        self.suppress_transformations = suppress_transformations
        self.generate_demo = generate_demo

    @staticmethod
    def get_lp_points(points, vehicle_image_shape, top_left_points):
        v_shape = np.array(vehicle_image_shape).reshape(2,1)
        tl_shape = np.array(top_left_points).reshape(2,1)
        return points * v_shape + tl_shape
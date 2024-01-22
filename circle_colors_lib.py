import numpy as np
from colour import RGB_to_XYZ, SDS_ILLUMINANTS, XYZ_to_Lab, delta_E
class circle_colors:

    def calculate_delta_E (self, reference, observe):
        observe = XYZ_to_Lab(observe)
        reference = XYZ_to_Lab(reference)
        deltas = np.zeros((observe.shape[0], 1))
        for i in range(observe.shape[0]):
            a = reference[i, :]
            b = observe[i, :]
            deltas[i] = delta_E(a, b, method="CIE 1976")
        return deltas

    def find_CA(self, pic):
        pass

    def find_circles(self, pic):
        pass
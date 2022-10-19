import numpy as np
import matplotlib.pyplot as plt

class PF(object):
    def __init__(self, pf_function, bearing_range=(-180,180), num_samples=360):
        self.pf_function = pf_function
        self.bearing_range = bearing_range
        b_min, b_max = bearing_range
        self.x_data = np.linspace(b_min, b_max, num=num_samples)
        self.y_data = np.zeros_like(self.x_data)
    
    def gen_potential_field(self, bearing, intensity=None, intensity_max=1):
        if intensity is None:
            intensity = intensity_max
        self.y_data = self.pf_function(self.x_data, bearing, intensity, intensity_max)

def goal_potential_field(x_array, bearing, intensity, intensity_max):
    '''
    Piece-wise triangular function.
    '''
    
    m = intensity_max/x_array.max() # Gradients
    c = (intensity - m*bearing, intensity + m*bearing)
    pw_res = np.piecewise(x_array, [x_array < bearing, x_array >= bearing], [lambda x: m*x+c[0], lambda x: -m*x+c[1]])

    # Clip results below zero and above intensity maximum:
    pw_res[pw_res<0] = 0
    pw_res[pw_res>intensity_max] = intensity_max
    
    return pw_res

if __name__ == '__main__':
    goal_pf = PF(goal_potential_field)
    
    goal_pf.gen_potential_field(90.0)
    
    plt.plot(goal_pf.x_data, goal_pf.y_data)
    plt.show()
    
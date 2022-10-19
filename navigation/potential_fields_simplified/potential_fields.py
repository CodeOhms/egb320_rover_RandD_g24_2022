import numpy as np
import matplotlib.pyplot as plt

class PF(object):
    def __init__(self, pf_function, bearing_range=(-180,180), num_samples=360):
        self.pf_function = pf_function
        self.bearing_range = bearing_range
        b_min, b_max = bearing_range
        self.x_data = np.linspace(b_min, b_max, num=num_samples)
        self.y_data = np.zeros_like(self.x_data)
    
    def gen_potential_field(self, bearings, intensity=None, intensity_max=1):
        if intensity is None:
            intensity = intensity_max
        pf_res = self.pf_function(self.x_data, bearings, intensity, intensity_max)
        
        # Clip results below zero and above intensity maximum:
        pf_res[pf_res<0] = 0
        pf_res[pf_res>intensity_max] = intensity_max
        
        self.y_data = pf_res

def goal_potential_field(x_array, bearings, intensity, intensity_max):
    '''
    Piecewise triangular function.
    '''
    
    bearing = bearings[0]
    
    m = intensity_max/x_array.max() # Gradients
    c = (intensity - m*bearing, intensity + m*bearing)
    return np.piecewise(x_array, [x_array < bearing, x_array >= bearing], [lambda x: m*x+c[0], lambda x: -m*x+c[1]])

def hazard_potential_field(x_array, bearings, intensity, intensity_max):
    '''
    Piecewise square function.
    '''
    
    b_min = bearings[0]
    b_max = bearings[1]
    return np.where(((b_min <= x_array) & (x_array <= b_max)), intensity, 0)
    

if __name__ == '__main__':
    goal_pf = PF(goal_potential_field)
    haz_pf = PF(hazard_potential_field)
    
    goal_pf.gen_potential_field((90.0,), 0.5)
    haz_pf.gen_potential_field((-140, -120), 0.7)
    
    plt.figure()
    plt.plot(goal_pf.x_data, goal_pf.y_data)
    plt.figure()
    plt.plot(haz_pf.x_data, haz_pf.y_data)
    plt.show()
    
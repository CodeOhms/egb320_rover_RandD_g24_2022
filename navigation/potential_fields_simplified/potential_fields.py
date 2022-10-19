import numpy as np
import matplotlib.pyplot as plt

class PF(object):
    def __init__(self, pf_function, bearing_range=(-180,180), num_samples=360, y_data=None):
        self.pf_function = pf_function
        self.bearing_range = bearing_range
        b_min, b_max = bearing_range
        self.x_data = np.linspace(b_min, b_max, num=num_samples)
        if y_data is None:
            self.y_data = np.zeros_like(self.x_data)
        else:
            self.y_data = y_data
    
    # Alternative initialiser:
    @classmethod
    def heading_field(cls, goal_pot_f, haz_pot_f):
        heading_pf = goal_pot_f.y_data - haz_pot_f.y_data
        
        # Clip results below zero:
        heading_pf[heading_pf<0] = 0

        b_range = (goal_pot_f.x_data.min(), goal_pot_f.x_data.max())
        n_samp = goal_pot_f.x_data.size
        return cls(None, bearing_range=b_range, num_samples=n_samp, y_data=heading_pf)
    
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
    
    
    goal_pf.gen_potential_field((0.0,), 0.6)
    haz_pf.gen_potential_field((-80, -20), 0.8)
    
    motor_heading_pf = PF.heading_field(goal_pf, haz_pf)
    
    plt.figure()
    plt.plot(goal_pf.x_data, goal_pf.y_data)
    plt.title('Goal Potential Field')
    
    plt.figure()
    plt.plot(haz_pf.x_data, haz_pf.y_data)
    plt.title('Hazard Potential Field')
    
    plt.figure()
    plt.plot(motor_heading_pf.x_data, motor_heading_pf.y_data)
    plt.title('Motor Heading Potential Field')
    plt.show()
    
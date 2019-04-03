import slime_mold_1D_final as slime1D
import scipy as sci
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from numpy.linalg import norm
from scipy.integrate import solve_ivp
import pickle

#I Parameters
param_dict = {}
#I.1 Timing/grid/initial positions
param_dict['N'] = 100 #number of particles for 1D or sqrt of number of partivles for 2D
param_dict['T'] = 2 #final time
param_dict['cell_start'] =  -2.1 #start of spacial grid
param_dict['cell_end'] = 2.1 #end of spacial grid
param_dict['h'] = np.divide(param_dict['cell_end'] - param_dict['cell_start'], param_dict['N']) #stepsize in spacial grid
param_dict['particle_cells'] = np.linspace(param_dict['cell_start'],param_dict['cell_end'], param_dict['N'] + 1) #spacial grid cells
param_dict['x0'] = param_dict['particle_cells'][0:param_dict['N']] + np.divide(param_dict['h'],2) #particle initial posistions (middle of each cell)
param_dict['maximum_step_size'] = .1 ##maximum step-size for the solver

#I.2 Model parameters
param_dict['m'] = 1 #diffusion exponent
param_dict['eps'] = np.power(param_dict['h'], .99) #mollifier parameter - see Carillo paper
param_dict['initial_mass_profile'] = 1 ##shape of initial profile 1:gaussian, 2: double bump
param_dict['kernal_choice'] = 3 ##choice of interaction kernal 1: polynomial, 2: gaussian w/ variance .3, 3: quadratic
param_dict['M'] = np.vectorize(slime1D.initial_masses)(param_dict['x0'],param_dict['initial_mass_profile']) * param_dict['h'] #particle masses
param_dict['y0'] = 1 ##food source 1
param_dict['y1'] = -1 ##food source 2
param_dict['ABC_params'] = [(1,5,10),(5,1,10),(1,1,10),(10,1,1),(10,1,5),(10,5,1),(1,10,1),(1,10,5),(5,10,1),(1,1,1),(10,1,10),(1,10,10), (10,10,1)] #set of A,B and C values to sweep over

#I.3 plotting/saving parameters
param_dict['plot_times'] = [1]  #times you want to plot a trajectory for
filename ='quadratic_kernal_422019'

def sweep_1D(ABC_parameters, N, x0, maximum_step_size, T, y0,y1,M, m, eps, kernal_choice, plot_times, filename):
    sol_dict = {}
    for set in ABC_parameters:

        sol = solve_ivp(lambda t,y:slime1D.x_prime(t,y,set[0],set[1],set[2],y0,y1,M,m,eps,kernal_choice), (0,T), x0, method = 'BDF', max_step = maximum_step_size)
        sol_dict[set] =  sol
        #slime1D.compute_and_plot_profiles(sol, M, plot_times, param_dict['cell_start'], param_dict['cell_end'], param_dict['eps'], param_dict['y0'], param_dict['y1'], 200, filename + "A%d_B%d,C%d" % (set[0], set[1], set[2]))

    return sol_dict


if __name__ == '__main__':
    sols = sweep_1D(param_dict['ABC_params'], param_dict['N'], param_dict['x0'], param_dict['maximum_step_size'], param_dict['T'], param_dict['y0'], param_dict['y1'], param_dict['M'], param_dict['m'], param_dict['eps'], param_dict['kernal_choice'], param_dict['plot_times'], filename)

    param_file = open(filename+'_param_dict.pkl', 'wb')
    sol_file = open(filename +'_solution_data.pkl', 'wb')

    pickle.dump(param_dict, param_file)
    pickle.dump(sols, sol_file)

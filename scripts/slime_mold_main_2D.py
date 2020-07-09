import slime_mold_2D_final as slime2D
import scipy as sci
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from numpy.linalg import norm
from scipy.integrate import solve_ivp
import pickle
from pathlib import Path
#I Parameters
param_dict = {}
#I.1 Timing/grid/initial positions
param_dict['N'] = 1#15 #number of particles for 1D or sqrt of number of partivles for 2D
param_dict['T'] = .1#6 #final time
param_dict['cell_start'] =  -2.1 #start of spacial grid
param_dict['cell_end'] = 2.1 #end of spacial grid
param_dict['h'] = np.divide(param_dict['cell_end'] - param_dict['cell_start'], param_dict['N']) #stepsize in spacial grid
param_dict['particle_cells'] = np.linspace(param_dict['cell_start'],param_dict['cell_end'], param_dict['N'] + 1) #spacial grid cells
x0_pre = param_dict['particle_cells'][0:param_dict['N']] + np.divide(param_dict['h'],2) #particle initial posistions (middle of each cell)
param_dict['x0'] = np.array(np.meshgrid(x0_pre, x0_pre)).T.reshape(param_dict['N']**2,2)
param_dict['maximum_step_size'] = .1 ##maximum step-size for the solver

#I.2 Model parameters
param_dict['m'] = 1 #diffusion exponent
param_dict['eps'] = np.power(param_dict['h'], .99) #mollifier parameter - see Carillo paper
param_dict['initial_mass_profile'] = 1 ##shape of initial profile 1:gaussian, 2: double bump
param_dict['kernal_choices'] = [1,2,3] ##choice of interaction kernal 1: polynomial, 2: gaussian w/ variance .3, 3: quadratic
M = np.apply_along_axis(slime2D.initial_masses, 1, param_dict['x0'],1) #particle masses
param_dict['M'] = M/np.sum(M)
param_dict['y0'] = np.array([1,0]) ##food source 1
param_dict['y1'] = y1 = np.array([-1,0]) ##food source 2
param_dict['ABC_params'] = [(10,10,1), (10,5,1)] #set of A,B and C values to sweep over

#I.3 plotting/saving parameters
param_dict['plot_profiles'] = 3  #times you want to plot a trajectory for

results_folder = 'test'#'../results/2020_2D_implementation_revision'
filename = 'test'#'/2020_2D_implementation_revision'
results_folder_path = Path(results_folder)
results_folder_path.mkdir(parents = True, exist_ok = True)

def sweep_2D(ABC_parameters, N, x0, maximum_step_size, T, y0,y1,M, m, eps, kernal_choices, plot_profiles, filename):
    sol_dict = {}
    x0 = np.squeeze(x0.reshape(2*(N**2),1,order = 'F'))
    kernal_names = ['polynomial', 'gaussian', 'quadratic']
    for choice in kernal_choices:
        kernal_name = kernal_names[choice -1]
        filename = filename = '_'+kernal_name
        for set in ABC_parameters:

            sol = solve_ivp(lambda t,y: slime2D.x_prime(t,y,set[0],set[1],set[2],y0,y1,N,M,m,eps,choice), (0,T), x0, method = 'BDF', max_step = maximum_step_size)
            sol_dict[set] =  sol

            #compute times to plot profiles
            steps = len(sol['y'][0, :])
            print(steps)
            steps_between_profiles = steps // plot_profiles
            plot_times = []
            plot_times = [i * steps_between_profiles for i in range(0,plot_profiles +1)]
            if steps % steps_between_profiles == 0:
                plot_times[len(plot_times) -1] = plot_times[len(plot_times) -1] -1
            else:
                plot_times.append(steps - 1)
            print(plot_times)

            for time in plot_times:
                slime2D.plot_profiles(sol, M, N, time, 200, param_dict['cell_start'], param_dict['cell_end'], param_dict['eps'], results_folder + filename + "A{}_B{}_C{}".format(set[0], set[1], set[2]) +' time = {}'.format(time))

    return sol_dict


if __name__ == '__main__':
    sols = sweep_2D(param_dict['ABC_params'], param_dict['N'], param_dict['x0'], param_dict['maximum_step_size'], param_dict['T'], param_dict['y0'], param_dict['y1'], param_dict['M'], param_dict['m'], param_dict['eps'], param_dict['kernal_choices'], param_dict['plot_profiles'], filename)

    param_file = open(results_folder+filename+'_param_dict.pkl', 'wb')
    sol_file = open(results_folder+filename +'_solution_data.pkl', 'wb')

    pickle.dump(param_dict, param_file)
    pickle.dump(sols, sol_file)

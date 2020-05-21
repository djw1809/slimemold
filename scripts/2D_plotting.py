import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pickle
import scipy
from slime_mold_1D_final import blob
import slime_mold_2D_final as slime
import seaborn as sns
#%%

with open('../../results/parameter_sweep_macro/2D/polynomial/polynomial_data.pickle', 'rb') as file:
    polynomial_data = pickle.load(file)


polynomial_data.keys()

test_data = polynomial_data[(5, 10,1)]
test_data['y'].shape

#%%
N = 15 #number of particles for 1D or sqrt of number of partivles for 2D
cell_start =  -2.1 #start of spacial grid
cell_end = 2.1 #end of spacial grid
h = np.divide(cell_end - cell_start, N) #stepsize in spacial grid
particle_cells = np.linspace(cell_start,cell_end, N + 1) #spacial grid cells
x0_pre = particle_cells[0:N] + np.divide(h,2) #particle initial posistions (middle of each cell)
x0 = np.array(np.meshgrid(x0_pre, x0_pre)).T.reshape(N**2,2)

M = np.apply_along_axis(slime.initial_masses, 1, x0,1) #particle masses
M = M/np.sum(M)
#%%
time_slice1 = test_data['y'][:,0]
time_slice2 = test_data['y'][:,15]
time_slice3 = test_data['y'][:,30]
time_slice4 = test_data['y'][:,45]

particles1 = time_slice1.reshape((15**2), 2, order = 'F')
particles2 = time_slice2.reshape((15**2), 2, order = 'F')
particles3 = time_slice3.reshape((15**2), 2, order = 'F')
particles4 = time_slice4.reshape((15**2), 2, order = 'F')

mesh = np.linspace(-2.1,2.1,200)

z1 = slime.blob(particles1, M, mesh, .06)
z2 = slime.blob(particles2, M, mesh, .06)
z3 = slime.blob(particles3, M, mesh, .06)
z4 = slime.blob(particles4, M, mesh, .06)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (9, 6))
sns.heatmap(z1, annot = False, fmt = "d", ax = ax1)
sns.heatmap(z2, annot = False, fmt = "d", ax = ax2)
sns.heatmap(z3, annot = False, fmt = "d", ax = ax3)
sns.heatmap(z4, annot = False, fmt = "d", ax = ax4)
fig

fig2 = plt.figure()
ax5 = fig2.add_subplot(111, projection = '3d')

x,y = np.meshgrid(mesh, mesh)
ax5.plot_surface(x,y,z1)
fig2

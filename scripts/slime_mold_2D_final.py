import scipy as sci
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib
from numpy.linalg import norm
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

### parameters/Initialization ###
m = 1 ##diffusion exponent
root_N = 10  ## sqrt of number of particles

 ##mollifier parameter - see carillio for choice
maximum_step_size = .1
T = 1 ##final time
y0 = np.array([1,0])
y1 = np.array([-1,0])


cell_start = -2.1 ##start of spacial grid
cell_end = 2.1 ##end of spacial grid
h = np.divide(cell_end-cell_start,root_N) #spacial step size
ep = np.power(h, .99)
particle_cells = np.linspace(cell_start,cell_end,root_N + 1)
cells_x, cells_y = np.meshgrid(particle_cells, particle_cells, indexing = 'ij') ##for plotting later
x0 = particle_cells[0:root_N] + np.divide(h,2) ##particle initial posistion coordinate components
initial_positions_vector = np.array(np.meshgrid(x0,x0)).T.reshape(root_N**2,2)  ##initial posisitons (center of each cell)

###reshape initial posisitons to be 1-D for solver###
initial_positions = np.squeeze(initial_positions_vector.reshape(2*(root_N**2),1,order = 'F'))
###mollifiers/interaction/drift kernals/blob ###

def mollifier(x, eps):
    output = np.divide(np.exp(np.divide(-(norm(x)**2),4*eps**2)), 4*np.pi*(eps**2))
    return output

def grad_mollifier(x,eps):
    output = -mollifier(x,eps) * np.divide(2, 4*(eps**2)) * x
    return output

def grad_W(x, kernal_choice):
    ##|x|^4/4 - |x| ##
    ##normy = norm(x)
    if kernal_choice == 1:
        output = norm(x)**2 - x
    if kernal_choice == 2:
        output = -10*grad_mollifier(x,np.sqrt(.3))
    if kernal_choice == 3:
        output = x
    return output

def V_prime(t,x,y0,y1):
    '''gradient term in a equation - "food smell"
        y0 and y1 are locations of food sources'''
    #output = -2*x*np.divide(1,4*t+.000001)*(mollifier(x - y1,np.sqrt(t+.000001)) + mollifier(x - y0,np.sqrt(t+.000001)))
    output = -2*(y1-x)*np.exp(-norm(y1-x)**2) + -2*(y0-x)*np.exp(-norm(y0-x)**2)
    #output = grad_mollifier(y0 - x, np.sqrt(2*(t+.000001))) + grad_mollifier(y1-x, np.sqrt(2*(t+.000001)))
    return output

def blob(particles, masses, mesh, ep):


    A = np.zeros((len(particles[:,0]),2,len(mesh),len(mesh)))
    x, y = np.meshgrid(mesh,mesh,indexing = 'ij')
    for i in range(len(x)):
        for j in range(len(y)):
            A[:,:,i,j] =  np.array([x[i,j], y[i,j]]) - particles

    output = np.zeros((len(mesh), len(mesh)))
    for i in range(len(x)):
        for j in range(len(y)):
            output[i,j] = np.dot(np.apply_along_axis(mollifier,1,A[:,:,i,j],ep), masses)

    return output


### particle masses ###

def initial_masses(x,i):
    if i == 1:  ##Gaussian
        tau = .0625 ##Gaussian parameter
        d = 1       ##dimension
        output = mollifier(x, np.sqrt(tau))
    if i ==2: ##two bumps
        tau1 = 0.0225
        tau2 = 0.0225
        output = 0.7*mollifier(x+.5,np.sqrt(tau1)) + 0.7*mollifier(x - .5, np.sqrt(tau2))

    return output

M = np.apply_along_axis(initial_masses, 1, initial_positions_vector, 1)
M = M / np.sum(M)


##### Right hand side - f(t,x) ##########
def x_prime(t,x, A, B, C, y0, y1,N, M, m, ep, kernal_choice):
    x = x.reshape((N**2),2,order = 'F') ##put the input into vector form


    diff = np.zeros((2, len(x[:,1]), len(x[:,1]))) ##tensor to store all the vector differences in
    for i in range(len(x[:,1])):
        diff[:, i, :] = (x[i, :] - x).T

    A_gradW = np.apply_along_axis(grad_W, 0, diff, kernal_choice)  ##apply functions to all elements of the difference tensor
    A_moll = np.apply_along_axis(mollifier, 0, diff, ep)
    A_grad_moll = np.apply_along_axis(grad_mollifier, 0, diff, ep)

    output = np.zeros((len(x[:,1]),2))  ##vector list to store changes in

    for i in range(len(x[:,1])):
        j = np.zeros(len(x[:,1]))  ##computed the nested sum in the interaction function
        for k in range(len(x[:,1])):
            j[i] = np.power(np.dot(A_moll[k, :], M), m-2)

        output[i, :] =  -np.sum(A_gradW[:,i,:] * M, axis = 1) - np.sum(A_grad_moll[:, i, :]*np.multiply(M,j), axis = 1) - np.power(np.dot(A_moll[i,:],M),m-2) * np.sum(A_grad_moll[:,i,:]*M, axis =1)-10*V_prime(t,x[i, :], y0, y1)

    output = np.squeeze(output.reshape(2*(N**2),1,order = 'F'))  #recast the output as a 1D array for the solver
    return output

##plotting
#### plotting ####


def plot_trajectories(sol, masses,N,title):
    t = sol['t']
    traj = sol['y']

    cmap = plt.get_cmap('viridis')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    cNorm = colors.Normalize(vmin = 0, vmax = max(masses))
    ScalarMap = cmx.ScalarMappable(norm = cNorm, cmap = cmap)
    for i in range(int(root_N**2)):
        linecolor = ScalarMap.to_rgba(masses[i])
        ax.plot(traj[i, :],traj[i+int(N**2), :],t, color = linecolor)
        #ax.scatter(initial_positions_vector[:,0], initial_positions_vector[:,1],0,s=10)
        ax.scatter([y0[0],y1[0]],[y0[1],y1[1]],0,s=10) #plot food sources
    ax2 = fig.add_axes([0.94,0.1,0.02,0.7])
    cb = matplotlib.colorbar.ColorbarBase(ax2, cmap='viridis', norm=cNorm)
    plt.savefig(title)

def plot_heatmap(sol, masses, N, time, gridpoints, cell_start, cell_end, ep, title):
    fig, ax = plt.subplots(figsize = (9,6))


def plot_profiles(sol, masses, N, time, gridpoints, cell_start, cell_end, ep, title):


    cmap = plt.get_cmap('viridis')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    cNorm = colors.Normalize(vmin = 0, vmax = max(masses))
    ScalarMap = cmx.ScalarMappable(norm = cNorm, cmap = cmap)

    mesh = np.linspace(cell_start, cell_end, gridpoints)

    timeslice = sol['y'][:,time]
    particles = timeslice.reshape((N**2),2,order = 'F')
    z = blob(particles, masses, mesh, ep)
    x, y = np.meshgrid(mesh, mesh)
    ax.plot_surface(x,y, z, cmap = cmap)

    ax2 = fig.add_axes([0.94,0.1,0.02,0.7])
    cb = matplotlib.colorbar.ColorbarBase(ax2, cmap='viridis', norm=cNorm)


    plt.savefig(title)
    plt.close()

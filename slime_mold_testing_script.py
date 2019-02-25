import scipy as sci
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib
from numpy.linalg import norm
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

### parameters/Initialization ###
m = 1 ##diffusion exponent
root_N = 15  ## sqrt of number of particles

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

def grad_W(x):
    ##|x|^4/4 - |x| ##
    ##normy = norm(x)
    output = x #-10*grad_mollifier(x,np.sqrt(.3))#x #(normy**2) * x
    return output

def V_prime(t,x):
    '''gradient term in a equation - "food smell"
        y0 and y1 are locations of food sources'''
    global y0
    global y1

    #output = -2*x*np.divide(1,4*t+.000001)*(mollifier(x - y1,np.sqrt(t+.000001)) + mollifier(x - y0,np.sqrt(t+.000001)))
    output = -2*(y1-x)*np.exp(-norm(y1-x)**2) + -2*(y0-x)*np.exp(-norm(y0-x)**2)
    #output = grad_mollifier(y0 - x, np.sqrt(2*(t+.000001))) + grad_mollifier(y1-x, np.sqrt(2*(t+.000001)))
    return output

def blob(particles, masses, mesh):
    global ep

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
def x_prime(t,x,pgrad,pkern,pdiff):
    global M
    global m
    global ep
    global root_N
    x = x.reshape((root_N**2),2,order = 'F') ##put the input into vector form


    A = np.zeros((2, len(x[:,1]), len(x[:,1]))) ##tensor to store all the vector differences in
    for i in range(len(x[:,1])):
        A[:, i, :] = (x[i, :] - x).T

    A_gradW = np.apply_along_axis(grad_W, 0, A)  ##apply functions to all elements of the difference tensor
    A_moll = np.apply_along_axis(mollifier, 0, A, ep)
    A_grad_moll = np.apply_along_axis(grad_mollifier,0,A, ep)

    output = np.zeros((len(x[:,1]),2))  ##vector list to store changes in

    for i in range(len(x[:,1])):
        j = np.zeros(len(x[:,1]))  ##computed the nested sum in the interaction function
        for k in range(len(x[:,1])):
            j[i] = np.power(np.dot(A_moll[k, :], M), m-2)

        output[i, :] =  -pkern*np.sum(A_gradW[:,i,:] * M, axis = 1) - pdiff*(np.sum(A_grad_moll[:, i, :]*np.multiply(M,j), axis = 1)) - pdiff*(np.power(np.dot(A_moll[i,:],M),m-2) * np.sum(A_grad_moll[:,i,:]*M, axis =1))-(pgrad*V_prime(t,x[i, :]))

    output = np.squeeze(output.reshape(2*(root_N**2),1,order = 'F'))  #recast the output as a 1D array for the solver
    return output

######## compute particle trajectories #############
def solve(method):
    global root_N
    global initial_positions
    global maximum_step_size
    global T
    sol = solve_ivp(x_prime, (0,T), initial_positions, method = method, max_step = maximum_step_size)
    return sol#, sol['t'], sol['y'].reshape((2,(root_N**2),len(sol['t'])),order = 'F'))

####parameter sweep####
def sweep(parameters):
    global root_N
    global initial_positions
    global maximum_step_size
    global T
    global M
    sol_dict = {}
    bad_list = []
    for set in parameters:
        try:
            sol = solve_ivp(lambda t,y:x_prime(t,y,set[0],set[1],set[2]),(0,T),initial_positions, method = 'BDF', max_step = maximum_step_size)
            sol_dict[set] = sol
        except:
            print("Error with solution parameters A = %d, B = %d, C = %d" % (set[0], set[1], set[2]) )
            bad_list.append(set)
        try:
            plot_trajectories(sol, M, "A%d_B%d_C%d" % (set[0], set[1], set[2]) )
            plot_profiles(sol, M, len(sol['t'])-1, 200, "A%d_B%d_C%d" % (set[0], set[1], set[2]))
        except:
            print("Error with plotting solution with parameters A = %d, B = %d, C = %d" % (set[0], set[1], set[2]) )

    return sol_dict, bad_list




#### plotting ####

def plot_trajectories(sol, masses, title):
    global root_N
    t = sol['t']
    traj = sol['y']

    cmap = plt.get_cmap('viridis')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    fig.suptitle(title)
    cNorm = colors.Normalize(vmin = 0, vmax = max(masses))
    ScalarMap = cmx.ScalarMappable(norm = cNorm, cmap = cmap)
    for i in range(int(root_N**2)):
        linecolor = ScalarMap.to_rgba(masses[i])
        ax.plot(traj[i, :],traj[i+int(root_N**2), :],t, color = linecolor)
        #ax.scatter(initial_positions_vector[:,0], initial_positions_vector[:,1],0,s=10)
        ax.scatter([y0[0],y1[0]],[y0[1],y1[1]],0,s=10) #plot food sources
    ax2 = fig.add_axes([0.94,0.1,0.02,0.7])
    cb = matplotlib.colorbar.ColorbarBase(ax2, cmap='viridis', norm=cNorm)
    plt.savefig(title+"_trajectories.pdf", bbox_inches = 'tight')
    #plt.show()

def plot_profiles(sol, masses, time, gridpoints, title):
    global cell_start
    global cell_end
    global root_N

    cmap = plt.get_cmap('viridis')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    fig.suptitle(title)
    cNorm = colors.Normalize(vmin = 0, vmax = max(masses))
    ScalarMap = cmx.ScalarMappable(norm = cNorm, cmap = cmap)

    mesh = np.linspace(cell_start, cell_end, gridpoints)

    timeslice = sol['y'][:,time]
    particles = timeslice.reshape((root_N**2),2,order = 'F')
    z = blob(particles, masses, mesh)
    x, y = np.meshgrid(mesh, mesh)
    ax.plot_surface(x,y, z, cmap = cmap)


    ax2 = fig.add_axes([0.94,0.1,0.02,0.7])
    cb = matplotlib.colorbar.ColorbarBase(ax2, cmap='viridis', norm=cNorm)
    plt.savefig(title+"_profile_time_%d.pdf" % time, bbox_inches = 'tight')


    #plt.show()

import scipy as sci
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from numpy.linalg import norm
from scipy.integrate import solve_ivp

### parameters/Initialization ###
m = 1 ##diffusion exponent
N = 100  ## Number of particles ##mollifier parameter - see carillio for choice
T = 2 ##final time
cell_start = -2.1 ##start of spacial grid
cell_end = 2.1 ##end of spacial grid
y0 = -1 ##food source locations
y1 = 1 ##food source locations
h = np.divide(cell_end-cell_start,N)#spacial step size
eps = np.power(h, .99)
particle_cells = np.linspace(cell_start,cell_end,N + 1)
x0 = particle_cells[0:N] + np.divide(h,2) ##particle initial posistions (middle of each cell)
maximum_step_size = .1


###masses###
def mollifier_tau(x, tau, d): #Gaussian w/ control over variance
    scaling = np.divide(1, np.power(4*np.pi*tau,np.divide(d,2)))
    exponential = np.exp(np.divide(-(norm(x)**2),4*tau))
    output = scaling *exponential
    return output

def mollifier_tau_prime(x, tau, d):
    output = np.divide(-2*x, 4*tau) * mollifier_tau(x,tau,d)
    return output

def initial_masses(x,i):
    if i == 1:  ##Gaussian
        tau = .0625 ##variance parameter
        d = 1       ##dimension
        output = mollifier_tau(x, tau, d)
    if i ==2: ##two bumps
        tau1 = 0.0225
        tau2 = 0.0225
        output = 0.7*mollifier_tau(x+.5,tau1,1) + 0.7*mollifier_tau(x - .5, tau2, 1)

    if i == 3: ### zero food stationary state
        beta = 1
        tau = np.power(np.divide(np.sqrt(np.pi)*beta,2),np.divide(2,3))
        output = beta*np.exp(-(np.power(x,2)*tau))
    if i == 4:
        output = -x**2 + 1.1**2

    return output

M = np.vectorize(initial_masses)(x0,1) * h
M = M/np.sum(M) ##normalized particle masses


##mollifiers/interaction/drift kernals/blob ###

def mollifier(x, eps):
    output = np.divide(np.exp(np.divide(-(x**2),4*eps**2)), np.power(4*np.pi*(eps**2),np.divide(1,2)))
    return output

def grad_mollifier(x, eps):
    output = -mollifier(x, eps) * np.divide(2, 4*(eps**2)) * x
    return output



def V_prime(t,x, y0, y1):
    '''gradient term in a equation - "food smell"
        y0 and y1 are locations of food sources'''

    output = -2*(y1-x)*np.exp(-(y1-x)**2) + -2*(y0-x)*np.exp(-(y0-x)**2)
    #output = mollifier_tau_prime(y1-x,t+.000001,1) + mollifier_tau_prime(y0-x,t+.000001,1)
    return output



def W_prime(x,kernal_choice):
    '''1-D derivative of W(x)'''
    if kernal_choice == 1:
        output =   x**3 - x  #-mollifier_tau_prime(x, .3, 1)   # x**3 - x
    if kernal_choice == 2:
        output = -mollifier_tau_prime(x, .3, 1)
    if kernal_choice == 3:
        output =  x
    return output

def blob(particles, grid, masses, eps):
    '''represents particle trajectories as blob profile'''

    A = np.zeros([len(grid),len(particles)])
    for i in range(len(grid)):
        A[i, :] = grid[i] - particles

    A = np.vectorize(mollifier)(A,eps)
    output = np.dot(A, masses)

    return output



##### f(t,y) ############

def x_prime(t,x,A, B, C,y0, y1, M, m, ep, kernal_choice):
    '''computes the righthand side of the blob method
        y0 - food source location 1, y1 - food source location 2, M - particle masses, m - diffusion exponent, ep - mollifier parameter, i - kernal choice'''

    diff = np.zeros([len(x), len(x)]) ###compute all the differences
    for i in range(len(x)):
        diff[i, :] = x[i] - x

    A_Wprime = np.vectorize(W_prime)(diff,kernal_choice)
    A_moll = np.vectorize(mollifier)(diff,ep)
    A_moll_prime = np.vectorize(grad_mollifier)(diff,ep)

    output = np.zeros(len(x))

    for i in range(len(output)):
        ###compute the nested term
        j = np.zeros(len(x))
        for k in range(len(x)):
            j[i] = np.power(np.dot(A_moll[k, :], M), m-2)

        output[i] =  -B*np.dot(A_Wprime[i, :], M) - C*np.dot(A_moll_prime[i,:], np.multiply(M, j)) - C*np.power(np.dot(A_moll[i,:],M),m-2)*np.dot(A_moll_prime[i,:],M) -A*(V_prime(t,x[i],y0,y1))

    return output

def solve(A,B,C,y0,y1,M,m,ep,kernel_choice, T,initial_positions, maximum_step_size):

    sol = solve_ivp(lambda t,y:x_prime(t,y,A,B,C,y0,y1,M,m,ep,kernel_choice), (0,T), initial_positions, method = 'BDF', max_step = maximum_step_size)

    return sol

#### plotting ####

def plot_trajectories(sol, masses):
    t = sol['t']
    traj = sol['y']

    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(figsize = (7,5))
    cNorm = colors.Normalize(vmin = 0, vmax = max(masses))
    ScalarMap = cmx.ScalarMappable(norm = cNorm, cmap = cmap)
    for i in range(len(masses)):
        linecolor = ScalarMap.to_rgba(masses[i])
        ax.plot(t, traj[i], color = linecolor)

    ax2 = fig.add_axes([0.94,0.1,0.02,0.7])
    cb = matplotlib.colorbar.ColorbarBase(ax2, cmap='viridis', norm=cNorm)
    plt.show()


def compute_and_plot_profiles(sol, masses, plot_times,  cell_start, cell_end,eps,y0,y1,gridpoints, title):
    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(figsize = (7,5))
    ax.set_title(title)
    ax.set_xlabel('position')
    ax.set_ylabel('density')
    cNorm = colors.Normalize(vmin = 0, vmax = max(plot_times))
    ScalarMap = cmx.ScalarMappable(norm = cNorm, cmap = cmap)
    grid = np.linspace(cell_start, cell_end, gridpoints)
    for i in plot_times:
        particles = sol['y'][:,i]
        profile = blob(particles, grid, masses, eps)
        linecolor = ScalarMap.to_rgba(i)
        ax.plot(grid, profile, color = linecolor)
    ax.plot([y0,y1],[0,0],'ro')

    ax2 = fig.add_axes([0.94,0.1,0.02,0.7])
    cb = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, norm=cNorm)
    plt.savefig(title)

def plot_both_things(sol,masses,plot_times, cell_start, cell_end, eps, y0, y1, gridpoints,title):
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_xlabel('time')
    ax1.set_ylabel('position')
    ax2.set_xlabel('position')
    ax2.set_ylabel('density')
    ax1.set_title('particle trajectories colored by mass')
    ax2.set_title('profiles colored by timestep')
    fig.suptitle(title)
    t = sol['t']
    traj = sol['y']

    cmap = plt.get_cmap('viridis')
    cNorm = colors.Normalize(vmin = 0, vmax = max(masses))
    ScalarMap1 = cmx.ScalarMappable(norm = cNorm, cmap = cmap)
    for i in range(len(masses)):
        linecolor = ScalarMap1.to_rgba(masses[i])
        ax1.plot(t, traj[i], color = linecolor)
    ax1.plot([0,0],[y0,y1], 'ro')
    ax3 = fig.add_axes([0.01,0.12,0.02,0.7])
    cb = matplotlib.colorbar.ColorbarBase(ax3, cmap=cmap, norm=cNorm)

    cNorm = colors.Normalize(vmin = 0, vmax = max(plot_times))
    ScalarMap2 = cmx.ScalarMappable(norm = cNorm, cmap = cmap)

    grid = np.linspace(np.min(np.array(sol['y']))-.1, np.max(np.array(sol['y']))+.1, gridpoints)
    for i in plot_times:
        particles = sol['y'][:,i]
        profile = blob(particles, grid, masses, eps)
        linecolor = ScalarMap2.to_rgba(i)
        ax2.plot(grid, profile, color = linecolor)
    ax2.plot([y0,y1],[0,0],'ro')
    ax4 = fig.add_axes([0.94,0.15,0.02,0.7])
    cb = matplotlib.colorbar.ColorbarBase(ax4, cmap=cmap, norm=cNorm)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(title + '.pdf')

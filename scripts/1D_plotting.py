import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pickle
import scipy
from slime_mold_1D_final import blob
#%%

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

def plot_both_things(sol,masses,plot_times, cell_start, cell_end, eps, y0, y1, gridpoints,title, save_location, T):
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

    cmap = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin = 0, vmax = max(masses))
    ScalarMap1 = cmx.ScalarMappable(norm = cNorm, cmap = cmap)
    for i in range(len(masses)):
        linecolor = ScalarMap1.to_rgba(masses[i])
        ax1.plot(t, traj[i], color = linecolor)
    ax1.plot([0,0],[y0,y1], 'ro')
    ax3 = fig.add_axes([0.01,0.12,0.02,0.7])
    cb = matplotlib.colorbar.ColorbarBase(ax3, cmap=cmap, norm=cNorm)

    cNorm = colors.Normalize(vmin = 0, vmax = T)
    ScalarMap2 = cmx.ScalarMappable(norm = cNorm, cmap = cmap)

    grid = np.linspace(np.min(np.array(sol['y']))-.1, np.max(np.array(sol['y']))+.1, gridpoints)
    for i in plot_times:
        particles = sol['y'][:,i]
        profile = blob(particles, grid, masses, eps)
        true_time = (T / (max(plot_times) -1)) * i
        linecolor = ScalarMap2.to_rgba(true_time)
        ax2.plot(grid, profile, color = linecolor)
    ax2.plot([y0,y1],[0,0],'ro')
    ax4 = fig.add_axes([0.94,0.15,0.02,0.7])
    cb = matplotlib.colorbar.ColorbarBase(ax4, cmap=cmap, norm=cNorm)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    fig.set_size_inches((10, 5), forward=False)
    fig.savefig(save_location + '.png', dpi=500)

    return fig


#%%
def compute_even_plot_times(time_range_length, number_of_plots):
    step = time_range_length // number_of_plots
    plot_times = [i *step for i in range(number_of_plots)]
    plot_times.append(time_range_length -1)

    return plot_times



#%%
def plot_four_profiles(x_label, y_label, sol_dict_path, param_dict_path, plot_number, title, moll_eps, param_set = "drift", custom_params = None, save = False, save_path = None):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, sharex = True, sharey = True)

    #ax1.set_xlabel('position')
    ax1.set_ylabel(y_label)
    #ax2.set_xlabel('position')
    #ax2.set_ylabel('density')
    ax3.set_xlabel(x_label)
    ax3.set_ylabel(y_label)
    ax4.set_xlabel(x_label)
    #ax4.set_ylabel('density')
    fig.suptitle(title)

    ax4_box = fig.axes[3].get_position()
    ax2_box = fig.axes[1].get_position()
    ax4_box.x1 = ax4_box.x1 - .04
    ax4_box.x0 = ax4_box.x0 - .04
    ax2_box.x1 = ax2_box.x1 - .04
    ax2_box.x0 = ax2_box.x0 - .04

    fig.axes[3].set_position(ax4_box)
    fig.axes[1].set_position(ax2_box)

    if param_set == 'drift':
        param_1 = (10,5,1)
        param_2 = (10,1,5)
        param_3 = (10,1,1)
        param_4 = (10,5,5)

    elif param_set == 'interaction':
        param_1 = (5,10,1)
        param_2 = (1,10,5)
        param_3 = (1,10,1)
        param_4 = (5,10,5)

    else:
        param_1 = custom_params[0]
        param_2 = custom_params[1]
        param_3 = custom_params[2]
        param_4 = custom_params[3]


    ax1.set_title('$A= {}$, $B ={} $, $C = {}$'.format(param_1[0], param_1[1], param_1[2]), pad = 3)
    ax2.set_title('$A= {}$, $B ={} $, $C = {}$'.format(param_2[0], param_2[1], param_2[2]), pad = 3)
    ax3.set_title('$A= {}$, $B ={} $, $C = {}$'.format(param_3[0], param_3[1], param_3[2]), pad = 3)
    ax4.set_title('$A= {}$, $B ={} $, $C = {}$'.format(param_4[0], param_4[1], param_4[2]), pad = 3)

    sol_dict = pickle.load(open(sol_dict_path, "rb"))
    param_dict = pickle.load(open(param_dict_path, "rb"))
    sol_dict.keys()

    N = param_dict['N']
    masses = param_dict['M']
    eps = param_dict['eps']
    y0 = param_dict['y0']
    y1 = param_dict['y1']
    T = param_dict['T']

    sols = [sol_dict[param_1], sol_dict[param_2], sol_dict[param_3], sol_dict[param_4]]
    plot_times = [compute_even_plot_times(len(sol['t']), plot_number) for sol in sols]
    axess = [ax1, ax2, ax3, ax4]

    cmap = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin = 0, vmax = T)
    ScalarMap = cmx.ScalarMappable(norm = cNorm, cmap = cmap)

    colorbar = fig.add_axes([0.90,0.15,0.02,0.7])
    colorbar.set_title('time')
    cb = matplotlib.colorbar.ColorbarBase(colorbar, cmap = cmap, norm = cNorm)


    for i in range(4):
        ax = axess[i]
        plot_time = plot_times[i]
        sol = sols[i]
        grid = np.linspace(-2.1, 2.1, 100)

        for i in plot_time:
            particles = sol['y'][:,i]
            profile = blob(particles, grid, masses, moll_eps)
            true_time = (T / (max(plot_time) -1)) * i
            linecolor = ScalarMap.to_rgba(true_time)
            ax.plot(grid, profile, color = linecolor)
            ax.plot([y0,y1],[0,0],'ro')


    return fig


def plot_two_profiles(x_label, y_label, sol_dict_path, param_dict_path, plot_number, title, params = None):
    fig, (ax1, ax2) = plt.subplots(1,2, sharex = True, sharey = True)

    #ax1.set_xlabel('position')
    ax1.set_ylabel(y_label)
    #ax2.set_xlabel('position')
    #ax2.set_ylabel('density')
    ax1.set_xlabel(x_label)
    ax2.set_xlabel(x_label)
    #ax4.set_ylabel('density')
    fig.suptitle(title)


    ax2_box = fig.axes[1].get_position()
    ax2_box.x1 = ax2_box.x1 - .04
    ax2_box.x0 = ax2_box.x0 - .04

    fig.axes[1].set_position(ax2_box)

    param_1 = params[0]
    param_2 = params[1]



    ax1.set_title('$A= {}$, $B ={} $, $C = {}$'.format(param_1[0], param_1[1], param_1[2]), pad = 3)
    ax2.set_title('$A= {}$, $B ={} $, $C = {}$'.format(param_2[0], param_2[1], param_2[2]), pad = 3)

    sol_dict = pickle.load(open(sol_dict_path, "rb"))
    param_dict = pickle.load(open(param_dict_path, "rb"))
    sol_dict.keys()

    N = param_dict['N']
    masses = param_dict['M']
    eps = param_dict['eps']
    y0 = param_dict['y0']
    y1 = param_dict['y1']
    T = param_dict['T']

    sols = [sol_dict[param_1], sol_dict[param_2]]
    plot_times = [compute_even_plot_times(len(sol['t']), plot_number) for sol in sols]
    axess = [ax1, ax2]

    cmap = plt.get_cmap('viridis')
    cNorm = colors.Normalize(vmin = 0, vmax = T)
    ScalarMap = cmx.ScalarMappable(norm = cNorm, cmap = cmap)

    colorbar = fig.add_axes([0.90,0.15,0.02,0.7])
    colorbar.set_title('time')
    cb = matplotlib.colorbar.ColorbarBase(colorbar, cmap = cmap, norm = cNorm)


    for i in range(2):
        ax = axess[i]
        plot_time = plot_times[i]
        sol = sols[i]
        grid = np.linspace(-2.1, 2.1, 75)

        for i in plot_time:
            particles = sol['y'][:,i]
            profile = blob(particles, grid, masses, eps)
            true_time = (T / (max(plot_time) -1)) * i
            linecolor = ScalarMap.to_rgba(true_time)
            ax.plot(grid, profile, color = linecolor)
            ax.plot([y0,y1],[0,0],'ro')


    return fig


def plot_six_profiles(x_label, y_label, sol_dict_path, param_dict_path, plot_number, title, moll_eps, params = None):
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, sharex = True, sharey = True)

    #ax1.set_xlabel('position')
    ax1.set_ylabel(y_label)
    #ax2.set_xlabel('position')
    #ax2.set_ylabel('density')
    ax3.set_ylabel(y_label)
    ax5.set_ylabel(y_label)
    ax5.set_xlabel(x_label)
    ax6.set_xlabel(x_label)
    #ax4.set_ylabel('density')
    fig.suptitle(title)

    ax4_box = fig.axes[3].get_position()
    ax2_box = fig.axes[1].get_position()
    ax6_box = fig.axes[5].get_position()
    ax4_box.x1 = ax4_box.x1 - .04
    ax4_box.x0 = ax4_box.x0 - .04
    ax2_box.x1 = ax2_box.x1 - .04
    ax2_box.x0 = ax2_box.x0 - .04
    ax6_box.x1 = ax6_box.x1 - .04
    ax6_box.x0 = ax6_box.x0 - .04

    fig.axes[3].set_position(ax4_box)
    fig.axes[1].set_position(ax2_box)
    fig.axes[5].set_position(ax6_box)

    fig.set_figheight(fig.get_figheight() + fig.get_figheight() * .3)

    param_1 = params[0]
    param_2 = params[1]
    param_3 = params[2]
    param_4 = params[3]
    param_5 = params[4]
    param_6 = params[5]


    ax1.set_title('$A= {}$, $B ={} $, $C = {}$'.format(param_1[0], param_1[1], param_1[2]), pad = 3)
    ax2.set_title('$A= {}$, $B ={} $, $C = {}$'.format(param_2[0], param_2[1], param_2[2]), pad = 3)
    ax3.set_title('$A= {}$, $B ={} $, $C = {}$'.format(param_3[0], param_3[1], param_3[2]), pad = 3)
    ax4.set_title('$A= {}$, $B ={} $, $C = {}$'.format(param_4[0], param_4[1], param_4[2]), pad = 3)
    ax5.set_title('$A= {}$, $B ={} $, $C = {}$'.format(param_5[0], param_5[1], param_5[2]), pad = 3)
    ax6.set_title('$A= {}$, $B ={} $, $C = {}$'.format(param_6[0], param_6[1], param_6[2]), pad = 3)

    sol_dict = pickle.load(open(sol_dict_path, "rb"))
    param_dict = pickle.load(open(param_dict_path, "rb"))
    sol_dict.keys()

    N = param_dict['N']
    masses = param_dict['M']
    eps = param_dict['eps']
    y0 = param_dict['y0']
    y1 = param_dict['y1']
    T = param_dict['T']

    sols = [sol_dict[param_1], sol_dict[param_2], sol_dict[param_3], sol_dict[param_4], sol_dict[param_5], sol_dict[param_6]]
    plot_times = [compute_even_plot_times(len(sol['t']), plot_number) for sol in sols]
    axess = [ax1, ax2, ax3, ax4, ax5, ax6]

    cmap = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin = 0, vmax = T)
    ScalarMap = cmx.ScalarMappable(norm = cNorm, cmap = cmap)

    colorbar = fig.add_axes([0.90,0.15,0.02,0.7])
    colorbar.set_title('time')
    cb = matplotlib.colorbar.ColorbarBase(colorbar, cmap = cmap, norm = cNorm)


    for i in range(6):
        ax = axess[i]
        plot_time = plot_times[i]
        sol = sols[i]
        grid = np.linspace(-2.1, 2.1, 100)

        for i in plot_time:
            particles = sol['y'][:,i]
            profile = blob(particles, grid, masses, moll_eps)
            true_time = (T / (max(plot_time) -1)) * i
            linecolor = ScalarMap.to_rgba(true_time)
            ax.plot(grid, profile, color = linecolor)
            ax.plot([y0,y1],[0,0],'ro')


    return fig

# %%
gaussian_solutions = pickle.load(open('../results/2020_sweep_gaussian_kernel/2020_sweep_gaussian_kernel_solution_data.pkl', 'rb'))
gaussian_params = pickle.load(open('../results/2020_sweep_gaussian_kernel/2020_sweep_gaussian_kernel_param_dict.pkl', 'rb'))
sol = gaussian_solutions[(10,10,1)]
gaussian_params['eps']
output = compute_and_plot_profiles(sol, gaussian_params['M'], compute_even_plot_times(len(sol['t']), 5), gaussian_params['cell_start'], gaussian_params['cell_end'], .04, gaussian_params['y0'], gaussian_params['y1'], 100,'test')

fair_solution = pickle.load(open('../results/2020_sweep_300_fair_quadratic_kernel/2020_sweep_300_fair_quadratic_kernel_solution_data.pkl', 'rb'))
fair_solution_params = pickle.load(open('../results/2020_sweep_300_fair_quadratic_kernel/2020_sweep_300_fair_quadratic_kernel_param_dict.pkl', 'rb'))
sol = fair_solution[(1,1,1)]
output = plot_both_things(sol, fair_solution_params['M'], compute_even_plot_times(len(sol['t']), 5), fair_solution_params['cell_start'], fair_solution_params['cell_end'], .02, fair_solution_params['y0'], fair_solution_params['y1'], 100, 'Quadratic kernel - fair regime', '../../tex/drafts/figures/quadratic_fair_300_2020', fair_solution_params['T'])
output
#%%
zerofood_solution = pickle.load(open('../results/2020_sweep_200_0food_quadratic_kernel/2020_sweep_200_0food_polynomial_kernel_solution_data.pkl', 'rb'))
zerofood_solution.keys()
zerofood_params = pickle.load(open('../results/2020_sweep_200_0food_quadratic_kernel/2020_sweep_200_0food_polynomial_kernel_param_dict.pkl', 'rb'))
sol = zerofood_solution[(0,1,1)]
output = plot_both_things(sol, zerofood_params['M'], compute_even_plot_times(len(sol['t']), 5), zerofood_params['cell_start'], zerofood_params['cell_end'], .02, zerofood_params['y0'], zerofood_params['y1'], 100, 'Quadratic kernel - no food source', '../../tex/drafts/figures/quadratic_0food_200', zerofood_params['T'])
output
# %%
gaussian_drift = plot_four_profiles('position', 'density', '../results/2020_sweep_gaussian_kernel/2020_sweep_gaussian_kernel_solution_data.pkl', '../results/2020_sweep_gaussian_kernel/2020_sweep_gaussian_kernel_param_dict.pkl', 5, "Drift dominated regime - gaussian kernel", .04, 'drift')
gaussian_drift
gaussian_drift.savefig('../../tex/drafts/figures/gaussian_drift_dominated_2020_200.pdf', format = 'pdf')

# %%
gaussian_interaction = plot_four_profiles('position', 'density', '../results/2020_sweep_gaussian_kernel/2020_sweep_gaussian_kernel_solution_data.pkl', '../results/2020_sweep_gaussian_kernel/2020_sweep_gaussian_kernel_param_dict.pkl', 5, "Interaction dominated regime - gaussian kernel", .04, 'interaction')
gaussian_interaction.savefig('../../tex/drafts/figures/gaussian_interaction_dominated_2020_200.pdf', format = 'pdf')

# %%
gaussian_competition = plot_two_profiles('position', 'density', '../results/2020_sweep_gaussian_kernel/2020_sweep_gaussian_kernel_solution_data.pkl', '../results/2020_sweep_gaussian_kernel/2020_sweep_gaussian_kernel_param_dict.pkl', 5, "Competition regime - gaussian kernel", params = [(10,10,1), (10,10,5)])
gaussian_competition.savefig('../../tex/drafts/figures/gaussian_competition_2020_200.pdf', format = 'pdf')
# %%
gaussian_fullcompetition = plot_six_profiles('position', 'density', '../results/2020_sweep_gaussian_kernel/2020_sweep_gaussian_kernel_solution_data.pkl', '../results/2020_sweep_gaussian_kernel/2020_sweep_gaussian_kernel_param_dict.pkl', 5, "Competition regime - gaussian kernel", .04, params = [(10,10,1), (10,10,5), (10,1,10), (10,5,10),(1,10,10), (5,10,10)])
gaussian_fullcompetition.savefig('../../tex/drafts/figures/gaussian_fullcompetition_2020_200.pdf', format = 'pdf')

# %%
quad_drift = plot_four_profiles('position', 'density', '../results/2020_sweep_quadratic_kernel/2020_sweep_quadratic_kernel_solution_data.pkl', '../results/2020_sweep_quadratic_kernel/2020_sweep_quadratic_kernel_param_dict.pkl', 5, "Drift dominated regime - quadratic kernel", .04, 'drift')
quad_drift.savefig('../../tex/drafts/figures/quadratic_drift_dominated_2020_200.pdf', format = 'pdf')

# %%
quad_interaction = plot_four_profiles('position', 'density', '../results/2020_sweep_quadratic_kernel/2020_sweep_quadratic_kernel_solution_data.pkl', '../results/2020_sweep_quadratic_kernel/2020_sweep_quadratic_kernel_param_dict.pkl', 5, "Interaction dominated regime - quadratic kernel",.03, 'interaction')
quad_interaction.savefig('../../tex/drafts/figures/quadratic_interaction_dominated_2020_200.pdf', format = 'pdf')
# %%
quad_competition = plot_two_profiles('position', 'density', '../results/2020_sweep_quadratic_kernel/2020_sweep_quadratic_kernel_solution_data.pkl', '../results/2020_sweep_quadratic_kernel/2020_sweep_quadratic_kernel_param_dict.pkl', 5, "Competition regime - quadratic kernel", params = [(10,10,1), (10,10,5)])
quad_competition.savefig('../../tex/drafts/figures/quadratic_competition_2020_200.pdf', format = 'pdf')

# %%
quad_fullcompetition = plot_six_profiles('position', 'density', '../results/2020_sweep_quadratic_kernel/2020_sweep_quadratic_kernel_solution_data.pkl', '../results/2020_sweep_quadratic_kernel/2020_sweep_quadratic_kernel_param_dict.pkl', 5, "Competition regime - quadratic kernel",.04, params = [(10,10,1), (10,10,5), (10,1,10), (10,5,10),(1,10,10), (5,10,10)])
quad_fullcompetition.savefig('../../tex/drafts/figures/quadratic_fullcompetition_2020_200.pdf', format = 'pdf')

# %%
poly_drift = plot_four_profiles('position', 'density', '../results/2020_sweep_polynomial_kernel/2020_sweep_polynomial_kernel_solution_data.pkl', '../results/2020_sweep_polynomial_kernel/2020_sweep_polynomial_kernel_param_dict.pkl', 5, "Drift dominated regime - polynomial kernel", .04,'drift')
poly_drift.savefig('../../tex/drafts/figures/polynomial_drift_dominated_2020_200.pdf', format = 'pdf')

# %%
poly_interaction = plot_four_profiles('position', 'density', '../results/2020_sweep_polynomial_kernel/2020_sweep_polynomial_kernel_solution_data.pkl', '../results/2020_sweep_polynomial_kernel/2020_sweep_polynomial_kernel_param_dict.pkl', 5, "Interaction dominated regime - polynomial kernel", .04, 'interaction')
poly_interaction.savefig('../../tex/drafts/figures/polynomial_interaction_dominated_2020_200.pdf', format = 'pdf')

# %%
poly_competition = plot_two_profiles('position', 'density', '../results/2020_sweep_polynomial_kernel/2020_sweep_polynomial_kernel_solution_data.pkl', '../results/2020_sweep_polynomial_kernel/2020_sweep_polynomial_kernel_param_dict.pkl', 5, "Competition regime - polynomial kernel", params = [(10,10,1), (10,10,5)])
poly_competition.savefig('../../tex/drafts/figures/2020_sweep_polynomial_kernel_param_dict_competition_2020_200.pdf', format = 'pdf')

#%%
poly_fullcompetition = plot_six_profiles('position', 'density', '../results/2020_sweep_polynomial_kernel/2020_sweep_polynomial_kernel_solution_data.pkl', '../results/2020_sweep_polynomial_kernel/2020_sweep_polynomial_kernel_param_dict.pkl', 5, "Competition regime - polynomial kernel", .04, params = [(10,10,1), (10,10,5), (10,1,10), (10,5,10),(1,10,10), (5,10,10)])
poly_fullcompetition.savefig('../../tex/drafts/figures/polynomial_fullcompetition_2020_200.pdf', format = 'pdf')

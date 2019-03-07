include("slime_mold_stochastic.jl")
using Plots
using JLD2
##does a parameter sweep over a set of paramter values for A,B and C
###particle/histogram parameters
runs = 5
N = 5
bin_start = -2.1
bin_end = 2.1
bin_number = 100
##initial_condition_parameters
sigma =.25
mu =0
##ODE parameters
T =3
dt =.01
kernal_choice =1
kernal_string ="quadratic_test"
y1 =-1
y0 =1

##parameters to be looped over/times to plot at
parameter_list = [(1,5,10),(5,1,10),(5,5,10),(10,1,1),(10,1,5),(10,5,1),(1,10,1),(1,10,5),(5,10,1),(1,1,1),(10,1,10),(1,10,10), (10,10,1)]
t_list = [0.1,1,2,3]

##place to save all outputs
particle_outputs = []

for i in 1:length(parameter_list)
    parameters = parameter_list[i]
    A = parameters[1]
    B = parameters[2]
    C = parameters[3]
    filename = "$(kernal_string)_A$(A)_B$(B)_C$(C)"

    averager_output =  Slime.averager(runs, N, bin_start, bin_end, bin_number, t_list, sigma, mu, T, dt, kernal_choice, A,B,C,y1,y0) ###run "runs" particle simulations with N particles and compute histograms
    push!(particle_outputs, averager_output[1]) #store the output data

    sample_trajectories = averager_output[5]  ##plot the particle trajectories from one of the runs
    plot(sample_trajectories[2], transpose(sample_trajectories[1]), color = "red", legend = false, title = "sample_trajectories_$(filename)")
    savefig("sample_trajectories_$(filename)")

    title!(averager_output[6], "averaged_profiles_$(filename)")
    savefig(averager_output[6], "averaged_profiles_$(filename)")
end

@save "$(kernal_string)_output_dict.jld" particle_outputs ##save all the trajectories from each run load using @load filename particle_outputs

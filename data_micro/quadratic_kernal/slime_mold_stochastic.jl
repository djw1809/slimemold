module Slime
using Plots
using Distributions
using StatsBase

###testing variables
min = -3.0
max =3.0
N =10
init = rand(N) * (max - min) .+ min

####functions
function gaussian(x, mean, var)
    output = (1/sqrt(4*pi*var))*exp(-(x-mean)^2/(4*var))
    return output
end

function gaussian_prime(x, mean, var)
    output = (-2*(x-mean)/4*var)*((1/sqrt(4*pi*var))*exp(-(x-mean)^2/(4*var)))
    return output
end

function V_prime(x, y1,y0)
    output = 2*(y1-x)*exp(-(y1-x)^2) + 2*(y0-x)*exp(-(y0-x)^2)
    return output
end

function W_prime(x, kernal_choice)

    if kernal_choice == 1 #quadratic kernal
        output = -x

    elseif kernal_choice == 2 #polynomial kernal
        output = x - x^3

    elseif kernal_choice == 3
        output = gaussian_prime(x,0,.3) #gaussian kernal

    end
    return output

end

###right hand side
"""compute the right hand side of the SDE, return the change in state given the current state
    x, the choice of kernal W, parameters and food source locations"""
function rhs_deterministic(x, kernal_choice, A,B,C,y1,y0)
    N = length(x)
    differences = zeros(length(x), length(x))
    d = Normal()

    for i in 1:length(x)
            differences[i,:] = x[i] .- x
    end

    differences_Wprime = W_prime.(differences, kernal_choice)

    drift = A*V_prime.(x, y1,y0) ##drift term of the SDE grad_V(X_i(t))
    kernal = (B/(N-1))*(differences_Wprime * ones(length(x))) ##compute the kernal term
    diffusion = (2*sqrt(C))*rand(d, length(x)) ##iid values drawn from normal distribution

    return (drift, kernal, diffusion)
end


###solver implementing Euler-Maruyama
"""solve the stochastic particle equation given the kernal choice, parameters(A,B,C), initial condition (x0), final time(T), timestep (dt), and food source locations (y0,y1), uses euler-maruyama"""
function solver(x0, T, dt, kernal_choice, A,B,C,y1,y0)
    steps = Int(round(T/dt))
    X = zeros(length(x0), steps)
    X[:, 1] = x0
    t = zeros(steps)

    for i in 1:steps-1
        change = rhs_deterministic(X[:, i], kernal_choice, A,B,C,y1,y0)
        X[:, i+1] = X[:,i] + dt*(change[1] + change[2]) + sqrt(dt)*(change[3])
        t[i+1] = t[i] + dt
    end

    return (X,t)
end

"""run the stochastic simulation "run" times with N particles with initial posistions drawn randomly from a gaussian (mu, sigma) each run.  Compute and plot normalized histograms on bins given by bin_start, bin_end and bin_number after all runs of particle positions at times given by t_list"""
function averager(runs, N, bin_start, bin_end, bin_number, t_list, sigma, mu,T,dt,kernal_choice,A,B,C,y1,y0)
    d = Normal(mu, sigma)
    outputs = []
    histogram_values = zeros(length(t_list), bin_number, runs)
    sample_output = 0 ##variable to store a sample of trajectories in

    ##compute histograms for many realizations of particle model with initial posistions drawn from d
    for i in 1:runs
        x0 = rand(d, N)
        output = solver(x0, T, dt, kernal_choice, A, B, C, y1, y0)
        if i == 1
            sample_output = (output[1], output[2])
        end
        push!(outputs, output)
        for j in 1:length(t_list)
            output_step = Int(round(t_list[j]/dt))
            histogram = fit(Histogram, output[1][:, output_step], range(bin_start, bin_end, length = bin_number +1))
            histogram_values[j, : ,i] = histogram.weights
        end
    end

    ##average the histograms from all the runs to get the final probability distribution and plot it for each time
    x = Array(range(bin_start, bin_end, length = bin_number ))
    step = x[2] - x[1]
    x = x .+ step
    histo_sum = zeros(length(t_list), bin_number)
    labels = []
    for k in 1:length(t_list)
        histo_sum[k,:] = sum(histogram_values[k, :, :], dims = 2)./(N*runs)
        push!(labels, "t = $(t_list[k])")
    end

    colors = ["red" "blue" "green" "orange"]

    f = plot(x, transpose(histo_sum), label = labels, color = colors)
    return (outputs, histogram_values, x, histo_sum, sample_output, f)
end

"""just computes/plots histogram of final time after running stochastic simulation 'runs' times with N particles.""" 
function averager_end(runs, N, bin_start, bin_end, bin_number, sigma, mu,T,dt,kernal_choice,A,B,C,y1,y0)
    d = Normal()
    histogram_values = zeros(bin_number, runs)
    histograms = []
    outputs = []

    for i in 1:runs
        x0 = rand(d, N)
        output = solver(x0, T, dt, kernal_choice, A, B, C, y1, y0)
        push!(outputs,output)
        histogram = fit(Histogram, output[1][:, length(output[2])], range(bin_start, bin_end, length = bin_number + 1))
        histogram_values[:, i] = histogram.weights
        push!(histograms, histogram)
    end

    histo_sum = sum(histogram_values, dims = 2)
    x = Array(range(bin_start, bin_end, length = bin_number + 1))[1:bin_number]
    x = x .+ (x[2] - x[1])
    display(plot(x, histo_sum))

    return (histo_sum, histogram_values, histograms, outputs, x)
end











###compute the histogram of a number of trials
































end  ##end module

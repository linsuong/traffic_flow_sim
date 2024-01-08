import numpy as np
import matplotlib.pyplot as plt
import random
import os
from base import Simulation
from base import Vehicle
from base import Obstacle
from base import Road

save_location = r'C:\Users\linus\Documents\traffic simulation plots\final plots'
#Density simulation#
seeds = 100
random.seed(seeds)
for j in range(2, 5, 1): #number of lanes
    fig, ax = plt.subplots(1, 1, figsize = (20, 10))
    for i in range(2, 14, 2): #velocities
        Sim = Simulation()
        Sim.plot_density(steps = 1000, length = 1000, max_velocity= i, slow_prob= 0.2, 
                        number_of_lanes= j,  labels = i, ax = ax)
    fig.suptitle(f'Fundamental relationship, max velocities, {j} lanes')
    ax.autoscale()
    ax.legend(loc = 'upper right')
    plt.savefig(os.path.join(save_location, f'Density Iteration dots, velocity, {j} lanes'))

#with obstacle
seeds = 100
random.seed(seeds)
for j in range(1, 5, 1): #number of lanes
    fig, ax = plt.subplots(1, 1, figsize = (20, 10))
    for i in range(2, 14, 2): #velocities
        Sim = Simulation()
        Sim.plot_density(steps = 1000, length = 1000, max_velocity= i, slow_prob= 0.2, number_of_lanes= j, 
                         obstacle= True, labels = i, ax = ax)
    fig.suptitle(f'Fundamental relationship with obstacle, max velocities, {j} lanes')
    ax.autoscale()
    ax.legend(loc = 'upper right')
    plt.savefig(os.path.join(save_location, f'Density Iteration dots with obstacle, velocity, {j} lanes'))

#Density-Slow Probability Iteration#
slowprobs = np.arange(0.2, 1.2, 0.2)
seeds = 100
random.seed(seeds)
for j in range(1, 5, 1): #number of lanes
    fig, ax = plt.subplots(1, 1, figsize = (20, 10))
    for i in slowprobs: #slow probabilities
        Sim = Simulation()
        Sim.plot_density(steps = 1000, length = 1000, max_velocity= 6, slow_prob= i, number_of_lanes= j, 
                          labels = i, ax = ax)
    fig.suptitle(f'Fundamental relationship, slow probabilities, {j} lanes')
    ax.autoscale()
    ax.legend(loc = 'upper right')
    plt.savefig(os.path.join(save_location, f'Density Iteration dots, slow probability, {j} lanes'))
    
#with obstacle
seeds = 100
random.seed(seeds)
for j in range(1, 5, 1): #number of lanes
    fig, ax = plt.subplots(1, 1, figsize = (20, 10))
    for i in slowprobs: #slow probabilities
        Sim = Simulation()
        Sim.plot_density(steps = 1000, length = 1000, max_velocity= 6, slow_prob= i, number_of_lanes= j, 
                         obstacle= True, labels = i, ax = ax)
    fig.suptitle(f'Fundamental relationship with obstacle, slow probabilities, {j} lanes')
    ax.autoscale()
    ax.legend(loc = 'upper right')
    plt.savefig(os.path.join(save_location, f'Density Iteration dots with obstacle, slow probability, {j} lanes'))

seeds = 100
random.seed(seeds)
for i in range(1, 5, 1):
    fig, ax = plt.subplots(figsize = (20, 10))
    Sim = Simulation()
    Sim.plot_density(steps = 1000, length = 1000, max_velocity= 6, slow_prob= 0.2, number_of_lanes= j, 
                    labels = i, ax = ax)
    fig.suptitle(f'Fundamental relationship, comparing number of lanes')
    ax.autoscale()
    ax.legend(loc = 'upper right')
    plt.savefig(os.path.join(save_location, f'Density Iteration dots number of lanes'))

seeds = 100
random.seed(seeds)
for i in range(1, 5, 1):
    fig, ax = plt.subplots(figsize = (20, 10))
    Sim = Simulation()
    Sim.plot_density(steps = 1000, length = 1000, max_velocity= 6, slow_prob= 0.2, number_of_lanes= j, 
                    labels = i, ax = ax)
    fig.suptitle(f'Fundamental relationship, comparing number of lanes with obstacle')
    ax.autoscale()
    ax.legend(loc = 'upper right')
    plt.savefig(os.path.join(save_location, f'Density Iteration dots with obstacle number of lanes'))


##investigating obstacle at differnet lane
#with obstacle
seeds = 100
random.seed(seeds)
for j in range(3, 5, 1): #number of lanes
    fig, ax = plt.subplots(1, 1, figsize = (20, 10))
    for i in range(2, 14, 2): #velocities
        Sim = Simulation()
        Sim.plot_density(steps = 1000, length = 1000, max_velocity= i, slow_prob= 0.2, number_of_lanes= j, 
                         obstacle= True, labels = i, ax = ax)
    fig.suptitle(f'Fundamental relationship with obstacle in lane 2, max velocities, {j} lanes')
    ax.autoscale()
    ax.legend(loc = 'upper right')
    plt.savefig(os.path.join(save_location, f'Density Iteration dots with obstacle in lane 2, velocity, {j} lanes'))

seeds = 100
random.seed(seeds)
for j in range(3, 5, 1): #number of lanes
    fig, ax = plt.subplots(1, 1, figsize = (20, 10))
    for i in slowprobs: #slow probabilities
        Sim = Simulation()
        Sim.plot_density(steps = 1000, length = 1000, max_velocity= 6, slow_prob= i, number_of_lanes= j, 
                         obstacle= True, labels = i, ax = ax)
    fig.suptitle(f'Fundamental relationship with obstacle in lane 2, slow probabilities, {j} lanes')
    ax.autoscale()
    ax.legend(loc = 'upper right')
    plt.savefig(os.path.join(save_location, f'Density Iteration dots with obstacle in lane 2, slow probability, {j} lanes'))

seeds = 100
random.seed(seeds)
for i in range(3, 5, 1):
    fig, ax = plt.subplots(figsize = (20, 10))
    Sim = Simulation()
    Sim.plot_density(steps = 1000, length = 1000, max_velocity= 6, slow_prob= 0.2, number_of_lanes= j, 
                    labels = i, ax = ax)
    fig.suptitle(f'Fundamental relationship, comparing number of lanes with obstacle in lane 2')
    ax.autoscale()
    ax.legend(loc = 'upper right')
    plt.savefig(os.path.join(save_location, f'Density Iteration dots with obstacle in lane 2, number of lanes'))
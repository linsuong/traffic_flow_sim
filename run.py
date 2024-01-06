import numpy as np
import matplotlib.pyplot as plt
import random
import os
from base import Simulation
from base import Vehicle
from base import Obstacle
from base import Road

save_location = r'C:\Users\linus\Documents\traffic simulation plots\final plots'

def start_simulation(max_velocity = 10, slow_prob = 0.2, length = 2000, density = 0.4, number_of_lanes = 1,
                     obstacle_position = None, obstacle_lane = None, obstacle_start_time = None, obstacle_stop_time = None):
    steps = 2000
    seeds = 100
    random.seed(seeds)
    sim = Simulation()
    sim.Vehicle = Vehicle(max_velocity, slow_prob)
    sim.Road = Road(length, density, number_of_lanes)
    if obstacle_position is not None:
        sim.add_obstacle(obstacle_start_time, obstacle_stop_time, obstacle_position, length = 1, lane = obstacle_lane)
    sim.initialize()
    sim.update(steps)

    return sim, steps
'''
##Normal Plotting##
fig, ax = plt.subplots(1, 1, figsize = (10, 10))
sim, steps = start_simulation()
sim.plot_timespace(steps, lane = 1, ax = ax)
plt.savefig(os.path.join(save_location, 'standard simulation 1 lane.png'))

fig, ax = plt.subplots(1, 2, figsize = (20, 10))
sim, steps = start_simulation(number_of_lanes= 2)
sim.plot_timespace(steps, lane = 1, ax = ax[0])
sim.plot_timespace(steps, lane = 2, ax = ax[1])
plt.savefig(os.path.join(save_location, 'standard simulation 2 lane.png'))

fig, ax = plt.subplots(1, 3, figsize = (30, 10))
sim, steps = start_simulation(number_of_lanes= 3)
sim.plot_timespace(steps, lane = 1, ax = ax[0])
sim.plot_timespace(steps, lane = 2, ax = ax[1])
sim.plot_timespace(steps, lane = 3, ax = ax[2])
plt.savefig(os.path.join(save_location, 'standard simulation 3 lane.png'))

fig, ax = plt.subplots(2, 2, figsize = (20, 20))
sim, steps = start_simulation(number_of_lanes= 4)
sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
plt.savefig(os.path.join(save_location, 'standard simulation 4 lane.png'))

##Normal plotting with obstacle##
fig, ax = plt.subplots(1, 1, figsize = (10, 10))
sim, steps = start_simulation(number_of_lanes = 1, obstacle_position= 1000, obstacle_start_time= 200, obstacle_stop_time= 800, obstacle_lane = 1)
sim.plot_timespace(steps, lane = 1, ax = ax)
plt.savefig(os.path.join(save_location, 'standard simulation with obstacle 1 lane.png'))

fig, ax = plt.subplots(1, 2, figsize = (20, 10))
sim, steps = start_simulation(number_of_lanes = 2, obstacle_position= 1000, obstacle_start_time= 200, obstacle_stop_time= 800, obstacle_lane = 1)
sim.plot_timespace(steps, lane = 1, ax = ax[0])
sim.plot_timespace(steps, lane = 2, ax = ax[1])
plt.savefig(os.path.join(save_location, 'standard simulation with obstacle 2 lane.png'))

fig, ax = plt.subplots(1, 3, figsize = (30, 10))
sim, steps = start_simulation(number_of_lanes = 3, obstacle_position= 1000, obstacle_start_time= 200, obstacle_stop_time= 800, obstacle_lane = 1)
sim.plot_timespace(steps, lane = 1, ax = ax[0])
sim.plot_timespace(steps, lane = 2, ax = ax[1])
sim.plot_timespace(steps, lane = 3, ax = ax[2])
plt.savefig(os.path.join(save_location, 'standard simulation with obstacl 3 lane.png'))

fig, ax = plt.subplots(2, 2, figsize = (20, 20))
sim, steps = start_simulation(number_of_lanes = 4, obstacle_position= 1000, obstacle_start_time= 200, obstacle_stop_time= 800, obstacle_lane = 1)
sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
plt.savefig(os.path.join(save_location, 'standard simulation with obstacle 4 lane.png'))
'''
##Velocity Iteration##
fig, ax = plt.subplots(1, 1, figsize = (10, 10))
for i in range(2, 16, 2):
    sim, steps = start_simulation(number_of_lanes = 1, max_velocity= i)
    sim.plot_timespace(steps, lane = 1, ax = ax)
    plt.savefig(os.path.join(save_location, f'1 lane simulation velocity {i}.png'))

fig, ax = plt.subplots(1, 2, figsize = (20, 10))
for i in range(2, 16, 2):
    sim, steps = start_simulation(number_of_lanes = 2, max_velocity= i)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    plt.savefig(os.path.join(save_location, f'2 lane simulation velocity {i}.png'))

fig, ax = plt.subplots(1, 3, figsize = (30, 10))
for i in range(2, 16, 2):
    sim, steps = start_simulation(number_of_lanes = 3, max_velocity= i)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    sim.plot_timespace(steps, lane = 3, ax = ax[2])
    plt.savefig(os.path.join(save_location, f'3 lane simulation velocity {i}.png'))

fig, ax = plt.subplots(2, 2, figsize = (20, 20))
for i in range(2, 16, 2):
    sim, steps = start_simulation(number_of_lanes = 4, max_velocity= i)
    sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
    sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
    sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
    plt.savefig(os.path.join(save_location, f'4 lane simulation velocity {i}.png'))

##Velocity simulation with obstacle

fig, ax = plt.subplots(1, 1, figsize = (10, 10))
for i in range(2, 16, 2):
    sim, steps = start_simulation(number_of_lanes = 4, max_velocity= i)sim.plot_timespace(steps, lane = 1, ax = ax)
    plt.savefig(os.path.join(save_location, f'1 lane simulation with obstacle and velocity {i}.png'))

fig, ax = plt.subplots(1, 2, figsize = (20, 10))
for i in range(2, 16, 2):
    sim, steps = start_simulation(number_of_lanes = 4, max_velocity= i)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    plt.savefig(os.path.join(save_location, f'2 lane simulation with obstacle and velocity {i}.png'))

fig, ax = plt.subplots(1, 3, figsize = (30, 10))
for i in range(2, 16, 2):
    sim, steps = start_simulation(number_of_lanes = 4, max_velocity= i)sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    sim.plot_timespace(steps, lane = 3, ax = ax[2])
    plt.savefig(os.path.join(save_location, f'3 lane simulation with obstacle and velocity {i}.png'))

fig, ax = plt.subplots(2, 2, figsize = (20, 20))
for i in range(2, 16, 2):
    sim, steps = start_simulation(number_of_lanes = 4, max_velocity= i)sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
    sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
    sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
    plt.savefig(os.path.join(save_location, f'4 lane simulation with obstacle and velocity {i}.png'))

continue_query = input("Continue simulation? This will take a while! (y/n):")
if continue_query.lower() == "n":
    break

else:
    print("Continuing with simulation")
    
##Density-Velocity Iterations
    steps = 2000
    seeds = 100
    random.seed(seeds)
    fig, ax = plt.subplots(1, 1, figsize = (20, 10))
    for i in range(2, 16, 2): #velocities
        for j in range(1, 5, 1): #number of lanes
            Simulation.plot_density(steps, length = 2000, max_velocity= i, slow_prob= 0.2, number_of_lanes= j, ax = ax)
        plt.savefig(save_location, f'Density Iteration, velocity {i}')

print('all complete.')


import numpy as np
import matplotlib.pyplot as plt
import random
import os
from base import Simulation
from base import Vehicle
from base import Obstacle
from base import Road

save_location = r'C:\Users\linus\Documents\traffic simulation plots\final plots'


def start_simulation(max_velocity = 10, slow_prob = 0.2, length = 1000, density = 0.4, number_of_lanes = 1,
                     obstacle_position = None, obstacle_lane = None, obstacle_start_time = None, obstacle_stop_time = None):
    steps = 1000
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

##Normal Plotting##
sim, steps = start_simulation()
sim.plot_timespace(steps, lane = 1)
plt.savefig(os.path.join(save_location, 'standard simulation 1 lane.png'))
sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, plot_obstacle= True)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation 1 lane.png'))

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
sim, steps = start_simulation(number_of_lanes= 2)
sim.plot_timespace(steps, lane = 1, ax = ax[0])
sim.plot_timespace(steps, lane = 2, ax = ax[1])
plt.savefig(os.path.join(save_location, 'standard simulation 2 lane.png'))

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[0], plot_obstacle= True)
sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 2, ax = ax[1], plot_obstacle= True)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation 2 lane.png'))

fig, ax = plt.subplots(1, 3, figsize = (15, 5))
sim, steps = start_simulation(number_of_lanes= 3)
sim.plot_timespace(steps, lane = 1, ax = ax[0])
sim.plot_timespace(steps, lane = 2, ax = ax[1])
sim.plot_timespace(steps, lane = 3, ax = ax[2])
plt.savefig(os.path.join(save_location, 'standard simulation 3 lane.png'))

fig, ax = plt.subplots(1, 3, figsize = (15, 5))
sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[0], plot_obstacle= True)
sim.avg_velocity_plot(steps, lane = 2, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[1], plot_obstacle= True)
sim.avg_velocity_plot(steps, lane = 3, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[2], plot_obstacle= True)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation 3 lane.png'))

fig, ax = plt.subplots(2, 2, figsize = (10, 10))
sim, steps = start_simulation(number_of_lanes= 4)
sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
plt.savefig(os.path.join(save_location, 'standard simulation 4 lane.png'))

fig, ax = plt.subplots(2, 2, figsize = (10, 10))
sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[0][0], plot_obstacle= True)
sim.avg_velocity_plot(steps, lane = 2, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[1][0], plot_obstacle= True)
sim.avg_velocity_plot(steps, lane = 3, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[0][1], plot_obstacle= True)
sim.avg_velocity_plot(steps, lane = 4, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[1][1], plot_obstacle= True)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation 4 lane.png'))

##Normal plotting with obstacle##
fig, ax = plt.subplots(1, 1, figsize = (5, 5))
sim, steps = start_simulation(number_of_lanes = 1, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
sim.plot_timespace(steps, lane = 1, ax = ax)
plt.savefig(os.path.join(save_location, 'standard simulation with obstacle 1 lane.png'))
sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, plot_obstacle= True)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation 1 lane.png'))

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
sim, steps = start_simulation(number_of_lanes = 2, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
sim.plot_timespace(steps, lane = 1, ax = ax[0])
sim.plot_timespace(steps, lane = 2, ax = ax[1])
plt.savefig(os.path.join(save_location, 'standard simulation with obstacle 2 lane.png'))

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[0], plot_obstacle= True)
sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 2, ax = ax[1], plot_obstacle= True)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation with obstacle 2 lane.png'))

fig, ax = plt.subplots(1, 3, figsize = (15, 5))
sim, steps = start_simulation(number_of_lanes = 3, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
sim.plot_timespace(steps, lane = 1, ax = ax[0])
sim.plot_timespace(steps, lane = 2, ax = ax[1])
sim.plot_timespace(steps, lane = 3, ax = ax[2])
plt.savefig(os.path.join(save_location, 'standard simulation with obstacle 3 lane.png'))

fig, ax = plt.subplots(1, 3, figsize = (15, 5))
sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[0], plot_obstacle= True)
sim.avg_velocity_plot(steps, lane = 2, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[1], plot_obstacle= True)
sim.avg_velocity_plot(steps, lane = 3, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[2], plot_obstacle= True)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation 3 lane.png'))

fig, ax = plt.subplots(1, 3, figsize = (15, 5))
sim, steps = start_simulation(number_of_lanes = 3, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 2)
sim.plot_timespace(steps, lane = 1, ax = ax[0])
sim.plot_timespace(steps, lane = 2, ax = ax[1])
sim.plot_timespace(steps, lane = 3, ax = ax[2])
plt.savefig(os.path.join(save_location, 'standard simulation with obstacle (in middle) 3 lane.png'))

fig, ax = plt.subplots(1, 3, figsize = (15, 5))
sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[0], plot_obstacle= True)
sim.avg_velocity_plot(steps, lane = 2, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[1], plot_obstacle= True)
sim.avg_velocity_plot(steps, lane = 3, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[2], plot_obstacle= True)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation with obstacle (in middle) 3 lane.png'))

fig, ax = plt.subplots(2, 2, figsize = (10, 10))
sim, steps = start_simulation(number_of_lanes = 4, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
plt.savefig(os.path.join(save_location, 'standard simulation with obstacle 4 lane.png'))

fig, ax = plt.subplots(2, 2, figsize = (10, 10))
sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[0][0], plot_obstacle= True)
sim.avg_velocity_plot(steps, lane = 2, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[1][0], plot_obstacle= True)
sim.avg_velocity_plot(steps, lane = 3, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[0][1], plot_obstacle= True)
sim.avg_velocity_plot(steps, lane = 4, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[1][1], plot_obstacle= True)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation with obstacle 4 lane.png'))

fig, ax = plt.subplots(2, 2, figsize = (10, 10))
sim, steps = start_simulation(number_of_lanes = 4, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 2)
sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
plt.savefig(os.path.join(save_location, 'standard simulation with obstacle (in middle) 4 lane.png'))

fig, ax = plt.subplots(2, 2, figsize = (10, 10))
sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[0][0], plot_obstacle= True)
sim.avg_velocity_plot(steps, lane = 2, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[1][0], plot_obstacle= True)
sim.avg_velocity_plot(steps, lane = 3, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[0][1], plot_obstacle= True)
sim.avg_velocity_plot(steps, lane = 4, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax[1][1], plot_obstacle= True)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation with obstacle (in middle) 4 lane.png'))

##Velocity Iteration##
for i in range(0, 16, 2):
    sim, steps = start_simulation(number_of_lanes = 1, max_velocity= i)
    sim.plot_timespace(steps, lane = 1)
    plt.savefig(os.path.join(save_location, f'1 lane simulation velocity {i}.png'))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 1 lane simulation velocity {i}.png'))

for i in range(0, 16, 2):
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    sim, steps = start_simulation(number_of_lanes = 2, max_velocity= i)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    plt.savefig(os.path.join(save_location, f'2 lane simulation velocity {i}.png'))
    
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 2, ax = ax[1], plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 2 lane simulation velocity {i}.png'))

for i in range(0, 16, 2):
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sim, steps = start_simulation(number_of_lanes = 3, max_velocity= i)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    sim.plot_timespace(steps, lane = 3, ax = ax[2])
    plt.savefig(os.path.join(save_location, f'3 lane simulation velocity {i}.png'))

    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 2, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[1], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 3, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[2], plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 3 lane simulation velocity {i}.png'))

for i in range(0, 16, 2):
    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    sim, steps = start_simulation(number_of_lanes = 4, max_velocity= i)
    sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
    sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
    sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
    plt.savefig(os.path.join(save_location, f'4 lane simulation velocity {i}.png'))

    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0][0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 2, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[1][0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 3, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0][1], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 4, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[1][1], plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 4 lane simulation velocity {i}.png'))

##Velocity simulation with obstacle##
for i in range(0, 16, 2):
    fig, ax = plt.subplots(1, 1, figsize = (5, 5))
    sim, steps = start_simulation(number_of_lanes = 1, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
    sim.plot_timespace(steps, lane = 1, ax = ax)
    plt.savefig(os.path.join(save_location, f'1 lane simulation with obstacle and velocity {i}.png'))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 1 lane simulation with obstacle and velocity {i}.png'))

for i in range(0, 16, 2):
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    sim, steps = start_simulation(number_of_lanes = 2, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    plt.savefig(os.path.join(save_location, f'2 lane simulation with obstacle and velocity {i}.png'))

    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 2, ax = ax[1], plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 2 lane simulation with obstacle and velocity {i}.png'))

for i in range(0, 16, 2):
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sim, steps = start_simulation(number_of_lanes = 3, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    sim.plot_timespace(steps, lane = 3, ax = ax[2])
    plt.savefig(os.path.join(save_location, f'3 lane simulation with obstacle and velocity {i}.png'))

    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 2, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[1], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 3, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[2], plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 3 lane simulation with obstacle and velocity {i}.png'))


for i in range(0, 16, 2):
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sim, steps = start_simulation(number_of_lanes = 3, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 2)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    sim.plot_timespace(steps, lane = 3, ax = ax[2])
    plt.savefig(os.path.join(save_location, f'3 lane simulation with obstacle (in middle) and velocity {i}.png'))

    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 2, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[1], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 3, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[2], plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 3 lane simulation with obstacle (in middle) and velocity {i}.png'))

for i in range(0, 16, 2):
    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    sim, steps = start_simulation(number_of_lanes = 4, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
    sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
    sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
    sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
    plt.savefig(os.path.join(save_location, f'4 lane simulation with obstacle and velocity {i}.png'))

    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0][0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 2, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[1][0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 3, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0][1], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 4, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[1][1], plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 4 lane simulation with obstacle and velocity {i}.png'))


for i in range(0, 16, 2):
    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    sim, steps = start_simulation(number_of_lanes = 4, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 2)
    sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
    sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
    sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
    plt.savefig(os.path.join(save_location, f'4 lane simulation with obstacle (in middle) and velocity {i}.png'))

    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0][0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 2, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[1][0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 3, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0][1], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 4, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[1][1], plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 4 lane simulation with obstacle (in middle) velocity {i}.png'))

###################################################################################

slowprobs = np.arange(0.2, 1.2, 0.2)

##slow prob Iteration##
for i in slowprobs:
    sim, steps = start_simulation(number_of_lanes = 1, max_velocity= i)
    sim.plot_timespace(steps, lane = 1)
    plt.savefig(os.path.join(save_location, f'1 lane simulation slow prob {i}.png'))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 1 lane simulation slow prob {i}.png'))

for i in slowprobs:
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    sim, steps = start_simulation(number_of_lanes = 2, max_velocity= i)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    plt.savefig(os.path.join(save_location, f'2 lane simulation slow prob {i}.png'))
    
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 2, ax = ax[1], plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 2 lane simulation slow prob {i}.png'))

for i in slowprobs:
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sim, steps = start_simulation(number_of_lanes = 3, max_velocity= i)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    sim.plot_timespace(steps, lane = 3, ax = ax[2])
    plt.savefig(os.path.join(save_location, f'3 lane simulation slow prob {i}.png'))

    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 2, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[1], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 3, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[2], plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 3 lane simulation slow prob {i}.png'))

for i in slowprobs:
    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    sim, steps = start_simulation(number_of_lanes = 4, max_velocity= i)
    sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
    sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
    sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
    plt.savefig(os.path.join(save_location, f'4 lane simulation slow prob {i}.png'))

    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0][0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 2, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[1][0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 3, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0][1], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 4, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[1][1], plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 4 lane simulation slow prob {i}.png'))

##slow prob iteration with obstacle##
for i in slowprobs:
    fig, ax = plt.subplots(1, 1, figsize = (5, 5))
    sim, steps = start_simulation(number_of_lanes = 1, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
    sim.plot_timespace(steps, lane = 1, ax = ax)
    plt.savefig(os.path.join(save_location, f'1 lane simulation with obstacle and slow prob {i}.png'))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 1 lane simulation with obstacle and slow prob {i}.png'))

for i in slowprobs:
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    sim, steps = start_simulation(number_of_lanes = 2, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    plt.savefig(os.path.join(save_location, f'2 lane simulation with obstacle and slow prob {i}.png'))

    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 2, ax = ax[1], plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 2 lane simulation with obstacle and slow prob {i}.png'))

for i in slowprobs:
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sim, steps = start_simulation(number_of_lanes = 3, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    sim.plot_timespace(steps, lane = 3, ax = ax[2])
    plt.savefig(os.path.join(save_location, f'3 lane simulation with obstacle and slow prob {i}.png'))

    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 2, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[1], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 3, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[2], plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 3 lane simulation with obstacle and slow prob {i}.png'))


for i in slowprobs:
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sim, steps = start_simulation(number_of_lanes = 3, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 2)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    sim.plot_timespace(steps, lane = 3, ax = ax[2])
    plt.savefig(os.path.join(save_location, f'3 lane simulation with obstacle (in middle) and slow prob {i}.png'))

    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 2, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[1], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 3, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[2], plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 3 lane simulation with obstacle (in middle) and slow prob {i}.png'))

for i in slowprobs:
    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    sim, steps = start_simulation(number_of_lanes = 4, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
    sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
    sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
    sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
    plt.savefig(os.path.join(save_location, f'4 lane simulation with obstacle and slow prob {i}.png'))

    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0][0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 2, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[1][0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 3, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0][1], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 4, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[1][1], plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 4 lane simulation with obstacle and slow prob {i}.png'))


for i in slowprobs:
    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    sim, steps = start_simulation(number_of_lanes = 4, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 2)
    sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
    sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
    sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
    plt.savefig(os.path.join(save_location, f'4 lane simulation with obstacle (in middle) and slow prob {i}.png'))

    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    sim.avg_velocity_plot(steps, lane = 1, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0][0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 2, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[1][0], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 3, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[0][1], plot_obstacle= True)
    sim.avg_velocity_plot(steps, lane = 4, time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax[1][1], plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 4 lane simulation with obstacle (in middle) velocity {i}.png'))

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
'''
seeds = 100
random.seed(seeds)
for j in range(1, 5, 1): #number of lanes
    fig, ax = plt.subplots(1, 1, figsize = (20, 10))
    for i in range(2, 14, 2): #velocities
        Sim = Simulation()
        Sim.plot_density_with_obstacle(steps = 1000, length = 1000, max_velocity= i, slow_prob= 0.2, 
                                       number_of_lanes= j, labels = i, ax = ax)
    fig.suptitle(f'Fundamental relationship with obstacle, max velocities, {j} lanes')
    ax.autoscale()
    ax.legend(loc = 'upper right')
    plt.savefig(os.path.join(save_location, f'Density Iteration dots with obstacle, velocity, {j} lanes'))
''''
#Density-Slow Probability Iteration##
slowprobs = np.arange(0.2, 1.2, 0.2)
seeds = 100
random.seed(seeds)
for j in range(1, 5, 1): #number of lanes
    fig, ax = plt.subplots(1, 1, figsize = (20, 10))
    for i in slowprobs: #slow probabilities
        Sim = Simulation()
        Sim.plot_density(steps = 1000, length = 1000, max_velocity= 10, slow_prob= i, number_of_lanes= j, 
                          labels = i, ax = ax)
    fig.suptitle(f'Fundamental relationship, slow probabilities, {j} lanes')
    ax.autoscale()
    ax.legend(loc = 'upper right')
    plt.savefig(os.path.join(save_location, f'Density Iteration dots, slow probability, {j} lanes'))
    
#with obstacle
'''
seeds = 100
random.seed(seeds)
for j in range(1, 5, 1): #number of lanes
    fig, ax = plt.subplots(1, 1, figsize = (20, 10))
    for i in slowprobs: #slow probabilities
        Sim = Simulation()
        Sim.plot_density_with_obstacle(steps = 1000, length = 1000, max_velocity= 10, slow_prob= i, number_of_lanes= j, 
                                        labels = i, ax = ax)
    fig.suptitle(f'Fundamental relationship with obstacle, slow probabilities, {j} lanes')
    ax.autoscale()
    ax.legend(loc = 'upper right')
    plt.savefig(os.path.join(save_location, f'Density Iteration dots with obstacle, slow probability, {j} lanes'))
'''
print('all complete.')
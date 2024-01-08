import numpy as np
import matplotlib.pyplot as plt
import random
import os
from base import Simulation
from base import Vehicle
from base import Obstacle
from base import Road

save_location = r'C:\Users\linus\Documents\traffic simulation plots\final plots'

def start_simulation(max_velocity = 6, slow_prob = 0.2, length = 1000, 
                     density = 0.4, number_of_lanes = 1,
                     obstacle_position = None, obstacle_lane = None, 
                     obstacle_start_time = None, obstacle_stop_time = None):
    steps = 1000
    seeds = 100
    random.seed(seeds)
    sim = Simulation()
    sim.Vehicle = Vehicle(max_velocity, slow_prob)
    sim.Road = Road(length, density, number_of_lanes)
    if obstacle_position is not None:
        sim.add_obstacle(obstacle_start_time, obstacle_stop_time, 
                         obstacle_position, length = 1, lane = obstacle_lane)
    sim.initialize()
    sim.update(steps)

    return sim, steps

##Normal Plotting##
sim, steps = start_simulation()
sim.plot_timespace(steps, lane = 1)
plt.savefig(os.path.join(save_location, 'standard simulation 1 lane.png'))

fig, ax = plt.subplots()
sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, ax = ax, plot_obstacle= True)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation 1 lane.png'))

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
sim, steps = start_simulation(number_of_lanes= 2)
sim.plot_timespace(steps, lane = 1, ax = ax[0])
sim.plot_timespace(steps, lane = 2, ax = ax[1])
plt.savefig(os.path.join(save_location, 'standard simulation 2 lane.png'))

fig, ax = plt.subplots()
for i in range(1, 3, 1):
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = i, ax = ax, plot_obstacle= True)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation 2 lane.png'))

fig, ax = plt.subplots(1, 3, figsize = (15, 5))
sim, steps = start_simulation(number_of_lanes= 3)
sim.plot_timespace(steps, lane = 1, ax = ax[0])
sim.plot_timespace(steps, lane = 2, ax = ax[1])
sim.plot_timespace(steps, lane = 3, ax = ax[2])
plt.savefig(os.path.join(save_location, 'standard simulation 3 lane.png'))

fig, ax = plt.subplots()
for i in range(1, 4, 1):
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = i, ax = ax, plot_obstacle= True)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation 3 lane.png'))

fig, ax = plt.subplots(2, 2, figsize = (10, 10))
sim, steps = start_simulation(number_of_lanes= 4)
sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
plt.savefig(os.path.join(save_location, 'standard simulation 4 lane.png'))

fig, ax = plt.subplots()
for i in range(1, 5, 1):
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = i, ax = ax[0], plot_obstacle= True)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation 4 lane.png'))

##Normal plotting with obstacle##
fig, ax = plt.subplots(1, 1, figsize = (5, 5))
sim, steps = start_simulation(number_of_lanes = 1, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
sim.plot_timespace(steps, lane = 1, ax = ax)
plt.savefig(os.path.join(save_location, 'standard simulation with obstacle 1 lane.png'))
sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, plot_obstacle= True)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation 1 lane.png'))

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
sim, steps = start_simulation(number_of_lanes = 2, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
sim.plot_timespace(steps, lane = 1, ax = ax[0])
sim.plot_timespace(steps, lane = 2, ax = ax[1])
plt.savefig(os.path.join(save_location, 'standard simulation with obstacle 2 lane.png'))

fig, ax = plt.subplots()
for i in range(1, 3, 1):
    if i == 1:
        plot_obstacles = True

    else:
        plot_obstacles = False

    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                    position_range= 1000, lane = i, ax = ax, plot_obstacle= plot_obstacles)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation with obstacle 2 lane.png'))

fig, ax = plt.subplots(1, 3, figsize = (15, 5))
sim, steps = start_simulation(number_of_lanes = 3, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
sim.plot_timespace(steps, lane = 1, ax = ax[0])
sim.plot_timespace(steps, lane = 2, ax = ax[1])
sim.plot_timespace(steps, lane = 3, ax = ax[2])
plt.savefig(os.path.join(save_location, 'standard simulation with obstacle 3 lane.png'))

fig, ax = plt.subplots()
for i in range(1, 4, 1):
    if i == 1:
        plot_obstacles = True

    else:
        plot_obstacles = False

    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                    position_range= 1000, lane = i, ax = ax, plot_obstacle= plot_obstacles)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation with obstacle 3 lane.png'))

fig, ax = plt.subplots(1, 3, figsize = (15, 5))
sim, steps = start_simulation(number_of_lanes = 3, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 2)
sim.plot_timespace(steps, lane = 1, ax = ax[0])
sim.plot_timespace(steps, lane = 2, ax = ax[1])
sim.plot_timespace(steps, lane = 3, ax = ax[2])
plt.savefig(os.path.join(save_location, 'standard simulation with obstacle (in middle) 3 lane.png'))

fig, ax = plt.subplots()
for i in range(1, 4, 1):
    if i == 1:
        plot_obstacles = True

    else:
        plot_obstacles = False

    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                    position_range= 1000, lane = i, ax = ax, plot_obstacle= plot_obstacles)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation with obstacle (in middle) 3 lane.png'))

fig, ax = plt.subplots(2, 2, figsize = (10, 10))
sim, steps = start_simulation(number_of_lanes = 4, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
plt.savefig(os.path.join(save_location, 'standard simulation with obstacle 4 lane.png'))

fig, ax = plt.subplots()
for i in range(1, 5, 1):
    if i == 1:
        plot_obstacles = True

    else:
        plot_obstacles = False

    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                    position_range= 1000, lane = i, ax = ax, plot_obstacle= plot_obstacles)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation with obstacle 4 lane.png'))

fig, ax = plt.subplots(2, 2, figsize = (10, 10))
sim, steps = start_simulation(number_of_lanes = 4, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 2)
sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
plt.savefig(os.path.join(save_location, 'standard simulation with obstacle (in middle) 4 lane.png'))

fig, ax = plt.subplots()
for i in range(1, 3, 1):
    if i == 1:
        plot_obstacles = True

    else:
        plot_obstacles = False

    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                    position_range= 1000, lane = i, ax = ax, plot_obstacle= plot_obstacles)
plt.savefig(os.path.join(save_location, 'avg vel standard simulation with obstacle (in middle) 4 lane.png'))

#####################################################################
##Velocity Iteration##
for i in range(0, 16, 2):
    sim, steps = start_simulation(number_of_lanes = 1, max_velocity= i)
    sim.plot_timespace(steps, lane = 1)
    plt.savefig(os.path.join(save_location, f'1 lane simulation velocity {i}.png'))

    fig, ax = plt.subplots()
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 1 lane simulation velocity {i}.png'))

for i in range(0, 16, 2):
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    sim, steps = start_simulation(number_of_lanes = 2, max_velocity= i)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    plt.savefig(os.path.join(save_location, f'2 lane simulation velocity {i}.png'))
    
    fig, ax = plt.subplots()
    for i in range(0, 1, 1):
        sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                            position_range= 1000, lane = 1, ax = ax, plot_obstacle= True)
        sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                            position_range= 1000, lane = 2, ax = ax, plot_obstacle= False)
    plt.savefig(os.path.join(save_location, f'avg vel 2 lane simulation velocity {i}.png'))

for i in range(0, 16, 2):
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sim, steps = start_simulation(number_of_lanes = 3, max_velocity= i)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    sim.plot_timespace(steps, lane = 3, ax = ax[2])
    plt.savefig(os.path.join(save_location, f'3 lane simulation velocity {i}.png'))

    fig, ax = plt.subplots()
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= True)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 2, ax = ax, plot_obstacle= False)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 3, ax = ax, plot_obstacle= False)
    plt.savefig(os.path.join(save_location, f'avg vel 3 lane simulation velocity {i}.png'))

for i in range(0, 16, 2):
    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    sim, steps = start_simulation(number_of_lanes = 4, max_velocity= i)
    sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
    sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
    sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
    plt.savefig(os.path.join(save_location, f'4 lane simulation velocity {i}.png'))

    fig, ax = plt.subplots()
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= True)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 2, ax = ax, plot_obstacle= False)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 3, ax = ax, plot_obstacle= False)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 4, ax = ax, plot_obstacle= False)
    plt.savefig(os.path.join(save_location, f'avg vel 4 lane simulation velocity {i}.png'))

##Velocity simulation with obstacle##
for i in range(0, 16, 2):
    sim, steps = start_simulation(number_of_lanes = 1, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
    sim.plot_timespace(steps, lane = 1)
    plt.savefig(os.path.join(save_location, f'1 lane simulation with obstacle and velocity {i}.png'))

    fig, ax = plt.subplots()
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 1 lane simulation with obstacle and velocity {i}.png'))

for i in range(0, 16, 2):
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    sim, steps = start_simulation(number_of_lanes = 2, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    plt.savefig(os.path.join(save_location, f'2 lane simulation with obstacle and velocity {i}.png'))

    fig, ax = plt.subplots()
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= True)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 2, ax = ax, plot_obstacle= False)
    plt.savefig(os.path.join(save_location, f'avg vel 2 lane simulation with obstacle and velocity {i}.png'))

for i in range(0, 16, 2):
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sim, steps = start_simulation(number_of_lanes = 3, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    sim.plot_timespace(steps, lane = 3, ax = ax[2])
    plt.savefig(os.path.join(save_location, f'3 lane simulation with obstacle and velocity {i}.png'))

    fig, ax = plt.subplots()
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= True)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 2, ax = ax, plot_obstacle= False)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 3, ax = ax, plot_obstacle= False)
    plt.savefig(os.path.join(save_location, f'avg vel 3 lane simulation with obstacle and velocity {i}.png'))


for i in range(0, 16, 2):
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sim, steps = start_simulation(number_of_lanes = 3, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 2)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    sim.plot_timespace(steps, lane = 3, ax = ax[2])
    plt.savefig(os.path.join(save_location, f'3 lane simulation with obstacle (in middle) and velocity {i}.png'))

    fig, ax = plt.subplots()
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= True)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 2, ax = ax, plot_obstacle= False)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 3, ax = ax, plot_obstacle= False)
    plt.savefig(os.path.join(save_location, f'avg vel 3 lane simulation with obstacle (in middle) and velocity {i}.png'))

for i in range(0, 16, 2):
    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    sim, steps = start_simulation(number_of_lanes = 4, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
    sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
    sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
    sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
    plt.savefig(os.path.join(save_location, f'4 lane simulation with obstacle and velocity {i}.png'))

    fig, ax = plt.subplots()
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= True)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= False)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= False)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= False)
    plt.savefig(os.path.join(save_location, f'avg vel 4 lane simulation with obstacle and velocity {i}.png'))


for i in range(0, 16, 2):
    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    sim, steps = start_simulation(number_of_lanes = 4, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 2)
    sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
    sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
    sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
    plt.savefig(os.path.join(save_location, f'4 lane simulation with obstacle (in middle) and velocity {i}.png'))

    fig, ax = plt.subplots()
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= True)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= False)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= False)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= False)
    plt.savefig(os.path.join(save_location, f'avg vel 4 lane simulation with obstacle (in middle) velocity {i}.png'))

###################################################################################

##Slow Prob Iteration##
slowprobs = np.arange(0.2, 1.2, 0.2)

for i in slowprobs:
    sim, steps = start_simulation(number_of_lanes = 1, max_velocity= i)
    sim.plot_timespace(steps, lane = 1)
    plt.savefig(os.path.join(save_location, f'1 lane simulation slow prob {i}.png'))

    fig, ax = plt.subplots()
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 1 lane simulation slow prob {i}.png'))

for i in slowprobs:
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    sim, steps = start_simulation(number_of_lanes = 2, max_velocity= i)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    plt.savefig(os.path.join(save_location, f'2 lane simulation slow prob {i}.png'))
    
    fig, ax = plt.subplots()
    for i in range(0, 1, 1):
        sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                            position_range= 1000, lane = 1, ax = ax, plot_obstacle= True)
        sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                            position_range= 1000, lane = 2, ax = ax, plot_obstacle= False)
    plt.savefig(os.path.join(save_location, f'avg vel 2 lane simulation slow prob {i}.png'))

for i in slowprobs:
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sim, steps = start_simulation(number_of_lanes = 3, max_velocity= i)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    sim.plot_timespace(steps, lane = 3, ax = ax[2])
    plt.savefig(os.path.join(save_location, f'3 lane simulation slow prob {i}.png'))

    fig, ax = plt.subplots()
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= True)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 2, ax = ax, plot_obstacle= False)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 3, ax = ax, plot_obstacle= False)
    plt.savefig(os.path.join(save_location, f'avg vel 3 lane simulation slow prob {i}.png'))

for i in slowprobs:
    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    sim, steps = start_simulation(number_of_lanes = 4, max_velocity= i)
    sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
    sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
    sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
    plt.savefig(os.path.join(save_location, f'4 lane simulation slow prob {i}.png'))

    fig, ax = plt.subplots()
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= True)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 2, ax = ax, plot_obstacle= False)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 3, ax = ax, plot_obstacle= False)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 4, ax = ax, plot_obstacle= False)
    plt.savefig(os.path.join(save_location, f'avg vel 4 lane simulation slow prob {i}.png'))

##Slow Prob simulation with obstacle##
for i in slowprobs:
    sim, steps = start_simulation(number_of_lanes = 1, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
    sim.plot_timespace(steps, lane = 1)
    plt.savefig(os.path.join(save_location, f'1 lane simulation with obstacle and slow prob {i}.png'))

    fig, ax = plt.subplots()
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                      position_range= 1000, lane = 1, plot_obstacle= True)
    plt.savefig(os.path.join(save_location, f'avg vel 1 lane simulation with obstacle and slow prob {i}.png'))

for i in slowprobs:
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    sim, steps = start_simulation(number_of_lanes = 2, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    plt.savefig(os.path.join(save_location, f'2 lane simulation with obstacle and slow prob {i}.png'))

    fig, ax = plt.subplots()
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= True)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 2, ax = ax, plot_obstacle= False)
    plt.savefig(os.path.join(save_location, f'avg vel 2 lane simulation with obstacle and slow prob {i}.png'))

for i in slowprobs:
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sim, steps = start_simulation(number_of_lanes = 3, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    sim.plot_timespace(steps, lane = 3, ax = ax[2])
    plt.savefig(os.path.join(save_location, f'3 lane simulation with obstacle and slow prob {i}.png'))

    fig, ax = plt.subplots()
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= True)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 2, ax = ax, plot_obstacle= False)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 3, ax = ax, plot_obstacle= False)
    plt.savefig(os.path.join(save_location, f'avg vel 3 lane simulation with obstacle and slow prob {i}.png'))


for i in slowprobs:
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sim, steps = start_simulation(number_of_lanes = 3, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 2)
    sim.plot_timespace(steps, lane = 1, ax = ax[0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1])
    sim.plot_timespace(steps, lane = 3, ax = ax[2])
    plt.savefig(os.path.join(save_location, f'3 lane simulation with obstacle (in middle) and slow prob {i}.png'))

    fig, ax = plt.subplots()
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= True)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 2, ax = ax, plot_obstacle= False)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 3, ax = ax, plot_obstacle= False)
    plt.savefig(os.path.join(save_location, f'avg vel 3 lane simulation with obstacle (in middle) and slow prob {i}.png'))

for i in slowprobs:
    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    sim, steps = start_simulation(number_of_lanes = 4, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 1)
    sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
    sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
    sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
    plt.savefig(os.path.join(save_location, f'4 lane simulation with obstacle and slow prob {i}.png'))

    fig, ax = plt.subplots()
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= True)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= False)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= False)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= False)
    plt.savefig(os.path.join(save_location, f'avg vel 4 lane simulation with obstacle and slow prob {i}.png'))


for i in slowprobs:
    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    sim, steps = start_simulation(number_of_lanes = 4, max_velocity= i, obstacle_position= 200, obstacle_start_time= 10, obstacle_stop_time= 210, obstacle_lane = 2)
    sim.plot_timespace(steps, lane = 1, ax = ax[0][0])
    sim.plot_timespace(steps, lane = 2, ax = ax[1][0])
    sim.plot_timespace(steps, lane = 3, ax = ax[0][1])
    sim.plot_timespace(steps, lane = 4, ax = ax[1][1])
    plt.savefig(os.path.join(save_location, f'4 lane simulation with obstacle (in middle) and slow prob {i}.png'))

    fig, ax = plt.subplots()
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= True)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= False)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= False)
    sim.avg_velocity_plot(time_start= 0, time_stop= 1000, position= 0, 
                        position_range= 1000, lane = 1, ax = ax, plot_obstacle= False)
    plt.savefig(os.path.join(save_location, f'avg vel 4 lane simulation with obstacle (in middle) slow prob {i}.png'))

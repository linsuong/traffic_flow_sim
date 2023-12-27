import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import copy
import random
import os
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from bisect import bisect_left

class Vehicle:
    def __init__ (self, length = 1, target_velocity = 4, max_velocity = 5, 
                    acceleration_time = 0.5, deceleration_time = 0.5, slow_prob = 0.3, kindness = False, reckless = False):
        #self.id = id
        self.length = length
        self.velocity = []
        self.position = []
        self.lane = None
        #self.current_velocity = current_velocity
        self.target_velocity = target_velocity 
        self.max_velocity = max_velocity
        self.acceleration_time = acceleration_time
        self.deceleration_time = deceleration_time
        self.slow_prob = slow_prob
        self.kindness = kindness
        self.reckless = reckless

    def acceleration(self):
        factor = (self.target_velocity - self.velocity)/self.acceleration_time
        final_velocity = self.velocity * factor
        #TODO: Is this the right way to do this? Or should the stop/unstop function be used?
        return final_velocity

    def stop(self, status):  #use damping equation for a stopped vehicle
        if status == True:
            #TODO: insert damping function implementation
            self.velocity = 0 
        else:
            #TODO: unstop function
            pass

    def lane_switch(self):
        #for i in range(self.Road.lanes):
            #if self.
        #TODO: make a lane switch fucntion that allows it to turn/switch lanes, lane switching also needs to check for headway
        # this can be implemented with a recklessness prbability maybe(?)
        pass

class Traffic_Light:
    def __init__ (self, color, position):
        self.color = color
        self.position = position
        self.Vehicle = Vehicle()
        self.color = ['red', 'green'] 

    def status(self, color):
        color = random.choice(self.color)
        print(color)

        if color == 'red':
            self.Vehicle.target_velocity = 0
            self.Vehicle.acceleration(self)
            self.Vehicle.stop(self, True)

            #TODO = set timer for red light to turn into green light
    
        if color == 'green':
            #TODO = set timer for green to turn into red - can this be done compactly?
            pass
      
class Obstacle:
    def __init__(self, start_time, end_time, position, length, lane):
        self.start_time = start_time
        self.end_time = end_time
        self.position = position
        self.length = length
        self.lane = lane

class Road:
    def __init__(self, length=100, density=1 / 100, speed_limit=2, number_of_lanes = 1, bend=False):
        self.length = length
        self.density = density
        self.number = int(density * length)
        self.speed_limit = speed_limit
        self.number_of_lanes = number_of_lanes
        self.bend = bend
        self.obstacle = None

    def has_obstacle(self, time_step, position, lane):
        return (
            self.obstacle is not None
            and self.obstacle.start_time <= time_step < self.obstacle.end_time
            and lane == self.obstacle.lane
            and self.obstacle.position <= position < (self.obstacle.position + self.obstacle.length)
            #TODO: add something for lanes
        )

    '''
    def bend(angle, entrance_length, exit_length):
        if self.bend = True:
            angle = 

            #TODO: Add "bend" fucntion to Road class to simulate bends

    '''
class Network: 
    #TODO: add connection function that joins roads together to form a network.
    def __init__(self) -> None:
        self.roads = []
        self.connect = {}
        self.Road = Road()

    def connect_roads(self, roads):
        #TODO: think of a way to join roads at a point in the road - this may be not the best way to do this, 
        #since the road generation is quite iffy. could set the joint at the end of the road? 
        #road generation is not a np.zeros array, so specifying a joint will be hard.
        pass
    
class Simulation:
    def __init__(self, save = False, output_dir = None):
        self.Road = Road()
        self.Vehicle = Vehicle()
        self.velocities = None
        self.positions = None
        self.data = []
        self.velocity_data = []
        self.output_dir = output_dir

        if save:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)


    def initialize(self):
        self.positions = []
        self.velocities = []
        self.lanes = []

        if self.Road.density > 1.0:
            raise ValueError("Density cannot be greater than 1.0")
        
        if self.Road.number > self.Road.length:
            raise ValueError("Density is too high relative to road length")

        else:
            self.Road.number = int(self.Road.length * self.Road.density)
            self.positions_by_lane = [[] for _ in range(self.Road.number_of_lanes)]
            self.velocities_by_lane = [[] for _ in range(self.Road.number_of_lanes)]

            for i in range(self.Road.number_of_lanes):
                position = random.sample(range(0, self.Road.length), self.Road.number)
                velocity = random.choices(range(0, self.Vehicle.max_velocity + 1), k = self.Road.number)
                position.sort()
                self.positions_by_lane[i] = position
                self.velocities_by_lane[i] = velocity

            #print("positions:", self.positions_by_lane)
            #print("velocities:", self.velocities_by_lane)    

        return self.positions_by_lane, self.velocities_by_lane

    def add_obstacle(self, start_time, end_time, position, length, lane):
        self.Road.obstacle = Obstacle(start_time, end_time, position, length, lane)

    def update(self, steps, flow_rate_interval = None):
        self.lane_swap = None
        self.flow_rate = []
        self.passes_per_lane = [0] * self.Road.number_of_lanes
                
        if self.velocities is None or self.positions is None:
            raise Exception("Please call initialize() before update()")
        
        else:
            for step in range(steps):
                #print('time', step)
                positions_after_lane_swap = [[] for _ in range(self.Road.number_of_lanes)]
                velocities_after_lane_swap = [[] for _ in range(self.Road.number_of_lanes)]
                positions_after_moving = [[] for _ in range(self.Road.number_of_lanes)]
                velocities_after_moving = [[] for _ in range(self.Road.number_of_lanes)]

                if self.Road.number_of_lanes != 1:
                    for lane_number in range(self.Road.number_of_lanes):
                        #print('lane swapping, t =', step, 'current lane =', lane_number)
                        current_lane_velocities = self.velocities_by_lane[lane_number]
                        current_lane_positions = self.positions_by_lane[lane_number]

                        if self.Road.number_of_lanes == 2:
                            if lane_number == 0:
                                next_lane_number = 1

                            else:
                                next_lane_number = 0

                        else:
                            if lane_number + 1 == self.Road.number_of_lanes:
                                next_lane_number = (lane_number - 1)

                            elif lane_number == 0:
                                next_lane_number = (lane_number + 1)
                            
                            else:
                                #print('middle lane')
                                if len(self.positions_by_lane[lane_number + 1]) > len(self.positions_by_lane[lane_number - 1]):
                                    next_lane_number = (lane_number - 1)

                                else:
                                    next_lane_number = (lane_number + 1)

                        next_lane_positions = self.positions_by_lane[next_lane_number]  
                        next_lane_velocities = self.velocities_by_lane[next_lane_number]

                        # Loop through a copy of positions in the current lane
                        indices_to_remove = []
                        moved_cars_indices = []
                        for i in reversed(range(len(list(current_lane_positions)))):
                            if i not in moved_cars_indices:
                                if random.random() < 0.4:
                                    position, velocity = current_lane_positions[i], current_lane_velocities[i]
                                    next_index = (i + 1) % len(current_lane_positions)
                                    prev_index = (i + 1) % len(current_lane_positions)

                                    # Check if there is enough space, and see if swaping is advantageous etc....
                                    if next_lane_positions:
                                        #investigate this! kindness is here on the lane swapping
                                        empty_space_required_forward = next_lane_velocities[next_index % len(next_lane_velocities)] + 1
                                        empty_space_required_backward = next_lane_velocities[prev_index % len(next_lane_velocities)] + 1

                                        empty_positions_ahead = set((position + offset) % self.Road.length for offset in range(0, empty_space_required_forward + 1))
                                        empty_positions_behind = set((position - offset) % self.Road.length for offset in range(0, empty_space_required_backward + 1))

                                        if not any(pos in next_lane_positions for pos in empty_positions_ahead) and not any(pos in next_lane_positions for pos in empty_positions_behind):
                                            #print('removing car in position', position, 'in lane', lane_number + 1, 'moving to position', position, 'in lane', next_lane_number + 1)
                                            insert_index = bisect_left(next_lane_positions, position)
                                            next_lane_positions.insert(insert_index, position)
                                            next_lane_velocities.insert(insert_index, velocity)

                                            moved_cars_indices.append(i)
                                            indices_to_remove.append(i)
                                    
                                    else:
                                        #print('removing car in position', position, 'in lane', lane_number + 1, 'moving to position', position, 'in lane', next_lane_number + 1)
                                        insert_index = bisect_left(next_lane_positions, position)
                                        next_lane_positions.insert(insert_index, position)
                                        next_lane_velocities.insert(insert_index, velocity)

                                        moved_cars_indices.append(i)
                                        indices_to_remove.append(i)

                        
                        #if self.lane_swap == True:
                        for index in indices_to_remove:
                            #print('removing index', index)
                            current_lane_positions.pop(index)
                            current_lane_velocities.pop(index)

                        positions_after_lane_swap[lane_number] = current_lane_positions
                        velocities_after_lane_swap[lane_number] = current_lane_velocities

                    self.positions_by_lane = positions_after_lane_swap
                    self.velocities_by_lane = velocities_after_lane_swap
                    #print('after swap', positions_after_lane_swap, velocities_after_lane_swap)

                for lane_number in range(self.Road.number_of_lanes):
                    print('lane moving, t =', step, 'current lane =', lane_number)
                    for i in range(len(self.velocities_by_lane[lane_number])):
                        position = self.positions_by_lane[lane_number][i]
                        velocity = self.velocities_by_lane[lane_number][i]
                        headway = (self.positions_by_lane[lane_number][(i + 1) % len(self.positions_by_lane[lane_number])] - self.positions_by_lane[lane_number][i]) % self.Road.length
                        
                        if self.Road.has_obstacle(step, position, lane_number + 1):
                            print('obstacle detected, t =', step)
                            if self.Road.obstacle.start_time <= step <= self.Road.obstacle.end_time:
                                obstacle_range = range(self.Road.obstacle.position - self.Vehicle.max_velocity, self.Road.obstacle.position)

                                if position in obstacle_range:
                                    velocity = 0

                                else:
                                    velocity = min(velocity + 1, self.Vehicle.max_velocity)
                                    velocity = min(velocity, max(headway - 1, 0))

                        else:
                            velocity = min(velocity + 1, self.Vehicle.max_velocity)
                            velocity = min(velocity, max(headway - 1, 0))

                        if velocity > 0 and random.random() < self.Vehicle.slow_prob:
                            #print('randomly slowing down at time', step)
                            velocity = max(velocity - 1, 0)

                        new_pos = (position + velocity) % self.Road.length

                        if new_pos < position:
                            #print('last position = ', self.positions_by_lane[lane_number][i], 'new position = ', new_pos)
                            self.passes_per_lane[lane_number] = self.passes_per_lane[lane_number] + 1

                        velocities_after_moving[lane_number].append(velocity)
                        positions_after_moving[lane_number].append(new_pos)
                        
                        self.velocities_by_lane[lane_number][i] = velocity
                        self.positions_by_lane[lane_number][i] = new_pos
                    
                self.data.append(positions_after_moving[:])
                self.velocity_data.append(velocities_after_moving[:])

                #print('after move', self.positions_by_lane, self.velocities_by_lane)
        
        #print(self.passes_per_lane)
        '''
        for i in len(self.passes_per_lane):
            flow_rate = self.passes_per_lane[i]/steps
            self.flow_rate.append(flow_rate)

        print('Flow rate =', self.flow_rate)
        '''

        #print(self.data)
        '''
            for i in range(self.Road.number):
            new_data_contour = [datas[i] for datas in self.data]
            print("id", i, ":", new_data_contour)
        '''
           
     
            #print(self.data)
            #print(self.velocity_data)
            #print(self.positions_by_lane)
            #print(self.velocities_by_lane)

        return self.data, self.velocity_data, self.positions_by_lane, self.velocities_by_lane, self.passes_per_lane
    
    def flow_rate_by_lane(self, steps, lane):
        self.flow_rate = self.passes_per_lane[lane - 1]/steps

        return self.flow_rate

    def flow_rate_total(self, steps):
        self.total_flow_rate = sum(self.passes_per_lane)/steps

        return self.total_flow_rate
    
    def flow_rate_average(self, steps):
        rates = []
        for i in range(self.Road.number_of_lanes):
            rates.append(self.passes_per_lane[i]/steps)

        self.average_flow_rate = sum(rates)/self.Road.number_of_lanes
        
        return self.average_flow_rate
                   
    def plot_timespace(self, steps, lane=None, plot=True, plot_obstacle=True, save=False, folder=None, number=None, ax=None):
        '''
            plots time space diagram using data from self.data

        Args:
            steps (int): time step
            lane(int): lane to plot
            plot (bool, optional): If True, shows plot of graph. Defaults to True.
            save (bool, optional): If True, will save to file. Defaults to False.
            folder (_type_, optional): Save location - required if save is True. Defaults to None.
            number (_type_, optional): Used for keeping track of plots when iterating. Defaults to None.
            ax (matplotlib.axes._subplots.AxesSubplot, optional): The Axes on which to plot the time-space diagram.

        '''
        print('Simulation Complete. Plotting graph...')
        time_steps = range(steps)
        lane_data = [data[lane - 1] for data in self.data]
 
        for i in range(steps):
            new_data = lane_data[i]
            
            if ax is not None:
                ax.plot(new_data, [i] * len(new_data), '.', markersize=0.5, color='grey')

            else:
                plt.plot(new_data, [i] * len(new_data), '.', markersize=0.5, color='grey')

            if self.Road.obstacle is not None and plot_obstacle == True:
                obstacle_range = np.arange(self.Road.obstacle.position, self.Road.obstacle.position + self.Road.obstacle.length, 1)
                obstacle_time_range = np.arange(self.Road.obstacle.start_time, self.Road.obstacle.end_time, 1)

                obstacle_positions = [pos for time in obstacle_time_range for pos in obstacle_range]

                for obstacle_pos, obstacle_time in zip(obstacle_positions, obstacle_time_range):
                    if self.Road.obstacle.lane == lane + 1:
                        plt.plot(obstacle_pos, obstacle_time, 'rx', markersize=5)

        if ax is not None:
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')
            ax.invert_yaxis()
            ax.set_title(f'{self.Road.number_of_lanes} lane configuration, lane {lane}')
            ax.set_xlabel('Vehicle Position')
            ax.set_ylabel('Time')
            #ax.figtext(0.1, 0.05, f'Density = {self.Road.density}, Number of Vehicles = {self.Road.number}, Slow Prob = {self.Vehicle.slow_prob}, Max velocity = {self.Vehicle.max_velocity}', fontsize=9, color='black')
        else:
            plt.gca().xaxis.set_ticks_position('top')
            plt.gca().xaxis.set_label_position('top')
            plt.gca().invert_yaxis()
            plt.title('Time Space diagram')
            plt.xlabel('Vehicle Position')
            plt.ylabel('Time')
            plt.figtext(0.1, 0.05, f'Density = {self.Road.density}, Number of Vehicles = {self.Road.number}, Slow Prob = {self.Vehicle.slow_prob}, Max velocity = {self.Vehicle.max_velocity}', fontsize=9, color='black')

        if save:
            self.output_dir = os.path.join(
                folder, f'Time Space Plot {number}.png')
            if ax is not None:
                fig = ax.figure
            else:
                fig = plt.gcf()
            fig.set_size_inches(12, 12)
            plt.savefig(self.output_dir, dpi=100)
            plt.clf()

        if ax is None:
            plt.show()

    def plot_contour(self, steps):
            time_steps = range(steps)
            time_steps_contour = np.array(time_steps)

            for i in range(self.Road.number):
                new_data_contour = [datas[i] for datas in self.data]
                velocity_data_contour = [vels[i] for vels in self.velocity_data]
                new_data_contour = np.array(new_data_contour)
                velocity_data_contour = np.array(velocity_data_contour)
                #print(velocity_data_contour)
                #print(new_data_contour)
            
                #print(np.shape(new_data_contour))
                #print(np.shape(time_steps))
                #print(np.shape(velocity_data_contour))
                
                plt.scatter(new_data_contour, 
                            time_steps_contour, 
                            c = velocity_data_contour, 
                            marker = 'o', 
                            cmap= cm.Greys, 
                            alpha = .8,
                            norm = Normalize(vmin=0, vmax=self.Vehicle.max_velocity),
                            linewidth = 0,
                            edgecolors= None)

                #plt.title('Scatter Plot')
                #plt.xlabel('New Data')
                #plt.ylabel('Time Steps')
            plt.colorbar(label='Velocity')
            plt.gca().xaxis.set_ticks_position('top')
            plt.gca().xaxis.set_label_position('top')
            plt.gca().invert_yaxis()
            plt.title('Time Space diagram')
            plt.xlabel('Vehicle Position')
            plt.ylabel('Time')
            plt.figtext(0.1, 0.05, f'Density = {self.Road.density}, Number of Vehicles = {self.Road.number}, Slow Prob = {self.Vehicle.slow_prob}, Max velocity = {self.Vehicle.max_velocity}', fontsize=9, color='black')
            plt.show()
            
    def plot_velocity(self, steps, save=False, folder=None):
        time_steps = range(steps)

        for i in range(self.Road.number):
            print("Vehicle ID: %s" % i)
            velocity_data = [item [i] for item in self.velocity_data]
            print("Velocity List: %s" % velocity_data)
            plt.plot(velocity_data, time_steps, '-', markersize=1, color='grey')

            #plt.gca().xaxis.set_ticks_position('top')
            #plt.gca().invert_yaxis()
            plt.title("Velocity - Time diagram")
            plt.xlabel("Vehicle Velocity")
            plt.ylabel("Time")
            plt.figtext(0.1, 0.005, f'Density = {self.Road.density}, Number of Vehicles = {self.Road.number}, Slow Prob = {self.Vehicle.slow_prob}, Max velocity = {self.Vehicle.max_velocity}', fontsize=9, color='black')
            plt.show()

            
    def plot_density(self, steps, length, max_velocity, slow_prob,
                     number_of_lanes, plot=True, isAvg=True, save=False, 
                     folder=None, number=None, plot_lanes=False, ax=None):
        # TODO: Multi-lane implementation.

        densities = []
        flow_rate = []
        flow_rates = [[] for _ in range(number_of_lanes)]
        #print('flow', flow_rates)
        flow_rate_avgs = []

        if plot:
            if isAvg:
                for i in range(10, 990):
                    p = 0.001 * i
                    #print('density: %f' % p)
                    densities.append(p)
                    sim = Simulation()
                    sim.Vehicle = Vehicle(max_velocity, slow_prob)
                    sim.Road = Road(length, density=p, number_of_lanes = number_of_lanes)
                    sim.initialize()
                    sim.update(steps)

                    if plot_lanes == False:
                        flow_rate.append(sim.flow_rate_average(steps))

                    else:
                        for j in range(number_of_lanes):
                            flow_rates[j].append(sim.flow_rate_by_lane(steps, j))
                            #print(flow_rates)

                if ax is not None:
                    if plot_lanes == False:
                        ax.plot(densities, flow_rate, linestyle='-', label= f'{number_of_lanes} lanes')

                    else:
                        for k in range(number_of_lanes):
                            ax.plot(densities, flow_rates[k], linestyle='-', label= f'lane {k + 1}')

                    ax.set_title(f'Average Flow Density Relationship - Number of lanes: {number_of_lanes}')
                    ax.set_xlabel("Density")
                    ax.set_ylabel("Flow rate")
                    ax.legend()
                    #ax.figtext(0.1, 0.005, f'Max velocity = {self.Vehicle.max_velocity}, Slow Prob = {self.Vehicle.slow_prob}', fontsize=9,
                            #color='black')
                    
                else:
                    if plot_lanes == False:
                        plt.plot(densities, flow_rate, linestyle='-', label=f'{number_of_lanes} lanes')
                
                    else:
                        for k in range(number_of_lanes):
                            plt.plot(densities, flow_rates[k], linestyle='.', label=f'Lane {k + 1}')

                    plt.title(f'Average Flow Density Relationship - Number of lanes: {number_of_lanes}')
                    plt.xlabel("Density")
                    plt.ylabel("Flow rate")
                    plt.legend()
                    plt.figtext(0.1, 0.005, f'Max velocity = {max_velocity}, Slow Prob = {slow_prob}', fontsize=9,
                            color='black')
                
                    if save:
                        self.output_dir = os.path.join(folder, f'Flow Density {number}.png')
                        ax.savefig(self.output_dir)
                        plt.clf()

                    else: 
                        plt.show()

    def plot_speed_camera(self, steps, interval, plot_obstacle = True):
        flows = []
        for i in range(10 + interval, steps, interval):
            print(i)
            flow = self.flow_rate_loop(i, interval = interval)
            flows.append(flow)

        fig, ax = plt.subplots()
        ax.plot(range(10 + interval, steps, interval), flows)

        if self.Road.obstacle is not None and plot_obstacle == True:
            rectangle = patches.Rectangle(
                (self.Road.obstacle.start_time, 0),
                self.Road.obstacle.end_time - self.Road.obstacle.start_time,
                max(flows),
                alpha=0.2,
                color='red')
            ax.add_patch(rectangle)

        plt.xlabel("Time Step")
        plt.ylabel("Flow Rate")
        plt.show()

    def avg_velocity_plot(self, time_start, time_stop, position, position_range, plot_obstacle = True):
        average_velocities = []
        time_range = np.arange(time_start, time_stop)

        for i in range(time_start, time_stop):
            total_velocity = 0
            num_car = 0
            positions = self.data[i]
            velocities = self.velocity_data[i]

            for time in range(len(time_range)):
                if position - position_range <= positions[time] <= position + position_range:
                    # Accumulate the velocity and count the number of vehicles
                    total_velocity += velocities[time]
                    num_car += 1

            average_velocity = total_velocity / num_car if num_car > 0 else 0
            average_velocities.append(average_velocity)

        fig, ax = plt.subplots()

        plt.title("Average Velocity at position %s to %s" % (position - position_range, position + position_range))
        if self.Road.obstacle is not None and plot_obstacle == True:
            rectangle = patches.Rectangle(
                (self.Road.obstacle.start_time, 0),
                self.Road.obstacle.end_time - self.Road.obstacle.start_time,
                max(average_velocities),
                alpha=0.2,
                color='red')
            ax.add_patch(rectangle)

        ax.plot(time_range, average_velocities, color="black")
        ax.set_ylabel("Average Velocity")
        ax.set_xlabel("Time")
        plt.figtext(0.1, 0.005, f'Density = {self.Road.density}, Number of Vehicles = {self.Road.number}, Slow Prob = {self.Vehicle.slow_prob}, Max velocity = {self.Vehicle.max_velocity}', fontsize=9, color='black')
        plt.show()


steps = 1000
seeds = 100

random.seed(seeds)
sim = Simulation()
sim.Vehicle = Vehicle(max_velocity=5, slow_prob=0.5)
sim.Road = Road(length=1000, density = 0.5, number_of_lanes = 1)
sim.initialize()
#sim.add_obstacle(start_time= 10, end_time= 30, position= 50, length= 1, lane= 1)
sim.update(steps)
sim.plot_timespace(steps, 1, plot_obstacle= True)

#sim.initialize()
#sim.update(steps)

#fig1, ax1 = plt.subplots()
#sim.plot_density(steps, length = 100, max_velocity = 5, slow_prob = 0.2, plot_lanes = False, number_of_lanes=1, ax = ax1)
#sim.plot_density(steps, length = 100, max_velocity = 5, slow_prob = 0.2, plot_lanes = False, number_of_lanes=2, ax = ax1)
#sim.plot_density(steps, length = 100, max_velocity = 5, slow_prob = 0.2, plot_lanes = False, number_of_lanes=3)
#sim.plot_density(steps, length = 100, max_velocity = 5, slow_prob = 0.2, plot_lanes = False, number_of_lanes=4, ax = ax1)

#plt.show()

#sim.plot_timespace(steps, lane = 1)
#sim.plot_timespace(steps, lane = 2)

"""

fig, ax = plt.subplots(4, 4, figsize=(30, 30))
#fig1, ax1 = plt.subplots(1, 1, figsize=(30,30)) 

steps = 100
seeds = 100
random.seed(seeds)
sim = Simulation()
sim.Vehicle = Vehicle(max_velocity=5, slow_prob=0.2)
sim.Road = Road(length=1000, density=20/100, number_of_lanes=1)
sim.initialize()
sim.update(steps)
sim.plot_timespace(steps, lane=1, ax=ax[0, 0])
#sim.plot_density(steps, number_of_lanes=1, ax = ax1)

steps = 100
seeds = 100
random.seed(seeds)
sim = Simulation()
sim.Vehicle = Vehicle(max_velocity=5, slow_prob=0.2)
sim.Road = Road(length=1000, density=20/100, number_of_lanes=2)
sim.initialize()
sim.update(steps)
sim.plot_timespace(steps, lane=1, ax=ax[1, 0])
sim.plot_timespace(steps, lane=2, ax=ax[1, 1])
#sim.plot_density(steps, number_of_lanes=2, ax=ax1)

sim = Simulation()
sim.Vehicle = Vehicle(max_velocity=5, slow_prob=0.2)
sim.Road = Road(length=1000, density=20/100, number_of_lanes=3)
sim.initialize()
sim.update(steps)
sim.plot_timespace(steps, lane=1, ax=ax[2, 0])
sim.plot_timespace(steps, lane=2, ax=ax[2, 1])
sim.plot_timespace(steps, lane=3, ax=ax[2, 2])
#sim.plot_density(steps, number_of_lanes=3, ax=ax1)

sim = Simulation()
sim.Vehicle = Vehicle(max_velocity=5, slow_prob=0.2)
sim.Road = Road(length=1000, density=20/100, number_of_lanes=4)
sim.initialize()
sim.update(steps)
sim.plot_timespace(steps, lane=1, ax=ax[3, 0])
sim.plot_timespace(steps, lane=2, ax=ax[3, 1])
sim.plot_timespace(steps, lane=3, ax=ax[3, 2])
sim.plot_timespace(steps, lane=4, ax=ax[3, 3])

fig.delaxes(ax[0, 1])
fig.delaxes(ax[0, 2])
fig.delaxes(ax[0, 3])
fig.delaxes(ax[1, 2])
fig.delaxes(ax[1, 3])
fig.delaxes(ax[2, 3])

plt.tight_layout()
plt.show()

#sim.avg_velocity_plot(time_start = 0, time_stop = 100, position = 75, position_range = 25)
#sim.flow_rate_loop(steps)
#sim.plot_contour(steps)
#sim.plot_velocity(steps)
#sim.plot_density(steps)

fig, ax = plt.subplots(5, 1)
steps = 100
seeds = 100
random.seed(seeds)
sim = Simulation()
sim.Vehicle = Vehicle(max_velocity=5, slow_prob=0.2)
sim.Road = Road(length=1000, density=20/100, number_of_lanes=5)
sim.initialize()
sim.add_obstacle(20, 30, 30, 1, lane = 1)
sim.update(steps)

plt.show()
#sim.plot_density(steps, number_of_lanes=2, ax=ax1)


"""
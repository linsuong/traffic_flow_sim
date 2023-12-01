import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import random
import os

class Vehicle:
    def __init__ (self, length = 1, target_velocity = 4, max_velocity = 5, 
                    acceleration_time = 0.5, deceleration_time = 0.5, slow_prob = 0.3, kindness = False, reckless = False):
        #self.id = id
        self.length = length
        self.velocity = []
        self.position = []
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

    def has_obstacle(self, position, time_step, lane):
        return (
            self.obstacle is not None
            and self.obstacle.start_time <= time_step < self.obstacle.end_time
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
    '''
    def initialize(self):
        self.positions = []
        self.velocities = []
        self.lanes = []

        #for _ in range(self.Road.number_of_lanes):
        if self.Road.density > 1.0:
            raise ValueError("Density cannot be greater than 1.0")
    
        self.Road.number = int(self.Road.length * self.Road.density * self.Road.number_of_lanes)

        if self.Road.number > self.Road.length:
            raise ValueError("Density is too high relative to road length")
        
        else:
            #print(self.Road.length)
            self.positions = random.choices(range(self.Road.length), k= self.Road.number)
            self.lanes = random.choices(range(self.Road.number_of_lanes), k = self.Road.number)
            self.velocities = [random.randint(1, self.Vehicle.max_velocity) for i in range(len(self.positions))]
          
            self.positions.sort()
            if np.shape(self.velocities) != np.shape(self.positions):
                print("Number of cars: %s" %np.shape(self.positions))
                print("Number of velocities: %s" %np.shape(self.velocities))
                raise Exception("Error - not all cars have velocity, or too many velocities, not enough cars")

                #else:
                    #print("Vehicles initialised successfully... starting simulation.")

        the_road = np.zeros((self.Road.number_of_lanes, self.Road.length), dtype = int)
            
        for i in range(len(self.positions)):
            the_road[self.lanes[i]][self.positions[i]] = self.velocities[i]

        print("the road:")
        print(the_road)            

        
        if self.Road.number_of_lanes > 1:
            lane_positions = [np.zeros((self.Road.number_of_lanes, len(self.positions)), dtype= int)]
            lane_velocities = np.zeros((self.Road.number_of_lanes, len(self.positions)), dtype= int)
            for i in range(len(self.positions)):
                    lane_positions[self.lanes[i]][i] = self.positions[i]
                    lane_velocities[self.lanes[i]][i] = self.velocities[i]
        
        #print(lane_positions)
        #print(lane_velocities)
        #self.positions = lane_positions
        #self.velocities = lane_velocities 
        print("lanes:", self.lanes)
        print("positions:", self.positions)
        print("velocities:", self.velocities)    

        return self.positions, self.velocities, self.lanes, the_road
    '''

    def initialize(self):
        self.positions = []
        self.velocities = []
        self.lanes = []
        self.road_matrix = np.zeros((self.Road.number_of_lanes, self.Road.length), dtype=int)

        if self.Road.density > 1.0:
            raise ValueError("Density cannot be greater than 1.0")

        self.Road.number = int(self.Road.length * self.Road.density)

        if self.Road.number > self.Road.length:
            raise ValueError("Density is too high relative to road length")

        else:
            for _ in range(self.Road.number):
                lane = random.randint(0, self.Road.number_of_lanes - 1)
                position = random.randint(0, self.Road.length - 1)
                velocity = random.randint(1, self.Vehicle.max_velocity)

                self.positions.append(position)
                self.lanes.append(lane)
                self.velocities.append(velocity)
                self.road_matrix[lane, position] = velocity

        print("the road:")
        print(self.road_matrix)
        print("lanes:", self.lanes)
        print("positions:", self.positions)
        print("velocities:", self.velocities)    

        return self.positions, self.velocities, self.lanes, self.road_matrix

    def add_obstacle(self, start_time, end_time, position, length):
        self.Road.obstacle = Obstacle(start_time, end_time, position, length)

    def update(self, steps):
        if self.velocities is None or self.positions is None:
            raise Exception("Please call initialize() before update()")
        else:
            for step in range(steps):
                new_velocities = []
                new_positions = []

                for i in range(len(self.positions)):
                    velocity = self.velocities[i]
                    headway = (self.positions[(i + 1) % len(self.positions)] - self.positions[i] - 1) % self.Road.length

                    if self.Road.has_obstacle(self.positions[i], step):
                        if self.Road.obstacle.start_time <= step <= self.Road.obstacle.end_time:
                            print('obstacle at time', step)
                            obstacle_range = range(self.Road.obstacle.position - self.Vehicle.max_velocity, self.Road.obstacle.position + 1)

                            if self.positions[i] in obstacle_range:
                                #print("obstacle detected: obstacle range is = ", obstacle_range)
                                velocity = 0

                            else:
                                #print("No obstacle detected at time:", step, ', accelerating...')
                                velocity = min(velocity + 1, self.Vehicle.max_velocity)
                                velocity = min(velocity, max(headway - 1, 0))

                    else:
                        #print('moving normally...', step)
                        velocity = min(velocity + 1, self.Vehicle.max_velocity)
                        velocity = min(velocity, max(headway - 1, 0))

                    if velocity > 0 and random.random() < self.Vehicle.slow_prob:
                        #print('randomly slowing down at time', step)
                        velocity = max(velocity - 1, 0)

                    new_velocities.append(velocity)

                    new_pos = (self.positions[i] + velocity) % self.Road.length
                    new_positions.append(new_pos)

                self.velocity_data.append(new_velocities)
                self.velocities = new_velocities
                new_positions = [(pos + vel) %
                                self.Road.length for pos, vel in zip(self.positions, new_velocities)]
                self.positions = new_positions
                self.data.append(new_positions[:])

        #print(np.shape(self.velocity_data))
        #print(np.shape(self.data))

        for i in range(self.Road.number):
            new_data_contour = [datas[i] for datas in self.data]
            print("id", i, ":", new_data_contour)
        
        print(self.data)
        print(self.velocity_data)

        return self.data, self.velocity_data, new_data_contour

    def flow_rate_ref_point(self, time_interval, reference_point=0):
        num_vehicles_passed = 0
        total_loops = 0

        for k in range(self.Road.number):
            positions = [entry[k] for entry in self.data]
            previous_position = positions[0]
            loops = 0

            for i in range(len(positions)):
                next_position = positions[(i + 1) % len(positions)]

                if previous_position > next_position:
                    loops += 1

                if (next_position >= reference_point and positions[i] <= reference_point) or (next_position <= reference_point and positions[i] >= reference_point):
                    num_vehicles_passed += loops

                previous_position = positions[i]

            total_loops += loops

        # Calculate the total number of loops
        total_loops += (num_vehicles_passed / len(self.data))

        #print("Total Loops: %f" % total_loops)
        #print("Num Vehicles Passed: %d" % num_vehicles_passed)
        
        flow_rate = (num_vehicles_passed / len(self.data)) * (1 / time_interval)
        #print('Flow rate = %f' % flow_rate)

        return flow_rate
    
    def flow_rate_loop(self, steps, interval= None):
        """
        flow rate loop counter

        Args:
            steps (int): 
                time step

            interval (int), optional: 
                time interval if you want to get the flow rate of a certain window.

        Returns:
            flow_rate(float): 
                flow rate at the end of the road
        """
        num_passes = 0

        for k in range(self.Road.number):
            positions = [entry[k] for entry in self.data]
            #print(positions)
            
            if interval is not None:
                positions = positions[steps - interval : steps]
            #print(positions)
            else:
                positions = positions[:steps]

            previous_position = positions[-1]

            for position in positions:
                if previous_position > position:
                    print('position = %d' % position)
                    print('previous position = %d' % previous_position)
                    print(self.Road.length - self.Vehicle.max_velocity + 1)
                    print(self.Vehicle.max_velocity)

                    if position < self.Vehicle.max_velocity + 1 and previous_position > self.Road.length - self.Vehicle.max_velocity - 1:
                        print("yes")
                        num_passes += 1

                    else:
                        print("no")

                previous_position = position

        print("Num Vehicles Passed: %d" % num_passes)
        
        flow_rate = (num_passes / steps)
        print('Flow rate = %f' % flow_rate)

        return flow_rate

            
    def plot_timespace(self, steps, plot=True, plot_obstacle = True, save=False, folder=None, number=None):
        '''
            plots time space diagram using data from self.data

        Args:
            steps (int): time step
            plot (bool, optional): If True, shows plot of graph. Defaults to True.
            save (bool, optional): If True, will save to file. Defaults to False.
            folder (_type_, optional): Save location - required if save is True. Defaults to None.
            number (_type_, optional): Used for keeping track of plots when iterating. Defaults to None.
        
        '''
        print('Simulation Complete. Plotting graph...')
        time_steps = range(steps)
        print(type(time_steps))

        for i in range(self.Road.number):
            new_data = [item[i] for item in self.data]

            plt.plot(new_data, time_steps, '.', markersize=0.5, color='grey')

            if self.Road.obstacle is not None and plot_obstacle == True:
                obstacle_range = np.arange(self.Road.obstacle.position, self.Road.obstacle.position + self.Road.obstacle.length, 1)
                obstacle_time_range = np.arange(self.Road.obstacle.start_time, self.Road.obstacle.end_time, 1)

                obstacle_positions = [pos for time in obstacle_time_range for pos in obstacle_range]

                for obstacle_pos, obstacle_time in zip(obstacle_positions, obstacle_time_range):
                    plt.plot(obstacle_pos, obstacle_time, 'rx', markersize=5)

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
            fig = plt.gcf()
            fig.set_size_inches(12, 12)
            plt.savefig(self.output_dir, dpi=100)
            plt.clf()

        else:
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

            
    def plot_density(self, steps, plot=True, isAvg = True, save = False, folder = None, number = None):
        densities = []
        flow_rate = []
        flow_rates = []
        flow_rate_avgs = []
        
        if plot:
            if isAvg:
                for i in range(10, 250):
                    p = 0.004*i
                    print('density: %f' % p)
                    densities.append(p)
                    sim = Simulation()
                    sim.Road = Road(density= p)                                  
                    sim.initialize()
                    sim.update(steps)
                    flow_rate.append(sim.flow_rate_loop(steps))

                    #for j in range(10):  
                        #sim = Simulation()
                        #sim.Road = Road(density= p)                                  
                        #sim.initialize()
                        #sim.update(steps)
                        #print(sim.flow_rate(steps))
                        #flow_rates.append(sim.flow_rate_loop(steps))       

                #flow_rate_avg = sum(flow_rates)/10
                #flow_rate_avgs.append(flow_rate_avg)
                #print(flow_rate_avgs)
                #print(np.shape(flow_rate_avgs))

                #print("Density:" + str(densities))
                #print(np.shape(densities))
                #print("Flow Rate:" + str(flow_rate_avg))
                #print(np.shape(flow_rate_avg))
                plt.plot(densities, flow_rate, linestyle = '-')
                plt.title('Average Flow Density Relationship')
                plt.xlabel("Density")
                plt.ylabel("Flow rate")
                plt.figtext(0.1, 0.005, f'Max velocity = {self.Vehicle.max_velocity}, Slow Prob = {self.Vehicle.slow_prob}', fontsize= 9, color = 'black')

                if save:
                    self.output_dir = os.path.join(folder, f'Flow Density {number}.png')
                    plt.savefig(self.output_dir)
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
       
debug = True

if debug:
    steps = 20
    seeds = 100
    random.seed(seeds)
    sim = Simulation()
    sim.Vehicle = Vehicle(max_velocity=5, slow_prob=0.1)
    sim.Road = Road(length=10, density=50/100, number_of_lanes = 2)
    sim.initialize()
    #sim.add_obstacle(start_time=20, end_time=50, position=100, length=1)
    #sim.update(steps)
    #sim.plot_timespace(steps, plot_obstacle= True)
    #sim.avg_velocity_plot(time_start = 0, time_stop = 100, position = 75, position_range = 25)
    #sim.flow_rate_loop(steps)
    #sim.plot_contour(steps)
    #sim.plot_velocity(steps)
    #sim.plot_density(steps)




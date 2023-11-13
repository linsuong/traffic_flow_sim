import numpy as np
import matplotlib.pyplot as plt
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
        #TODO: make a lane switch fucntion that allows it to turn/switch lanes, lane switching also needs to check for headway
        # this can be implemented with a recklessness prbability maybe(?)
        pass
    
    def obstacle(self):
        #TODO: collision function that simulates accident/obstacle that cars will lane switch to manouver around this obstacle
        #obstacle present in a certain number of time steps, then after the time step is removed, how long will it take for the flow to be recovered?
        pass


    #TODO: capture speed at a certain point. there is a backward wave that makes cars move at a certain speed. can we label a car and measure it's velocity? 
    # yes - there is VechicleID output, just add velocity output for the vehicle.

class Traffic_Light:
    def __init__ (self, color):
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
    def __init__(self, start_time, end_time, position, length):
        self.start_time = start_time
        self.end_time = end_time
        self.position = position
        self.length = length

class Road:
    def __init__(self, length=100, density=1 / 100, speed_limit=2, bend=False):
        self.length = length
        self.density = density
        self.number = int(density * length)
        self.speed_limit = speed_limit
        self.bend = bend
        self.obstacle = None

    def has_obstacle(self, position, time_step):
        return (
            self.obstacle is not None
            and self.obstacle.start_time <= time_step < self.obstacle.end_time
            and self.obstacle.position <= position < (self.obstacle.position + self.obstacle.length)
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
        if self.Road.density > 1.0:
            raise ValueError("Density cannot be greater than 1.0")
    
        num_vehicles = int(self.Road.length * self.Road.density)

        if num_vehicles > self.Road.length:
            raise ValueError("Density is too high relative to road length")
        
        else:
            #print(self.Road.length)
            self.positions = random.choices(range(self.Road.length), k=num_vehicles)
            self.positions.sort()
            self.velocities = [random.randint(1, self.Vehicle.max_velocity) for i in range(len(self.positions))]
            
            if np.shape(self.velocities) != np.shape(self.positions):
                print("Number of cars: %s" %np.shape(self.positions))
                print("Number of velocities: %s" %np.shape(self.velocities))
                raise Exception("Error - not all cars have velocity, or too many velocities, not enough cars")

            #else:
                #print("Vehicles initialised successfully... starting simulation.")

        return num_vehicles

    def add_obstacle(self, start_time, end_time, position, length):
        self.Road.obstacle = Obstacle(start_time, end_time, position, length)

    def update(self, steps):
        if self.velocities is None or self.positions is None:
            raise Exception("Please call initialize() before update()")
        else:
            for step in range(steps):
                new_velocities = []

                for i in range(len(self.positions)):
                    velocity = self.velocities[i]
                    headway = (self.positions[(i + 1) %
                                            len(self.positions)] - self.positions[i] - 1) % self.Road.length

                    if self.Road.has_obstacle(self.positions[i], step):
                        velocity = min(velocity + 1, headway - 1)

                    else:
                        velocity = min(velocity + 1, self.Vehicle.max_velocity)
                        velocity = min(velocity, max(headway - 1, 0))

                    if velocity > 0 and random.random() < self.Vehicle.slow_prob:
                        velocity = max(velocity - 1, 0)

                    new_velocities.append(velocity)

                self.velocity_data.append(new_velocities)
                self.velocities = new_velocities
                new_positions = [(pos + vel) %
                                self.Road.length for pos, vel in zip(self.positions, new_velocities)]
                self.positions = new_positions
                self.data.append(new_positions[:])

        return self.data, self.velocity_data

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
    
    def flow_rate_loop(self, time_interval):
        """
        flow rate loop counter

        Args:
            time_interval (int): time step

        Returns:
            flow_rate(float): flow rate at the end of the road
        """
        num_passes = 0
        print(self.Vehicle.max_velocity)

        for k in range(self.Road.number):
            positions = [entry[k] for entry in self.data]
            previous_position = positions[-1]
            #TODO: grab velocity of car that passes through flow points - add a way to collect avgerages at different times
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
        
        flow_rate = (num_passes / time_interval)
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
        
        if plot:
            print('Simulation Complete. Plotting graph...')
            time_steps = range(steps)

            for i in range(self.Road.number):
                new_data = [item[i] for item in self.data]

                plt.plot(new_data, time_steps, '.', markersize=0.5, color='grey')

                if self.Road.obstacle is not None and plot_obstacle == True:
                    obstacle_range = np.arange(self.Road.obstacle.position, self.Road.obstacle.position + self.Road.obstacle.length, 1)
                    obstacle_time_range = np.arange(self.Road.obstacle.start_time, self.Road.obstacle.end_time, 1)

                    obstacle_positions = [pos for time in obstacle_time_range for pos in obstacle_range]

                    for obstacle_pos, obstacle_time in zip(obstacle_positions, obstacle_time_range):
                        if obstacle_pos < len(new_data) and obstacle_pos >= 0:
                            plt.plot(obstacle_pos, obstacle_time, 'rx', markersize=5)

            plt.gca().xaxis.set_ticks_position('top')
            plt.gca().invert_yaxis()
            plt.title('Time Space diagram')
            plt.xlabel('Vehicle Position')
            plt.ylabel('Time')
            plt.figtext(0.1, 0.005, f'Density = {self.Road.density}, Slow Prob = {self.Vehicle.slow_prob}, Max velocity = {self.Vehicle.max_velocity}', fontsize=9, color='black')

            if save:
                self.output_dir = os.path.join(
                    folder, f'Time Space Plot {number}.png')
                fig = plt.gcf()
                fig.set_size_inches(12, 12)
                plt.savefig(self.output_dir, dpi=100)
                plt.clf()

            else:
                plt.show()
        
    def plot_velocity(self, steps, plot=True, save=False, folder=None):

        if plot:
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
                plt.figtext(0.1, 0.005, f'Density = {self.Road.density}, Slow Prob = {self.Vehicle.slow_prob}, Max velocity = {self.Vehicle.max_velocity}', fontsize=9, color='black')
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

debug = True

if debug:
    steps = 1000
    seeds = 100
    random.seed(seeds)
    sim = Simulation()
    sim.Vehicle = Vehicle(max_velocity=10, slow_prob=0.5)
    sim.Road = Road(length=1000, density=30/1000)
    sim.initialize()
    sim.add_obstacle(start_time=200, end_time=600, position=500, length=10)
    sim.update(steps)
    sim.flow_rate_loop(steps)
    sim.plot_timespace(steps, plot_obstacle= True)
    sim.plot_velocity(steps)
    sim.plot_density(steps)




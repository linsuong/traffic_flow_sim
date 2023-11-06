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
        
class Road: #do i NEED this road class, or can this just be in the simulation clasS? 
    def __init__(self, length = 100, density = 1/100, speed_limit = 2, bend = False):
        self.length = length
        self.density = density
        self.number = int(density * length)
        self.speed_limit = speed_limit 
        self.bend = bend
        
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
            self.positions = random.sample(range(self.Road.length), int(self.Road.length * self.Road.density)) 
            self.positions.sort()
            self.velocities = [random.randint(0, self.Vehicle.max_velocity) for i in range(len(self.positions))]
            
            if np.shape(self.velocities) != np.shape(self.positions):
                print("Number of cars: %s" %np.shape(self.positions))
                print("Number of velocities: %s" %np.shape(self.velocities))
                raise Exception("Error - not all cars have velocity, or too many velocities, not enough cars")

            #else:
                #print("Vehicles initialised successfully... starting simulation.")

        return num_vehicles

    def update(self, steps):
        if self.velocities is None or self.positions is None:
            raise Exception("Please call initialize() before update()")
        
        else:
            for _ in range(steps):
                new_velocities = []

                for i in range(len(self.positions)):
                    velocity = self.velocities[i]
                    headway = (self.positions[(i + 1) % len(self.positions)] - self.positions[i] - 1) % self.Road.length

                    if self.Vehicle.kindness == True:
                        kindness = 1 + np.random.random()
                        headway = headway * kindness

                    if self.Vehicle.reckless == True:
                        reckless = np.random.random()
                        headway = headway * reckless

                    velocity = min(velocity + 1, self.Vehicle.max_velocity)
                    velocity = min(velocity, headway)

                    if velocity > 0 and random.random() < self.Vehicle.slow_prob:
                        velocity = max(velocity - 1, 0)

                    new_velocities.append(velocity)

                self.velocities = new_velocities
                new_positions = [(pos + vel) % self.Road.length for pos, vel in zip(self.positions, self.velocities)]
                self.positions = new_positions
                self.data.append(self.positions[:])

        print(self.data)

        return self.data

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

        print("Total Loops: %f" % total_loops)
        print("Num Vehicles Passed: %d" % num_vehicles_passed)
        
        flow_rate = (num_vehicles_passed / len(self.data)) * (1 / time_interval)
        print('Flow rate = %f' % flow_rate)

        return flow_rate
    
    def flow_rate_loop(self, time_interval):
        num_passes = 0
        print(self.Vehicle.max_velocity)

        for k in range(self.Road.number):
            positions = [entry[k] for entry in self.data]
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
        
        flow_rate = (num_passes / time_interval)
        print('Flow rate = %f' % flow_rate)

        return flow_rate

            
    def plot_timespace(self, steps, plot = True, save = False, folder = None, number = None):
        if plot == True:
            print('Simulation Complete. Plotting graph...')
            time_steps = range(steps)
            new_data = []
        
            for i in range(self.Road.number):
                #print(self.data[i][1])
                print('Vehicle ID: %s' %i)
                new_data = [item[i] for item in self.data]
                print('Position List: %s' %new_data)
                plt.plot(new_data, time_steps, '-')

            plt.title('Time Space diagram')
            plt.xlabel('Vechicle Position')
            plt.ylabel('Time')
            plt.figtext(0.1, 0.005, f'Density = {self.Road.density}', fontsize= 9, color='black')

            if save:
                self.output_dir = os.path.join(folder, f'Time Space Plot {number}.png')
                plt.savefig(self.output_dir)
                plt.clf()

            else:
                plt.show()

        else:
            print('Simulation Complete. Set plot = True to see the plot.')
        
    def plot_density(self, steps, plot=True, isAvg = True, save = False, folder = None, number = None):
        densities = []
        flow_rates = []
        flow_rate_avgs = []
        
        if plot:
            if isAvg:
                for i in range(1, 199):
                    p = 0.005*i
                    print('density: %f' % p)
                    densities.append(p)

                    for j in range(30):  
                        sim = Simulation()
                        sim.Road = Road(density= p)                                  
                        sim.initialize()
                        sim.update(steps)
                        #print(sim.flow_rate(steps))
                        flow_rates.append(sim.flow_rate_loop(steps))       

                    flow_rate_avg = sum(flow_rates)/30
                    flow_rate_avgs.append(flow_rate_avg)
                    print(flow_rate_avgs)

                #print("Density:" + str(densities))
                #print(np.shape(densities))
                #print("Flow Rate:" + str(flow_rate_avg))
                #print(np.shape(flow_rate_avg))
                plt.plot(densities, flow_rate_avgs, linestyle = '.')
                plt.title('Average Flow Density Relationship')
                plt.xlabel("Density")
                plt.ylabel("Flow rate")
                plt.figtext(0.1, 0.005, f'Max velocity = {self.Vehicle.max_velocity}', fontsize= 9, color='black')

                if save:
                    self.output_dir = os.path.join(folder, f'Flow Density {number}.png')
                    plt.savefig(self.output_dir)
                    plt.clf()

                else:
                    plt.show()

debug = False

if debug:
    steps = 100
    seeds = 100
    random.seed(seeds)
    sim = Simulation()
    sim.Vehicle = Vehicle(max_velocity = 5, slow_prob = 0.5, kindness = True)
    sim.Road = Road(length= 1000, density=10/1000)
    sim.initialize()
    sim.update(steps)
    sim.flow_rate_loop(steps)
    sim.plot_timespace(steps)
    sim.plot_density(steps, isAvg = True)


steps = 1000
seeds = 100
counter_1 = 0
counter_2 = 0

random.seed(seeds)
sim = Simulation()

for density_range in range(10, 100, 10):
    counter_1 = counter_1 + 1
    print("counter =", counter_1)
    folder = r"C:\Users\linus\Documents\traffic simulation plots\densities"

    if not os.path.exists(folder):
        os.makedirs(folder)

    sim.Vehicle = Vehicle(max_velocity = 5, slow_prob = 0.5, kindness = True)
    sim.Road = Road(length= 1000, density = density_range/1000)
    sim.initialize()
    sim.update(steps)
    sim.flow_rate_loop(steps)
    sim.plot_timespace(steps, save = True, folder = folder, number = counter_1)
    description = os.path.join(r"C:\Users\linus\Documents\traffic simulation plots", "config.txt")

with open(description, "w") as description_file:
    description_file.write("configuration(seed, max_velocity, slow_prob, kindess/recklness/none, length, steps)")
    description_file.write(f"{seeds}, 5, 0.5, kindness = True, 1000, {steps}")


for velocity_range in range(10, 100, 10):
    counter_2 = counter_2 + 1
    folder = os.path.join(r"C:\Users\linus\Documents\traffic simulation plots\max velocities", f"plot set {velocity_range}")

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    sim.Vehicle = Vehicle(max_velocity = velocity_range, slow_prob = 0.5, kindness = True)
    sim.Road = Road(length= 1000, density=100/1000)
    sim.initialize()
    sim.update(steps)
    sim.flow_rate_loop(steps)
    sim.plot_timespace(steps, save = True, folder = folder, number = counter_2)
    sim.plot_density(steps, isAvg = True, save = True, folder = folder, number = counter_2)
    description = os.path.join(r"C:\Users\linus\Documents\traffic simulation plots", "config.txt")

    with open(description, "w") as description_file:
        description_file.write("configuration(max_velocity, slow_prob, kindess/recklness/none, length, density, steps)")
        description_file.write(f"{seeds}, {velocity_range}, 0.5, kindness = True, 1000, 100/1000, {steps}")
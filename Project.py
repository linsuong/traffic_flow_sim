import numpy as np
import matplotlib.pyplot as plt
import random

class Vehicle:
    def __init__ (self, length = 1, target_velocity = 4, max_velocity = 10, 
                    acceleration_time = 0.5, deceleration_time = 0.5, slow_prob = 0.8):
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
    def __init__(self, length = 100, density = 5/100, speed_limit = 5, bend = False):
        self.length = length
        self.density = density
        self.number = density * length
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

    def connect_roads(self, roads):
        pass
class Simulation:
    def __init__(self):
        self.Road = Road()
        self.Vehicle = Vehicle()
        self.velocities = None
        self.positions = None
        self.data = []

        return

    def initialize(self):
        #print(self.Road.length)
        self.positions = random.sample(range(self.Road.length), int(self.Road.length * self.Road.density)) 
        self.positions.sort()
        self.velocities = [random.randint(0, self.Vehicle.max_velocity) for i in range(len(self.positions))]
        
        if np.shape(self.velocities) != np.shape(self.positions):
            print("Number of cars: %s" %np.shape(self.positions))
            print("Number of velocities: %s" %np.shape(self.velocities))
            raise Exception("Error - not all cars have velocity, or too many velocities, not enough cars")

        else:
            print("Vehicles initialised successfully... starting simulation.")

    def update(self, steps):
        if self.velocities is None or self.positions is None:
            raise Exception("Please call initialize() before update()")
        
        else:
            for _ in range(steps):
                for i in range(len(self.positions)):
                    velocity = self.velocities[i]
                    headway = (self.positions[(i + 1) % len(self.positions)] - self.positions[i] - 1) % self.Road.length
                    velocity = min(velocity + 1, self.Vehicle.max_velocity)
                    velocity = min(velocity, headway)

                    if velocity > 0 and random.random() < self.Vehicle.slow_prob:
                        velocity = max(velocity - 1, 0)

                    self.velocities[i] = velocity
                    print(np.shape(self.positions))

                new_positions = [(pos + vel) % self.Road.length for pos, vel in zip(self.positions, self.velocities)]
                self.positions = new_positions
                self.data.append(self.positions[:])

        print(self.data)
        #print(self.positions)
        time_steps = range(steps)
        new_data = []
        self.Road.number = int(self.Road.number)
    
        for i in range(self.Road.number):
            #print(self.data[i][1])
            print('Vehicle ID: %s' %i)
            new_data = [item[i] for item in self.data]
            print('Position List: %s' %new_data)
            plt.plot(new_data, time_steps, '.')

        plt.title('')
        plt.xlabel('Vechicle Position')
        plt.ylabel('Time')
        plt.show()


    def plot_density(self):
            for i in range(1500,2000):
                avg_velocity += (np.sum(self.Vehicle.position[i,1,:])/self.Road.length * self.Road.density)
            avg_velocity = avg_velocity/500
            velocities = np.append(velocities,avg_velocity)
            type(velocities)
            densities = np.linspace(0.02,1,num=49,endpoint=False)
            plt.plot(densities, np.multiply(densities, velocities))
            plt.xlabel('Density')
            plt.ylabel('Flow Rate')
            plt.show()

            return None
        
        #TODO: define plot fucntion for density and plot visualisaton of cars on road.

random.seed(100)
sim = Simulation()
sim.initialize()
print("Simulation.initalize called.")
sim.update(100)
print("Simulation.update called.")

flow_rate = []
densities = []
'''
for i in range(1, 20):
        p = 0.05*i
        densities.append(p) #density
        Road(density= p) 
        sim.initialize()
        sim.update(100)#simulation
        p_all_flow_rates = [] #store the flow rates of each simulation for one density
    
        for j in range(50): #repeat simulation each density for 20 times
            for i in range(time_step): #time step = 200
                sim.update()
            p_all_flow_rates.append(T.flow_count/time_step)

    #average flow rate for one density
        p_flow_rate = sum(p_all_flow_rates)/50.0
        flow_rate.append(p_flow_rate)

plt.plot(densities,flow_rate)
plt.xlabel("Density")
plt.ylabel("Flow rate")
'''
import numpy as np
import matplotlib.pyplot as plt
import random

class Vehicle:
    def __init__ (self, length = 1, target_velocity = 10, max_velocity = 50, acceleration_time = 0.5, deceleration_time = 0.5, slow_prob = 0.25):
        #self.id = id
        self.length = length
        self.velocity = []
        #self.current_velocity = current_velocity
        self.target_velocity = target_velocity 
        self.max_velocity = max_velocity
        self.position = []
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

class Traffic_Light:
    def __init__ (self, color):
        self.color = ['red', 'green'] 

    def status(self, color):
        color = random.choice(self.color)
        print(color)

        if color == 'red':
            Vehicle.target_velocity = 0
            Vehicle.acceleration(self)
            Vehicle.stop(self, True)

            #TODO = set timer for red light to turn into green light
    
        if color == 'green':
            #TODO = set timer for green to turn into red - can this be done compactly?
            pass
        
class Road: 
    def __init__(self, length = 100, density = 0.5, speed_limit = 5, bend = False):
        self.length = length
        self.density = density #number of cars = density * length of road
        self.speed_limit = speed_limit 
        self.bend = bend
'''
    def bend(angle, entrance_length, exit_length):
        if self.bend = True:
            angle = 

            #TODO: Add "bend" fucntion to Road class to simulate bends

'''

class Network:
    def __init__(self) -> None:
        self.roads = []
        self.connect = {}

    def connect_roads(self, roads):
        pass
class Simulation:
    def __init__(self):
        self.Road = Road()
        self.Vehicle = Vehicle()
        return

    def initialize(self):
        print(self.Road.length)
        self.Vehicle.position = random.sample(range(self.Road.length), int(self.Road.length * self.Road.density))
        self.Vehicle.position.sort()
        self.Vehicle.velocity = [random.randint(0, self.Road.speed_limit) for i in range(len(self.Vehicle.position))] 

        return self.Vehicle.velocity

    def update(self, steps):
        self.data = []
        for _ in range(steps):
            # Update vehicle velocities
            for i in range(len(self.Vehicle.position)):
                velocity = self.Vehicle.velocity[i]
                headway = (self.Vehicle.position[(i + 1) % len(self.Vehicle.position)] - self.Vehicle.position[i] - 1) % self.Road.length
                print(headway)
                velocity = min(velocity + 1, self.Vehicle.max_velocity)
                velocity = min(velocity, headway)

                if velocity > 0 and random.random() < self.Vehicle.slow_prob:
                    velocity = max(velocity - 1, 0)

                self.Vehicle.velocity[i] = velocity

            # Update vehicle positions
            new_positions = [(pos + vel) % self.Road.length for pos, vel in zip(self.Vehicle.position, self.Vehicle.velocity)]
            self.Vehicle.position = new_positions
            self.data.append(self.Vehicle.position[:])

#TODO: define plot fucntion for density and plot visualisaton of cars on road.

random.seed(100)
sim = Simulation()
sim.initialize()
sim.update(100)

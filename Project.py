import numpy as np
import matplotlib.pyplot as plt
import random

class Vehicle:
    def __init__ (self, id, length, initial_velocity, current_velocity, target_velocity, position, max_velocity, acceleration_time, deceleration_time):
        self.id = id
        self.length = length
        self.initial_velocity = []
        self.current_velocity = current_velocity
        self.target_velocity = target_velocity 
        self.max_velocity = max_velocity
        self.position = []
        self.acceleration_time = acceleration_time
        self.deceleration_time = deceleration_time

    def accelerate(self):
        factor = (self.current_velocity - self.target_velocity)/self.acceleration_time
        final_velocity = self.current_velocity * factor

        return final_velocity
        
    def decelerate(self):
        factor = (self.current_velocity - self.target_velocity)/self.deceleration_time
        final_velocity = -1 * self.current_velocity * factor

        return final_velocity
    
        
class Traffic_Light:
    def __init__ (self, status):
        self.status = status

    def status(color):
        if color = 'red':
            velocity = 0

        if color = 'green'
            pass

        if color = 'amber'
            Vehicle.decelerate()
        
class Road: 
    def __init__(self, length, density, speed_limit):
        self.length = length
        self.density = density #number of cars = density * length of road
        self.speed_limit = speed_limit


class Simulation:
    def __init__(self, runtime):
        self.runtime = runtime


    def initialize(self):
        Vehicle.position = random.sample(range(Road.length), int(Road.length * Road.density))
        Vehicle.position.sort()
        Vehicle.initial_velocity = [random.randint(0, Road.speed_limit) for i in range(len(Vehicle.position))]

    def update(self, steps):
        self.data = []
        for _ in range(steps):
            # Update vehicle velocities
            for i in range(len(self.vehicle_positions)):
                v = self.vehicle_velocities[i]
                distance_to_next = (self.vehicle_positions[(i + 1) % len(self.vehicle_positions)]
                                    - self.vehicle_positions[i] - 1) % self.road_length
                v = min(v + 1, self.max_velocity)
                v = min(v, distance_to_next)

                if v > 0 and random.random() < self.slow_prob:
                    v = max(v - 1, 0)

                self.vehicle_velocities[i] = v

            # Update vehicle positions
            new_positions = [(pos + vel) % self.road_length for pos, vel in zip(self.vehicle_positions, self.vehicle_velocities)]
            self.vehicle_positions = new_positions
            self.data.append(self.vehicle_positions[:])        
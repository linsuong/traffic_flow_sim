import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

start_time = time.time()
print("Runtime = " %(time.time() - start_time))


class Driver:
    def __init__(self, probability):
        self.probability = 0.25 #probability that the driver slows down

class Vehicle:
    def __init__ (self, car_length, velocity, max_velocity, headway):
        self.car_length = car_length
        self.velocity = velocity
        self.headway = headway 

    def acceleration():
        if v < m and 


class Road:
    def __init__(self, length, number_of_cars, density):
        self.length = length
        self.number_of_cars = number_of_cars
        self.density = number_of_cars/length
        self.vehicle_count = 0 #counter for cars enter and exit
    
    def road():
        road = np.zeros(len(number_of_cars))
        return road

    def bend(angle, length):
        self.angle = angle
        self.bend = bend

    def update(self):
        m = Vehicle.max_velocity
        next_state = -sp.
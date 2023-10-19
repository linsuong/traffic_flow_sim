import numpy as np
import matplotlib.pyplot as plt
import random

#global variables
v_max = 60
road_length = 500
vehicle_length = 1 #n. of cells a vehicle occupies
p = 0.8 #probablility vehicle slows down
density = 0.3
steps = 2000 
velocities = np.zeros(0,dtype=float)
densities = np.linspace(0.02,1,num=49,endpoint=False)

def move(number, road_length, position, velocity):
    road = np.zeros((2, road_length), dtype = int)
    for i in range(number):
        count = int(position[i])
        road[0, count] = 1
        road[1, count] = velocity[i] 
    return road

for d in densities:
    avg_velocity = 0
    number = int(road_length*density)
    velocity = np.zeros((number), dtype= int) #store car velocity
    position = np.zeros((number), dtype= int) #store car positon
    print(position)
    data = np.zeros((0, 2, road_length), dtype = int) #store data of runtime

#initialization
position = np.array(random.sample(range(road_length), number)) #generate cars and assign random positions on the road
#print(position)
position.sort() #sorting positions
#print(position)
velocity = np.random.randint(v_max + 1, size = number)
dataset = move(number, road_length, position, velocity)
data = np.append(data, [dataset], axis = 0)

for i in range(steps):
    velocity = np.min(np.vstack((velocity + 1, v_max*np.ones_like(velocity))),axis=0)
    print(velocity)





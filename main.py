import numpy as np
import matplotlib.pyplot as plt
import random
import os
import Vechicle
import Traffic_Lights
import Network
import Simulation as sim

steps = 100
seeds = 100

random.seed(seeds)
sim = Simulation()
sim.Vehicle = Vehicle(max_velocity = 5, slow_prob = 0.5)
sim.Road = Road(length= 100, density=2/100)
sim.initialize()
sim.update(steps)
sim.flow_rate_loop(steps)
sim.plot_timespace(steps)
sim.plot_velocity(steps)
sim.plot_density(steps, isAvg = True)
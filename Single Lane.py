import numpy as np
import matplotlib.pyplot as plt
import random

class NaSchTrafficSimulation:
    def __init__(self, road_length=200, density=0.7, max_velocity=5, slow_prob=0.5):
        self.road_length = road_length
        self.density = density
        self.max_velocity = max_velocity
        self.slow_prob = slow_prob
        self.data = []  # Stores the state of the road at each time step
        self.vehicle_positions = []

    def initialize(self):
        self.vehicle_positions = random.sample(range(self.road_length), int(self.road_length * self.density))
        self.vehicle_positions.sort()

        # Initialize vehicle velocities randomly
        self.vehicle_velocities = [random.randint(0, self.max_velocity) for _ in range(len(self.vehicle_positions))]
        print(self.vehicle_velocities)

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

    def plot_vehicle_positions_over_time(self):
        plt.figure(figsize=(10, 6))
        for i in range(len(self.data)):
            positions = self.data[i]
            time_steps = [i] * len(positions)
            plt.plot(positions, time_steps, '.', 'black',markersize=5)
        plt.title('Vehicle Positions Over Time')
        plt.xlabel('Position on Road')
        plt.ylabel('Time Step')
        plt.show()

# Main simulation
random.seed(100)
sim = NaSchTrafficSimulation(density=0.3)
sim.initialize()
sim.update(30)
sim.plot_vehicle_positions_over_time()

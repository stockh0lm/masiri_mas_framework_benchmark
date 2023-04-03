import sys
import time
import random
import math
import numpy as np
from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from scipy.stats import norm
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt

from queue import Queue, Empty
from resource import getrusage, RUSAGE_SELF
from threading import Thread

T_SLICE = 5 # minutes
FILENAME = './benchmark_results_python.csv'
FIBONACCI = 10

def memory_monitor(command_queue: Queue, poll_interval=1):
    old_max = 0
    while True:
        try:
            command_queue.get(timeout=poll_interval)
            #print(f'Stopping memory monitor\n\nmax RSS {max_rss/2**10:9.3f} MiB\n\n')
            command_queue.put(max_rss / 2 ** 10)
            return
        except Empty:
            max_rss = getrusage(RUSAGE_SELF).ru_maxrss
            if max_rss > old_max:
                old_max = max_rss


class User(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.trip_duration = 0

    def step(self):
        time_of_day = (self.model.clock - 0.5) * T_SLICE / 60

        if self.model.is_active(time_of_day):
            trip_duration = random.uniform(0.5, 2)
            self.trip_duration = trip_duration
            battery_capacity = 60
            charging_power = 50 / 60
            charging_time = self.model.calculate_charging_time(trip_duration, battery_capacity, charging_power)

            self.model.total_energy_used += trip_duration * battery_capacity / charging_time
            self.model.total_energy_generated += charging_time * charging_power


class House(Agent):
    def __init__(self, unique_id, model, size, solar_power, heat_pump, air_conditioning, electric_car):
        super().__init__(unique_id, model)
        self.size = size
        self.solar_power = solar_power
        self.heat_pump = heat_pump
        self.air_conditioning = air_conditioning
        self.electric_car = electric_car

    def step(self):
        day_of_year = (self.model.clock - 1) // (24 * 60 // T_SLICE) + 1
        time_of_day = (self.model.clock - 0.5) * T_SLICE / 60
        temperature = self.model.temperature_data[day_of_year - 1, self.model.clock - 1]
        sunshine_intensity = self.model.sunshine_data[day_of_year - 1, self.model.clock - 1]

        generated_energy = self.model.energy_generation(self, sunshine_intensity)
        consumed_energy = self.model.energy_consumption(self, temperature)

        if self.electric_car:
            if self.model.car_on_trip[self.unique_id]:
                car_energy = 0
            else:
                battery_capacity = 60
                charging_power = 50 / 60
                car_energy = min(generated_energy - consumed_energy, charging_power)
                car_energy = max(car_energy, -battery_capacity)
        else:
            car_energy = 0

        net_energy = generated_energy - consumed_energy + car_energy
        self.model.total_energy_generated += max(0, net_energy)
        self.model.total_energy_used += max(0, -net_energy)


class EnergyModel(Model):
    def __init__(self, n_users=100, n_houses=100):
        self.num_agents = n_users + n_houses
        self.schedule = SimultaneousActivation(self)

        self.temperature_data, self.sunshine_data = self.generate_temperature_and_sunshine_data()

        for i in range(n_users):
            a = User(i, self)
            self.schedule.add(a)

        for i in range(n_houses):
            size, solar_power, heat_pump, air_conditioning, electric_car = self.house_properties()
            a = House(n_users + i, self, size, solar_power, heat_pump, air_conditioning, electric_car)
            self.schedule.add(a)

        self.total_energy_used = 0
        self.total_energy_generated = 0
        self.clock = 1
        self.car_on_trip = {i: False for i in range(n_users, n_users + n_houses)}
        self.intervals_per_day = 24 * 60 // T_SLICE
        self.datacollector = DataCollector(
            model_reporters={"Total_energy_used": lambda m: m.get_total_energy_used(),
                             "Total_energy_generated": lambda m: m.get_total_energy_generated(),
                             }
        )
        self.running = True


    def step(self):
        self.datacollector.collect(self)  # Collect data before taking a step
        self.clock = ((self.schedule.time - 1) % self.intervals_per_day) + 1
        clock = self.clock
        self.schedule.step()

    def get_total_energy_used(self):
        return self.total_energy_used

    def get_total_energy_generated(self):
        return self.total_energy_generated


    @staticmethod
    def is_active(time_of_day):
        mean_active_time = 14  # Mean of the normal distribution at 14:00 (14 hours)
        std_dev = 3  # Standard deviation of 3 hours
        probability = norm(mean_active_time, std_dev).pdf(time_of_day)
        return random.random() < probability

    @staticmethod
    def calculate_charging_time(trip_duration, battery_capacity, charging_power):
        energy_used = battery_capacity * trip_duration
        charging_time = energy_used / charging_power
        return charging_time

    @staticmethod
    def house_properties():
        size = random.uniform(0.5, 2)
        solar_power = random.random() < 0.7
        heat_pump = random.random() < 0.5
        air_conditioning = random.random() < 0.5
        electric_car = random.random() < 0.5
        return size, solar_power, heat_pump, air_conditioning, electric_car

    @staticmethod
    def energy_generation(house, sunshine_intensity):
        solar_power_capacity = 10 * house.size
        return sunshine_intensity * solar_power_capacity if house.solar_power else 0

    @staticmethod
    def energy_consumption(house, temperature):
        heat_pump_consumption = max(0, 18 - temperature) * house.size if house.heat_pump else 0
        air_conditioning_consumption = max(0, temperature - 24) * house.size if house.air_conditioning else 0
        return heat_pump_consumption + air_conditioning_consumption

    @staticmethod
    def luebeck_temperature(day_of_year, time_of_day):
        daily_avg_temp = 2 * math.sin(2 * math.pi * (day_of_year - 31) / 365) + 10
        daily_temp_variation = 5 * math.sin(2 * math.pi * time_of_day / 24)
        noise = random.gauss(0, 1)
        return daily_avg_temp + daily_temp_variation + noise

    @staticmethod
    def luebeck_sunshine(day_of_year, time_of_day):
        max_daylight_hours = 17
        min_daylight_hours = 7
        daylight_hours = min_daylight_hours + (max_daylight_hours - min_daylight_hours) / 2 * (
                1 + math.sin(2 * math.pi * (day_of_year - 172) / 365))
        sunrise = 12 - daylight_hours / 2
        sunset = 12 + daylight_hours / 2
        if sunrise <= time_of_day <= sunset:
            sunshine_intensity = math.sin(math.pi * (time_of_day - sunrise) / daylight_hours)
            noise = random.uniform(-0.1, 0.1)
            return sunshine_intensity + noise
        else:
            return 0

    def generate_temperature_and_sunshine_data(self):
        n_days = 365
        intervals_per_day = 24 * 60 // T_SLICE

        temperature_data = np.empty((n_days, intervals_per_day))
        sunshine_data = np.empty((n_days, intervals_per_day))

        for day_of_year in range(n_days):
            for interval_weather in range(intervals_per_day):
                time_of_day = (interval_weather - 0.5) * T_SLICE / 60
                temperature_data[day_of_year, interval_weather] = self.luebeck_temperature(day_of_year + 1, time_of_day)
                sunshine_data[day_of_year, interval_weather] = self.luebeck_sunshine(day_of_year + 1, time_of_day)

        return temperature_data, sunshine_data


def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

def append_numbers_to_file(filename, numbers):
    with open(filename, 'a') as f:
        f.write(','.join(map(str, numbers)) + '\n')



def main(num_agents):

    queue = Queue()
    poll_interval = 0.1
    monitor_thread = Thread(target=memory_monitor, args=(queue, poll_interval))
    monitor_thread.start()

    model = EnergyModel(n_users=num_agents, n_houses=num_agents)
    n_days = 365
    intervals_per_day = 24 * 60 // T_SLICE

    start_time = time.time()
    for day in range(n_days):
#        print(f"Day: {day + 1}")
        for interval in range(intervals_per_day):
            model.step()
#        print(f"Net energy: {model.total_energy_generated - model.total_energy_used:.2f} kWh")
#        print("=" * 50)
        todays_energy_used = model.get_total_energy_used()
        todays_energy_generated = model.get_total_energy_generated()

    sim_duration = time.time() - start_time

    queue.put('stop')
    monitor_thread.join()
    rss = queue.get()
    numbers = [num_agents, rss, sim_duration]
    append_numbers_to_file(FILENAME, numbers)

    # extract the data from the data collector
    # model_data = model.datacollector.get_model_vars_dataframe()
    # plt.figure(figsize=(10, 5))
    #
    #
    #
    # model_data = model.datacollector.get_model_vars_dataframe()
    # plt.figure(figsize=(10, 5))
    # en_used = model_data["Total_energy_used"].to_numpy()
    # en_gen = model_data["Total_energy_generated"].to_numpy()
    # # plot the derived data
    # en_gen_diff = np.diff(en_gen)
    # en_used_diff = np.diff(en_used)
    # plt.plot(en_gen_diff, label="Total Energy Generated")
    # plt.plot(en_used_diff, label="Total Energy Used")
    # #plt.plot(model_data["Total_energy_used"], label="Total Energy Used")
    # #plt.plot(model_data["Total_energy_generated"], label="Total Energy Generated")
    # plt.xlabel("Time step")
    # plt.ylabel("Energy")
    # plt.legend()
    # plt.show()
    # # plot temperature and sunshine over the year
    # temperature_data, sunshine_data = model.temperature_data, model.sunshine_data
    # plt.figure(figsize=(10, 5))
    # #plt.plot(temperature_data.flatten(), label="Temperature")
    # plt.plot(sunshine_data.flatten(), label="Sunshine")
    # plt.xlabel("Time step")
    # plt.ylabel("Temperature / Sunshine")
    # plt.legend()
    # plt.show()




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <num_agents>")
        sys.exit(1)

    try:
        num_agents = int(sys.argv[1])
    except ValueError:
        print("Error: num_agents should be an integer.")
        sys.exit(1)

    main(num_agents)

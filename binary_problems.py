from mlrose_hiive.fitness import four_peaks
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose_hiive
from mlrose_hiive import fitness
import numpy as np
import time

def analyze_knapsack():
    def knapsack(state):
        weights = [4, 5, 1, 7, 9, 2, 6, 10, 3, 2]
        values = [9, 7, 4, 12, 21, 7, 9, 18, 4, 2]

        max_weight = 20

        weight = 0
        value = 0
        for i in range(len(state)):
            weight += state[i]*weights[i]
            value += state[i]*values[i]

            if (weight>max_weight):
                return 0

        return value

    knapsack_fitness = mlrose_hiive.CustomFitness(knapsack, 'discrete')
    knapsack_problem = mlrose_hiive.DiscreteOpt(length = 10, fitness_fn = knapsack_fitness, maximize = True, max_val = 2)

    start_time = time.time()
    best_state, best_fitness, _ = mlrose_hiive.random_hill_climb(knapsack_problem, init_state=np.array([0,0,0,0,0,0,0,0,0,0]), random_state=12)
    runtime = time.time() - start_time
    print("Best hill climbing values and score for knapsack problem: ")
    print(best_state, best_fitness)
    print("Runtime: {:.4f}".format(runtime))

    start_time = time.time()
    schedule = mlrose_hiive.ExpDecay()
    best_state, best_fitness, _ = mlrose_hiive.simulated_annealing(knapsack_problem, schedule=schedule, random_state=12)
    runtime = time.time() - start_time
    print("Best simulated annealing values and score for knapsack problem: ")
    print(best_state, best_fitness)
    print("Runtime: {:.4f}".format(runtime))

    start_time = time.time()
    best_state, best_fitness, _ = mlrose_hiive.genetic_alg(knapsack_problem, pop_size = 10, random_state=12)
    runtime = time.time() - start_time
    print("Best genetic algorithm values and score for knapsack problem: ")
    print(best_state, best_fitness)
    print("Runtime: {:.4f}".format(runtime))

    start_time = time.time()
    best_state, best_fitness, _ = mlrose_hiive.mimic(knapsack_problem, pop_size = 10, random_state=12)
    runtime = time.time() - start_time
    print("Best mimic values and score for knapsack problem: ")
    print(best_state, best_fitness)
    print("Runtime: {:.4f}".format(runtime))

def analyze_four_peaks():
    four_peaks_problem = mlrose_hiive.DiscreteOpt(length=50, fitness_fn=mlrose_hiive.FourPeaks(0.2), maximize = True, max_val = 2)

    start_time = time.time()
    best_state, best_fitness, _ = mlrose_hiive.random_hill_climb(four_peaks_problem, random_state=12)
    runtime = time.time() - start_time
    print("Best hill climbing values and score for four peaks problem: ")
    print(best_state, best_fitness)
    print("Runtime: {:.4f}".format(runtime))

    schedule = mlrose_hiive.ExpDecay()
    start_time = time.time()
    best_state, best_fitness, _ = mlrose_hiive.simulated_annealing(four_peaks_problem, schedule=schedule, random_state=12)
    runtime = time.time() - start_time
    print("Best simulated annealing values and score for four peaks problem: ")
    print(best_state, best_fitness)
    print("Runtime: {:.4f}".format(runtime))

    start_time = time.time()
    best_state, best_fitness, _ = mlrose_hiive.genetic_alg(four_peaks_problem, random_state=12)
    runtime = time.time() - start_time
    print("Best genetic algorithm values and score for four peaks problem: ")
    print(best_state, best_fitness)
    print("Runtime: {:.4f}".format(runtime))

    start_time = time.time()
    best_state, best_fitness, _ = mlrose_hiive.mimic(four_peaks_problem, random_state=12)
    runtime = time.time() - start_time
    print("Best mimic values and score for four peaks problem: ")
    print(best_state, best_fitness)
    print("Runtime: {:.4f}".format(runtime))


def analyze_flip_flop():
    flip_flop_fitness = mlrose_hiive.FlipFlop()
    flip_flop_problem = mlrose_hiive.DiscreteOpt(length=50, fitness_fn=flip_flop_fitness, maximize = True, max_val = 2)

    start_time = time.time()
    best_state, best_fitness, _ = mlrose_hiive.random_hill_climb(flip_flop_problem, random_state=12)
    runtime = time.time() - start_time
    print("Best hill climbing values and score for flip flop problem: ")
    print(best_state, best_fitness)
    print("Runtime: {:.4f}".format(runtime))

    schedule = mlrose_hiive.ArithDecay()
    start_time = time.time()
    best_state, best_fitness, _ = mlrose_hiive.simulated_annealing(flip_flop_problem, schedule=schedule, random_state=12)
    runtime = time.time() - start_time
    print("Best simulated annealing values and score for flip flop problem: ")
    print(best_state, best_fitness)
    print("Runtime: {:.4f}".format(runtime))

    start_time = time.time()
    best_state, best_fitness, _ = mlrose_hiive.genetic_alg(flip_flop_problem, random_state=12)
    runtime = time.time() - start_time
    print("Best genetic algorithm values and score for flip flop problem: ")
    print(best_state, best_fitness)
    print("Runtime: {:.4f}".format(runtime))

    start_time = time.time()
    best_state, best_fitness, _ = mlrose_hiive.mimic(flip_flop_problem, random_state=12)
    runtime = time.time() - start_time
    print("Best mimic values and score for flip flop problem: ")
    print(best_state, best_fitness)
    print("Runtime: {:.4f}".format(runtime))

if (__name__ == '__main__'):
    analyze_knapsack()
    analyze_four_peaks()
    analyze_flip_flop()
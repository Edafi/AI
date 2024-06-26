from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import numpy

import matplotlib.pyplot as plt
import seaborn as sns

import backpack

backpack = backpack.Backpack()
POPULATION_SIZE = 100
P_CROSSOVER = 0.75  
P_MUTATION = 0.1   
MAX_GENERATIONS = 100
HALL_OF_FAME_SIZE = 1

toolbox = base.Toolbox()
toolbox.register("func_randomZeroOne", random.randint, 0, 1)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("func_individualCreator", tools.initRepeat, creator.Individual, toolbox.func_randomZeroOne, len(backpack))
toolbox.register("func_populationCreator", tools.initRepeat, list, toolbox.func_individualCreator)

def BackpackValue(individual):
    return backpack.getValue(individual),  # return a tuple


toolbox.register("evaluate", BackpackValue)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/len(backpack))

def main():
    population = toolbox.func_populationCreator(n=POPULATION_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)
    best = hof.items[0]
    print("Best Ever Individual - ", best)
    print("Best Ever Fitness - ", best.fitness.values[0])

    print("Backpack Items: ")
    backpack.printItems(best)
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()


if __name__ == "__main__":
    main()
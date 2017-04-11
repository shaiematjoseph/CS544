from random import *
import inspyred
from math import sin, sqrt
from time import time


def generate_schwefel(random, args):
    size = args.get('num_inputs', 2)
    data = [random.uniform(-500, 500) for i in range(size)]
    #print data
    return data


def evaluate_schwefel(candidates, args):
    fitness = []
    for cs in candidates:
        fit = 418.9829 * len(cs) - sum([-1 * x * sin(sqrt(abs(x))) for x in cs])
        fitness.append(fit)
    return fitness

rand = Random()
rand.seed(int(time()))
my_ec = inspyred.ec.EvolutionaryComputation(rand)
my_ec.selector = inspyred.ec.selectors.tournament_selection
my_ec.variator = [inspyred.ec.variators.uniform_crossover, inspyred.ec.variators.gaussian_mutation]
my_ec.replacer = inspyred.ec.replacers.generational_replacement
my_ec.terminator = inspyred.ec.terminators.evaluation_termination
final_pop = my_ec.evolve(generator=generate_schwefel,
                         evaluator=evaluate_schwefel,
                         pop_size=100,
                         maximize=False,
                         bounder=inspyred.ec.Bounder(-500, 500),
                         num_selected=100,
                         tournament_size=2,
                         num_elites=1,
                         num_inputs=2,
                         mutation_rate=0.25,
                         max_evaluations=20000)
final_pop.sort(reverse=True)
print final_pop[0]

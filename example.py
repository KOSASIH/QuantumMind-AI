import random

# Define the TSP problem
cities = {
    'A': (0, 0),
    'B': (1, 5),
    'C': (5, 2),
    'D': (3, 6),
    'E': (8, 3)
}
num_cities = len(cities)

# Define the fitness function
def tsp_fitness(solution):
    total_distance = 0
    for i in range(num_cities - 1):
        city1 = solution[i]
        city2 = solution[i + 1]
        distance = ((cities[city2][0] - cities[city1][0])**2 + (cities[city2][1] - cities[city1][1])**2)**0.5
        total_distance += distance
    return -total_distance  # Negative value for maximization

# Initialize the population
population_size = 50
population = []
for _ in range(population_size):
    solution = list(cities.keys())
    random.shuffle(solution)
    population.append(solution)

# QIGA algorithm
num_generations = 100
for generation in range(num_generations):
    # Evaluation
    fitness_scores = [tsp_fitness(solution) for solution in population]

    # Selection
    selected_indices = random.choices(range(population_size), weights=fitness_scores, k=population_size)

    # Quantum-inspired crossover
    offspring = []
    for i in range(0, population_size, 2):
        parent1 = population[selected_indices[i]]
        parent2 = population[selected_indices[i + 1]]
        crossover_point = random.randint(1, num_cities - 1)
        child1 = parent1[:crossover_point] + [city for city in parent2 if city not in parent1[:crossover_point]]
        child2 = parent2[:crossover_point] + [city for city in parent1 if city not in parent2[:crossover_point]]
        offspring.extend([child1, child2])

    # Quantum-inspired mutation
    for i in range(population_size):
        solution = offspring[i]
        mutation_point1 = random.randint(0, num_cities - 1)
        mutation_point2 = random.randint(0, num_cities - 1)
        solution[mutation_point1], solution[mutation_point2] = solution[mutation_point2], solution[mutation_point1]

    # Replace the old population with the new generation
    population = offspring

# Find the best solution
best_solution = max(population, key=tsp_fitness)
best_fitness = tsp_fitness(best_solution)

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)

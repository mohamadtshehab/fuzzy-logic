import numpy as np
import random
from typing import List, Tuple, Dict, Optional
import pandas as pd
from dataclasses import dataclass
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@dataclass
class City:
    id: int
    x: float
    y: float
    
    def distance_to(self, other: 'City') -> float:
        
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class TSPGeneticAlgorithm:    
    def __init__(self, cities: List[City], population_size: int = 50, 
                 mutation_rate: float = 0.1, tournament_size: int = 3):
        
        self.cities = cities
        self.num_cities = len(cities)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        
        
        self.distance_matrix = self._create_distance_matrix()
        
        
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.generation_history = []
        
    def _create_distance_matrix(self) -> np.ndarray:
        
        matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    matrix[i][j] = self.cities[i].distance_to(self.cities[j])
        return matrix
    
    def create_chromosome(self) -> List[int]:
        return random.sample(range(self.num_cities), self.num_cities)
    
    def calculate_fitness(self, chromosome: List[int]) -> float:
        total_distance = 0
        for i in range(len(chromosome)):
            current_city = chromosome[i]
            next_city = chromosome[(i + 1) % len(chromosome)]
            total_distance += self.distance_matrix[current_city][next_city]
        return total_distance
    
    def tournament_selection(self, population: List[List[int]], 
                           fitness_values: List[float]) -> List[int]:
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        
        
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        size = len(parent1)
        
        
        point1, point2 = sorted(random.sample(range(size), 2))
        
        
        offspring1 = [-1] * size
        offspring2 = [-1] * size
        
        
        offspring1[point1:point2] = parent1[point1:point2]
        offspring2[point1:point2] = parent2[point1:point2]
        
        
        for offspring, parent in [(offspring1, parent2), (offspring2, parent1)]:
            remaining = [x for x in parent if x not in offspring[point1:point2]]
            j = 0
            for i in range(size):
                if offspring[i] == -1:
                    offspring[i] = remaining[j]
                    j += 1
        
        return offspring1, offspring2
    
    def swap_mutation(self, chromosome: List[int]) -> List[int]:
        if random.random() < self.mutation_rate:
            
            pos1, pos2 = random.sample(range(len(chromosome)), 2)
            chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]
        return chromosome
    
    def create_initial_population(self) -> List[List[int]]:
        return [self.create_chromosome() for _ in range(self.population_size)]
    
    def evaluate_population(self, population: List[List[int]]) -> Tuple[List[float], int]:
        
        fitness_values = [self.calculate_fitness(chrom) for chrom in population]
        best_idx = np.argmin(fitness_values)  
        return fitness_values, best_idx
    
    def evolve(self, generations: int, verbose: bool = True) -> Dict:
        
        
        population = self.create_initial_population()
        fitness_values, best_idx = self.evaluate_population(population)
        
        best_fitness = fitness_values[best_idx]
        best_chromosome = population[best_idx].copy()
        
        
        self.best_fitness_history = [best_fitness]
        self.avg_fitness_history = [np.mean(fitness_values)]
        self.generation_history = [0]
        
        if verbose:
            print(f"Initial best fitness: {best_fitness:.2f}")
        
        for generation in range(1, generations + 1):
            
            new_population = []
            
            new_population.append(population[best_idx].copy())
            
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_values)
                parent2 = self.tournament_selection(population, fitness_values)
                
                offspring1, offspring2 = self.order_crossover(parent1, parent2)
                
                offspring1 = self.swap_mutation(offspring1)
                offspring2 = self.swap_mutation(offspring2)
                
                new_population.extend([offspring1, offspring2])
            
            new_population = new_population[:self.population_size]
            
            population = new_population
            fitness_values, best_idx = self.evaluate_population(population)
            
            if fitness_values[best_idx] < best_fitness:
                best_fitness = fitness_values[best_idx]
                best_chromosome = population[best_idx].copy()
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitness_values))
            self.generation_history.append(generation)
            
            if verbose and generation % 10 == 0:
                print(f"Generation {generation}: Best = {best_fitness:.2f}, Avg = {np.mean(fitness_values):.2f}")
        
        return {
            'best_chromosome': best_chromosome,
            'best_fitness': best_fitness,
            'best_route': [self.cities[i] for i in best_chromosome],
            'generations': generations,
            'final_population': population,
            'final_fitness_values': fitness_values
        }
        
    def plot_evolution(self, save_path: Optional[str] = "src/genetic_algorithm/evolution_plot.png"):
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.generation_history, self.best_fitness_history, 'b-', label='Best Fitness')
        plt.plot(self.generation_history, self.avg_fitness_history, 'r--', label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Distance)')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.generation_history, self.avg_fitness_history, 'g-', label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Average Fitness')
        plt.title('Average Fitness Trend')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Evolution plot saved to {save_path}")

    def plot_best_route(self, route: List[City], save_path: Optional[str] = "src/genetic_algorithm/best_route_plot.png"):
        
        plt.figure(figsize=(10, 8))
        
        x_coords = [city.x for city in route] + [route[0].x]
        y_coords = [city.y for city in route] + [route[0].y]
        
        plt.plot(x_coords, y_coords, 'b-o', linewidth=2, markersize=8, label='Route')
        for i, city in enumerate(route):
            plt.annotate(f'City {city.id}', (city.x, city.y), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Best TSP Route (Distance: {self.calculate_fitness([c.id for c in route]):.2f})')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Best route plot saved to {save_path}")
    
    def generate_test_cities(self, num_cities: int = 10) -> List[City]:
        
        cities = []
        for i in range(num_cities):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            cities.append(City(i, x, y))
        return cities


def run_tsp_example(no_plots: bool = False):
    
    print("=== TSP Genetic Algorithm Example ===\n")
    
    ga = TSPGeneticAlgorithm([])
    cities = ga.generate_test_cities(10)
    
    print(f"Generated {len(cities)} test cities:")
    for city in cities:
        print(f"City {city.id}: ({city.x:.1f}, {city.y:.1f})")
    
    ga = TSPGeneticAlgorithm(cities, population_size=50, mutation_rate=0.1)
    
    print(f"\nGA Parameters:")
    print(f"- Population Size: {ga.population_size}")
    print(f"- Mutation Rate: {ga.mutation_rate}")
    print(f"- Tournament Size: {ga.tournament_size}")
    
    print(f"\nStarting evolution for 50 generations...")
    start_time = time.time()
    results = ga.evolve(generations=50, verbose=True)
    end_time = time.time()
    
    print(f"\nEvolution completed in {end_time - start_time:.2f} seconds")
    print(f"Best fitness found: {results['best_fitness']:.2f}")
    print(f"Best route: {[c.id for c in results['best_route']]}")

    if not no_plots:
        try:
            ga.plot_evolution()
            ga.plot_best_route(results['best_route'])
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
    
    return ga, results

if __name__ == "__main__":
    run_tsp_example() 
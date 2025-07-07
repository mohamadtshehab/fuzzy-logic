"""
Genetic Algorithm for Traveling Salesman Problem (TSP)
====================================================

This module implements a complete Genetic Algorithm system to solve the TSP,
which is one of the most frequent and practical optimization problems.

Problem Statement:
- Given a set of cities and distances between them
- Find the shortest possible route that visits each city exactly once
- Return to the starting city
- Goal: Minimize total travel distance
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Dict, Optional
import pandas as pd
from dataclasses import dataclass
import time


@dataclass
class City:
    """Represents a city with coordinates."""
    id: int
    x: float
    y: float
    
    def distance_to(self, other: 'City') -> float:
        """Calculate Euclidean distance to another city."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class TSPGeneticAlgorithm:
    """
    Genetic Algorithm implementation for Traveling Salesman Problem.
    
    This class implements a complete GA system with:
    - Permutation-based chromosome representation
    - Tournament selection
    - Order Crossover (OX)
    - Swap mutation
    - Fitness function (total route distance)
    - Evolutionary loop with progress tracking
    """
    
    def __init__(self, cities: List[City], population_size: int = 50, 
                 mutation_rate: float = 0.1, tournament_size: int = 3):
        """
        Initialize the GA for TSP.
        
        Args:
            cities: List of City objects
            population_size: Size of the population
            mutation_rate: Probability of mutation
            tournament_size: Size of tournament for selection
        """
        self.cities = cities
        self.num_cities = len(cities)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        
        # Distance matrix for faster calculations
        self.distance_matrix = self._create_distance_matrix()
        
        # Evolution tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.generation_history = []
        
    def _create_distance_matrix(self) -> np.ndarray:
        """Create a distance matrix for all city pairs."""
        matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    matrix[i][j] = self.cities[i].distance_to(self.cities[j])
        return matrix
    
    def create_chromosome(self) -> List[int]:
        """
        Create a random chromosome (permutation of cities).
        
        Returns:
            List of city IDs representing a route
        """
        return random.sample(range(self.num_cities), self.num_cities)
    
    def calculate_fitness(self, chromosome: List[int]) -> float:
        """
        Calculate fitness (total route distance).
        
        Args:
            chromosome: List of city IDs representing a route
            
        Returns:
            Total distance of the route (lower is better)
        """
        total_distance = 0
        for i in range(len(chromosome)):
            current_city = chromosome[i]
            next_city = chromosome[(i + 1) % len(chromosome)]
            total_distance += self.distance_matrix[current_city][next_city]
        return total_distance
    
    def tournament_selection(self, population: List[List[int]], 
                           fitness_values: List[float]) -> List[int]:
        """
        Perform tournament selection to choose a parent.
        
        Args:
            population: List of chromosomes
            fitness_values: List of fitness values
            
        Returns:
            Selected chromosome
        """
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        
        # Select the best from tournament (lowest distance for TSP)
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Perform Order Crossover (OX) for permutation chromosomes.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of two offspring chromosomes
        """
        size = len(parent1)
        
        # Choose two crossover points
        point1, point2 = sorted(random.sample(range(size), 2))
        
        # Create offspring
        offspring1 = [-1] * size
        offspring2 = [-1] * size
        
        # Copy the segment between crossover points
        offspring1[point1:point2] = parent1[point1:point2]
        offspring2[point1:point2] = parent2[point1:point2]
        
        # Fill the remaining positions
        for offspring, parent in [(offspring1, parent2), (offspring2, parent1)]:
            remaining = [x for x in parent if x not in offspring[point1:point2]]
            j = 0
            for i in range(size):
                if offspring[i] == -1:
                    offspring[i] = remaining[j]
                    j += 1
        
        return offspring1, offspring2
    
    def swap_mutation(self, chromosome: List[int]) -> List[int]:
        """
        Perform swap mutation on a chromosome.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        if random.random() < self.mutation_rate:
            # Choose two random positions to swap
            pos1, pos2 = random.sample(range(len(chromosome)), 2)
            chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]
        return chromosome
    
    def create_initial_population(self) -> List[List[int]]:
        """
        Create the initial population.
        
        Returns:
            List of chromosomes
        """
        return [self.create_chromosome() for _ in range(self.population_size)]
    
    def evaluate_population(self, population: List[List[int]]) -> Tuple[List[float], int]:
        """
        Evaluate the fitness of all chromosomes in the population.
        
        Args:
            population: List of chromosomes
            
        Returns:
            Tuple of (fitness_values, best_chromosome_index)
        """
        fitness_values = [self.calculate_fitness(chrom) for chrom in population]
        best_idx = np.argmin(fitness_values)  # Lower is better for TSP
        return fitness_values, best_idx
    
    def evolve(self, generations: int, verbose: bool = True) -> Dict:
        """
        Run the genetic algorithm for the specified number of generations.
        
        Args:
            generations: Number of generations to evolve
            verbose: Whether to print progress
            
        Returns:
            Dictionary with results and statistics
        """
        # Initialize population
        population = self.create_initial_population()
        fitness_values, best_idx = self.evaluate_population(population)
        
        best_fitness = fitness_values[best_idx]
        best_chromosome = population[best_idx].copy()
        
        # Track progress
        self.best_fitness_history = [best_fitness]
        self.avg_fitness_history = [np.mean(fitness_values)]
        self.generation_history = [0]
        
        if verbose:
            print(f"Initial best fitness: {best_fitness:.2f}")
        
        for generation in range(1, generations + 1):
            # Create new population
            new_population = []
            
            # Elitism: keep the best individual
            new_population.append(population[best_idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitness_values)
                parent2 = self.tournament_selection(population, fitness_values)
                
                # Crossover
                offspring1, offspring2 = self.order_crossover(parent1, parent2)
                
                # Mutation
                offspring1 = self.swap_mutation(offspring1)
                offspring2 = self.swap_mutation(offspring2)
                
                new_population.extend([offspring1, offspring2])
            
            # Ensure population size is correct
            new_population = new_population[:self.population_size]
            
            # Update population and evaluate
            population = new_population
            fitness_values, best_idx = self.evaluate_population(population)
            
            # Update best solution
            if fitness_values[best_idx] < best_fitness:
                best_fitness = fitness_values[best_idx]
                best_chromosome = population[best_idx].copy()
            
            # Track progress
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
    
    def plot_evolution(self, save_path: Optional[str] = None):
        """Plot the evolution of fitness over generations."""
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
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_best_route(self, route: List[City], save_path: Optional[str] = None):
        """Plot the best route found."""
        plt.figure(figsize=(10, 8))
        
        # Plot cities
        x_coords = [city.x for city in route]
        y_coords = [city.y for city in route]
        
        # Add the starting city to complete the route
        x_coords.append(route[0].x)
        y_coords.append(route[0].y)
        
        # Plot the route
        plt.plot(x_coords, y_coords, 'b-o', linewidth=2, markersize=8, label='Route')
        
        # Plot city labels
        for i, city in enumerate(route):
            plt.annotate(f'City {city.id}', (city.x, city.y), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Best TSP Route (Distance: {self.calculate_fitness([c.id for c in route]):.2f})')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_test_cities(self, num_cities: int = 10) -> List[City]:
        """Generate random test cities."""
        cities = []
        for i in range(num_cities):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            cities.append(City(i, x, y))
        return cities


def run_tsp_example():
    """Run a complete example of the TSP Genetic Algorithm."""
    print("=== TSP Genetic Algorithm Example ===\n")
    
    # Generate test cities
    ga = TSPGeneticAlgorithm([])  # Temporary instance to access method
    cities = ga.generate_test_cities(10)
    
    print(f"Generated {len(cities)} test cities:")
    for city in cities:
        print(f"City {city.id}: ({city.x:.1f}, {city.y:.1f})")
    
    # Create GA instance
    ga = TSPGeneticAlgorithm(cities, population_size=50, mutation_rate=0.1)
    
    print(f"\nGA Parameters:")
    print(f"- Population Size: {ga.population_size}")
    print(f"- Mutation Rate: {ga.mutation_rate}")
    print(f"- Tournament Size: {ga.tournament_size}")
    
    # Run evolution
    print(f"\nStarting evolution for 50 generations...")
    start_time = time.time()
    results = ga.evolve(generations=50, verbose=True)
    end_time = time.time()
    
    print(f"\nEvolution completed in {end_time - start_time:.2f} seconds")
    print(f"Best fitness found: {results['best_fitness']:.2f}")
    print(f"Best route: {[c.id for c in results['best_route']]}")
    
    # Plot results
    ga.plot_evolution()
    ga.plot_best_route(results['best_route'])
    
    return ga, results


if __name__ == "__main__":
    run_tsp_example() 
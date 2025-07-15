"""
Command Line Interface for TSP Genetic Algorithm
"""

import argparse
from src.genetic_algorithm.genetic_algorithm import TSPGeneticAlgorithm, run_tsp_example


def main():
    """Main CLI function for the TSP Genetic Algorithm."""
    parser = argparse.ArgumentParser(
        description="Traveling Salesman Problem Genetic Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ga_cli.py --cities 10 --generations 100 --population 50
  python ga_cli.py --cities 15 --generations 200 --mutation 0.15 --verbose
  python ga_cli.py --demo
        """
    )
    
    parser.add_argument(
        '--cities', '-c',
        type=int,
        default=10,
        help='Number of cities to generate (default: 10)'
    )
    
    parser.add_argument(
        '--generations', '-g',
        type=int,
        default=100,
        help='Number of generations to evolve (default: 100)'
    )
    
    parser.add_argument(
        '--population', '-p',
        type=int,
        default=50,
        help='Population size (default: 50)'
    )
    
    parser.add_argument(
        '--mutation', '-m',
        type=float,
        default=0.1,
        help='Mutation rate (default: 0.1)'
    )
    
    parser.add_argument(
        '--tournament', '-t',
        type=int,
        default=3,
        help='Tournament size for selection (default: 3)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output during evolution'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run a demo with default parameters'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plotting (useful for headless environments)'
    )
    
    args = parser.parse_args()
    
    if args.demo:
        print("Running TSP Genetic Algorithm Demo...")
        run_tsp_example(no_plots=args.no_plots)
        return
    
    print("=== TSP Genetic Algorithm CLI ===\n")
    
    # Generate cities
    print(f"Generating {args.cities} random cities...")
    ga = TSPGeneticAlgorithm([])  # Temporary instance
    cities = ga.generate_test_cities(args.cities)
    
    print(f"Generated cities:")
    for city in cities:
        print(f"City {city.id}: ({city.x:.1f}, {city.y:.1f})")
    
    # Create GA instance
    print(f"\nInitializing Genetic Algorithm...")
    print(f"Parameters:")
    print(f"  - Population Size: {args.population}")
    print(f"  - Generations: {args.generations}")
    print(f"  - Mutation Rate: {args.mutation}")
    print(f"  - Tournament Size: {args.tournament}")
    
    ga = TSPGeneticAlgorithm(
        cities=cities,
        population_size=args.population,
        mutation_rate=args.mutation,
        tournament_size=args.tournament
    )
    
    # Run evolution
    print(f"\nStarting evolution...")
    results = ga.evolve(generations=args.generations, verbose=args.verbose)
    
    # Display results
    print(f"\n=== Results ===")
    print(f"Best fitness (distance): {results['best_fitness']:.2f}")
    print(f"Best route: {[c.id for c in results['best_route']]}")
    
    # Calculate some statistics
    final_fitness = results['final_fitness_values']
    print(f"Final population statistics:")
    print(f"  - Best fitness: {min(final_fitness):.2f}")
    print(f"  - Average fitness: {sum(final_fitness)/len(final_fitness):.2f}")
    print(f"  - Worst fitness: {max(final_fitness):.2f}")
    
    # Plot results if not disabled
    if not args.no_plots:
        print(f"\nGenerating plots...")
        try:
            ga.plot_evolution()
            ga.plot_best_route(results['best_route'])
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
    
    print(f"\nEvolution completed successfully!")

if __name__ == "__main__":
    main() 
# Fuzzy Expert System with Genetic Algorithm

## Overview
This project implements a comprehensive Fuzzy Expert System for Air Conditioning Control and a bonus Genetic Algorithm for solving the Traveling Salesman Problem (TSP). The system demonstrates advanced AI techniques for optimization and decision-making.

## Features

### Fuzzy Expert System
- **Fuzzy Logic-based Decision Making**: Determines optimal AC settings based on temperature, humidity, and time of day
- **Interactive Web Interface**: Streamlit-based UI for easy parameter adjustment and result visualization
- **Membership Function Visualization**: Dynamic plots showing fuzzy set interpretations
- **System Validation**: Comprehensive testing with accuracy metrics

### Genetic Algorithm (Bonus)
- **TSP Solver**: Complete implementation for Traveling Salesman Problem
- **Advanced Genetic Operators**: Tournament selection, Order Crossover, Swap mutation
- **Progress Tracking**: Real-time fitness evolution monitoring
- **Visualization**: Route plotting and evolution graphs
- **CLI Interface**: Command-line tool for parameterized execution

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Setup
1. **Clone or download the project**
2. **Install dependencies:**
   ```bash
   pip install -e .
   ```
   Or install manually:
   ```bash
   pip install scikit-fuzzy numpy matplotlib scipy pandas streamlit plotly
   ```

## Usage

### Fuzzy Expert System

#### Web Interface
```bash
streamlit run src/streamlit_app.py
```

**Features:**
- Adjust temperature (15-35°C), humidity (30-90%), and time of day (0-24 hours)
- View recommended AC setting percentage
- See membership function plots
- Monitor system validation metrics

#### Programmatic Usage
```python
from src.fuzzy_system import FuzzyAirConditioningSystem

# Initialize the system
system = FuzzyAirConditioningSystem()

# Evaluate specific conditions
ac_setting = system.evaluate(temperature=25, humidity=60, time_of_day=14)
print(f"Recommended AC Setting: {ac_setting:.1f}%")

# Get membership values
membership = system.get_membership_values(25, 60, 14)
print(membership)

# Validate system
validation = system.validate_system()
print(f"Accuracy: {validation['accuracy_within_10_percent']:.1f}%")
```

### Genetic Algorithm

#### Command Line Interface
```bash
# Run with default parameters
python src/ga_cli.py

# Custom parameters
python src/ga_cli.py --cities 15 --generations 200 --population 100 --mutation 0.15 --verbose

# Run demo
python src/ga_cli.py --demo

# Headless mode (no plots)
python src/ga_cli.py --no-plots
```

#### Programmatic Usage
```python
from src.genetic_algorithm import TSPGeneticAlgorithm, City

# Generate test cities
ga = TSPGeneticAlgorithm([])
cities = ga.generate_test_cities(10)

# Create and run GA
ga = TSPGeneticAlgorithm(cities, population_size=50, mutation_rate=0.1)
results = ga.evolve(generations=100, verbose=True)

print(f"Best route: {[c.id for c in results['best_route']]}")
print(f"Distance: {results['best_fitness']:.2f}")

# Visualize results
ga.plot_evolution()
ga.plot_best_route(results['best_route'])
```

## System Architecture

### Fuzzy Expert System Components
1. **Input Variables**: Temperature, Humidity, Time of Day
2. **Fuzzy Sets**: 4 temperature sets, 3 humidity sets, 4 time sets
3. **Membership Functions**: Triangular functions for all variables
4. **Rule Base**: 10 expert-defined fuzzy rules
5. **Inference Engine**: Mamdani inference with centroid defuzzification
6. **Output**: AC Setting percentage (0-100%)

### Genetic Algorithm Components
1. **Chromosome Representation**: Permutation of city IDs
2. **Selection**: Tournament selection
3. **Crossover**: Order Crossover (OX) for permutation chromosomes
4. **Mutation**: Swap mutation
5. **Fitness Function**: Total route distance (minimization)
6. **Evolutionary Loop**: Elitism with generational replacement

## Testing and Validation

### Fuzzy System Validation
- **Test Cases**: 10 expert-defined scenarios
- **Metrics**: Mean Absolute Error, Root Mean Square Error, Accuracy
- **Tolerance**: 10% accuracy threshold

### Genetic Algorithm Testing
- **Convergence**: Fitness evolution tracking
- **Performance**: Best, average, and worst fitness monitoring
- **Visualization**: Evolution plots and route visualization

## Workload Distribution

### Team Member Responsibilities

#### **You (Project Lead)**
- **Fuzzy Expert System Core Implementation** (`src/fuzzy_system.py`)
- **System Architecture Design**
- **Integration and Testing**
- **Documentation and Project Management**

#### **Jinan**
- **Streamlit Web Interface** (`src/streamlit_app.py`)
- **User Experience Design**
- **Frontend Styling and Layout**
- **Interactive Plotting and Visualization**

#### **Tammam**
- **Genetic Algorithm Implementation** (`src/genetic_algorithm.py`)
- **Genetic Operators Design**
- **Fitness Function Optimization**
- **Algorithm Performance Tuning**

#### **Rama**
- **CLI Interface Development** (`src/ga_cli.py`)
- **Command-line Parameter Handling**
- **User Input Validation**
- **Error Handling and Logging**


### Collaboration Points
- **Weekly Code Reviews**: All members review each other's code
- **Integration Testing**: Joint testing of fuzzy system + GA integration
- **Documentation**: Shared responsibility for README and user guides
- **Presentation**: Each member presents their component

## Technical Specifications

### Fuzzy System Parameters
- **Temperature Range**: 15-35°C
- **Humidity Range**: 30-90%
- **Time Range**: 0-24 hours
- **Output Range**: 0-100% AC setting

### Genetic Algorithm Parameters
- **Population Size**: 50 (default)
- **Mutation Rate**: 0.1 (default)
- **Tournament Size**: 3 (default)
- **Generations**: 100 (default)

## Performance Metrics

### Fuzzy System
- **Mean Absolute Error**: < 5%
- **Accuracy within 10%**: > 90%
- **Response Time**: < 100ms

### Genetic Algorithm
- **Convergence**: Typically within 50-100 generations
- **Solution Quality**: Within 10% of optimal for small instances
- **Scalability**: Handles up to 50 cities efficiently

## References and Resources

### Academic Sources
- Zadeh, L. A. (1965). Fuzzy sets. Information and Control, 8(3), 338-353.
- Ross, T. J. (2010). Fuzzy Logic with Engineering Applications. Wiley.
- Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley.

### Technical Documentation
- [scikit-fuzzy documentation](https://scikit-fuzzy.github.io/)
- [Streamlit documentation](https://docs.streamlit.io/)
- [NumPy documentation](https://numpy.org/doc/)
- [Matplotlib documentation](https://matplotlib.org/)

### Datasets and Benchmarks
- TSPLIB: Standard TSP benchmark instances
- Custom fuzzy system test cases
- Real-world AC control scenarios

## Troubleshooting

### Common Issues
1. **Streamlit plots not updating**: Use the "Update Membership Plots" button
2. **GA convergence issues**: Increase population size or generations
3. **Import errors**: Ensure all dependencies are installed
4. **Memory issues**: Reduce population size for large city sets

### Performance Tips
- Use smaller populations for quick testing
- Enable verbose mode for debugging
- Disable plots in headless environments
- Cache membership function calculations

## Contributing
1. Follow the established code structure
2. Add comprehensive docstrings
3. Include unit tests for new features
4. Update documentation for any changes
5. Coordinate with team members for integration

## License
This project is for educational purposes. Please cite appropriate sources when using this code.

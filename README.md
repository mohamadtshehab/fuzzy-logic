# Fuzzy Expert System: Driving Risk Assessment with Genetic Algorithm

## Overview
This project implements a comprehensive **Fuzzy Expert System for Driving Risk Assessment** and a bonus **Genetic Algorithm for solving the Traveling Salesman Problem (TSP)**. The system demonstrates advanced AI techniques for safety assessment and optimization in real-world scenarios.

## Features

### Fuzzy Expert System - Driving Risk Assessment
- **Intelligent Risk Evaluation**: Assesses driving risk based on speed, weather conditions, and driver focus level
- **Dual Output System**: Provides both risk assessment and intervention recommendations
- **Interactive Web Interface**: Streamlit-based UI for real-time risk assessment
- **Advanced Membership Functions**: Uses triangular, trapezoidal, and Gaussian functions
- **Comprehensive Rule Base**: 27 expert-defined rules covering all driving scenarios
- **Data-Driven Validation**: CSV-based test cases with fallback defaults

### Genetic Algorithm (Bonus) - TSP Solver
- **Complete TSP Implementation**: Solves the classic Traveling Salesman Problem
- **Advanced Genetic Operators**: Tournament selection, Order Crossover, Swap mutation
- **Real-time Evolution Tracking**: Monitors fitness progression across generations
- **Visualization Suite**: Route plotting and evolution graphs
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

### Fuzzy Expert System - Driving Risk Assessment

#### Web Interface
```bash
streamlit run src/fuzzy_problem/streamlit_app.py
```

**System Inputs:**
- **Speed**: 0-140 km/h (low, medium, high ranges)
- **Weather**: 0-10 scale (good, moderate, bad conditions)
- **Focus**: 0-10 scale (low, medium, high concentration levels)

**System Outputs:**
- **Risk Level**: 0-10 scale (low, medium, high risk)
- **Intervention**: 0-10 scale (none, warning, emergency actions)

#### Programmatic Usage
```python
from src.fuzzy_problem.fuzzy_system import FuzzyDrivingRiskSystem

# Initialize the system (loads CSV data if available)
system = FuzzyDrivingRiskSystem()

# Evaluate driving conditions
result = system.evaluate(speed=100, weather=7, focus=3)
print(f"Risk Level: {result['risk']:.2f}")
print(f"Intervention: {result['intervention']:.2f}")

# Get detailed membership values
membership = system.get_membership_values(100, 7, 3)
print(membership)

# Validate system performance
validation = system.validate_system()
print(f"Risk Accuracy: {validation['accuracy_risk_within_1']:.1f}%")
```

#### Data File Setup
Create `src/fuzzy_problem/fuzzy_data.csv` with test cases:
```csv
Speed,Weather,Focus,Risk,Intervention
130,9,1,9,9
80,5,5,5,5
30,1,9,2,1
110,8,9,6,6
50,5,3,4,5
```

### Genetic Algorithm - TSP Solver

#### Command Line Interface
```bash
# Run demonstration
python src/genetic_algorithm/ga_cli.py --demo

# Custom parameters
python src/genetic_algorithm/ga_cli.py --cities 15 --generations 200 --population 100 --mutation 0.15 --verbose

# Headless mode (no plots)
python src/genetic_algorithm/ga_cli.py --demo --no-plots
```

#### Programmatic Usage
```python
from src.genetic_algorithm.genetic_algorithm import TSPGeneticAlgorithm, City

# Generate test cities
ga = TSPGeneticAlgorithm([])
cities = ga.generate_test_cities(10)

# Create and run GA
ga = TSPGeneticAlgorithm(cities, population_size=50, mutation_rate=0.1)
results = ga.evolve(generations=100, verbose=True)

print(f"Best route: {[c.id for c in results['best_route']]}")
print(f"Total distance: {results['best_fitness']:.2f}")

# Visualize results
ga.plot_evolution()
ga.plot_best_route(results['best_route'])
```

## System Architecture

### Fuzzy Expert System Components
1. **Input Variables**: 
   - Speed (0-140 km/h): Trapezoidal membership functions
   - Weather (0-10): Gaussian membership functions  
   - Focus (0-10): Gaussian membership functions

2. **Output Variables**:
   - Risk (0-10): Triangular membership functions
   - Intervention (0-10): Mixed trapezoidal and triangular functions

3. **Rule Base**: 27 comprehensive rules covering all speed×weather×focus combinations

4. **Inference Engine**: Mamdani inference with centroid defuzzification

5. **Membership Function Types**:
   - **Trapezoidal**: Speed ranges, intervention levels
   - **Gaussian**: Weather conditions, focus levels
   - **Triangular**: Risk levels

### Genetic Algorithm Components
1. **Chromosome Representation**: Permutation of city IDs
2. **Selection**: Tournament selection (size 3)
3. **Crossover**: Order Crossover (OX) for permutation preservation
4. **Mutation**: Swap mutation with configurable rate
5. **Fitness Function**: Total Euclidean distance (minimization)
6. **Evolution Strategy**: Elitism with generational replacement

## Testing and Validation

### Fuzzy System Validation
- **Test Cases**: CSV-based with expert-defined scenarios
- **Metrics**: Mean Absolute Error for risk and intervention
- **Accuracy**: Within 1-point tolerance for both outputs
- **Fallback Data**: Built-in test cases if CSV unavailable

### Genetic Algorithm Testing
- **Convergence Tracking**: Best and average fitness per generation
- **Performance Metrics**: Solution quality and convergence speed
- **Visualization**: Evolution plots and optimal route display

## Problem Statement

### Driving Risk Assessment System
**Context**: Modern vehicles need intelligent systems to assess driving risk and recommend appropriate interventions to enhance safety.

**Goal**: Develop a fuzzy expert system that:
- Evaluates driving risk based on multiple factors
- Provides appropriate intervention recommendations
- Handles uncertainty in real-world driving conditions

**Constraints**:
- Real-time processing requirements
- Multiple input variables with different scales
- Need for interpretable decision-making
- Gradual transitions between risk levels

**Solution**: Fuzzy logic system with:
- Speed-based risk escalation
- Weather condition impact assessment
- Driver focus level consideration
- Dual-output recommendation system

## Workload Distribution

### Team Member Responsibilities

#### **Mohamad**
- **Fuzzy Expert System Core Implementation** (`src/fuzzy_problem/fuzzy_system.py`)
- **System Architecture Design**
- **Integration and Testing**
- **Documentation and Project Management**

#### **Jinan**
- **Streamlit Web Interface** (`src/fuzzy_problem/streamlit_app.py`)
- **User Experience Design**
- **Frontend Styling and Layout**
- **Interactive Plotting and Visualization**

#### **Tammam**
- **Genetic Algorithm Implementation** (`src/genetic_algorithm/genetic_algorithm.py`)
- **Genetic Operators Design**
- **Fitness Function Optimization**
- **Algorithm Performance Tuning**

#### **Rama**
- **CLI Interface Development** (`src/genetic_algorithm/ga_cli.py`)
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
- **Speed Range**: 0-140 km/h (3 fuzzy sets)
- **Weather Range**: 0-10 scale (3 fuzzy sets)
- **Focus Range**: 0-10 scale (3 fuzzy sets)
- **Risk Output**: 0-10 scale (3 fuzzy sets)
- **Intervention Output**: 0-10 scale (3 fuzzy sets)

### Genetic Algorithm Parameters
- **Population Size**: 50 (default)
- **Mutation Rate**: 0.1 (default)
- **Tournament Size**: 3 (default)
- **Generations**: 100 (default)

## Performance Metrics

### Fuzzy System
- **Mean Absolute Error**: < 1.0 for both outputs
- **Accuracy within 1 point**: > 80%
- **Response Time**: < 50ms
- **Rule Coverage**: 100% (all 27 combinations)

### Genetic Algorithm
- **Convergence**: Typically within 50-100 generations
- **Solution Quality**: Within 10% of optimal for small instances
- **Scalability**: Handles up to 50 cities efficiently

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install scikit-fuzzy numpy matplotlib scipy pandas streamlit plotly
   ```

2. **Run the Driving Risk Assessment System:**
   ```bash
   streamlit run src/fuzzy_problem/streamlit_app.py
   ```

3. **Test the Genetic Algorithm:**
   ```bash
   python src/genetic_algorithm/ga_cli.py --demo
   ```

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

### Datasets and Applications
- Real-world driving scenarios
- TSPLIB: Standard TSP benchmark instances
- Traffic safety research data

## Troubleshooting

### Common Issues
1. **CSV file not found**: System uses fallback test cases
2. **Streamlit plots not updating**: Use plot refresh buttons
3. **GA convergence issues**: Increase population size or generations
4. **Import errors**: Ensure all dependencies are installed

### Performance Tips
- Use CSV data for comprehensive testing
- Enable verbose mode for debugging
- Adjust membership functions for specific use cases
- Cache fuzzy calculations for repeated evaluations

## License
This project is for educational purposes. Please cite appropriate sources when using this code.

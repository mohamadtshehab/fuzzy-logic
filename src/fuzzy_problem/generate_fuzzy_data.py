import pandas as pd
import random
from src.fuzzy_problem.fuzzy_system import FuzzyDrivingRiskSystem  


num_samples = 100

fuzzy_system = FuzzyDrivingRiskSystem()

data = []

for _ in range(num_samples):
    speed = random.randint(0, 140)
    weather = random.uniform(0, 10)
    focus = random.uniform(0, 10)

    output = fuzzy_system.evaluate(speed, weather, focus)

    data.append((
        round(speed, 2),
        round(weather, 2),
        round(focus, 2),
        round(output['risk'], 2),
        round(output['intervention'], 2)
    ))

df = pd.DataFrame(data, columns=['Speed', 'Weather', 'Focus', 'Risk', 'Intervention'])

df.to_csv('fuzzy_data_generated.csv', index=False)
print("don... fuzzy_data_generated.csv")

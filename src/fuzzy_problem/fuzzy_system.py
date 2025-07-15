import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from typing import Dict, List, Tuple

class FuzzyDrivingRiskSystem:
    def __init__(self, csv_file_path='fuzzy_data.csv'):
       
        self.test_cases: List[Tuple[float, float, float, float, float]] = []
        try:
            df = pd.read_csv(csv_file_path)
            for _, row in df.iterrows():
                self.test_cases.append((
                    float(row['Speed']),
                    float(row['Weather']),
                    float(row['Focus']),
                    float(row['Risk']),
                    float(row['Intervention'])
                ))
        except FileNotFoundError:
            print(f"erorr file not found {csv_file_path} ")

            self.test_cases = [
                (130, 9, 1, 9, 9), (80, 5, 5, 5, 5), (30, 1, 9, 2, 1),
                (110, 8, 9, 6, 6), (50, 5, 3, 4, 5)
            ]  
            
        except Exception as e:
            print(f"read field {e}")
            self.test_cases = [
                (130, 9, 1, 9, 9), (80, 5, 5, 5, 5), (30, 1, 9, 2, 1),
                (110, 8, 9, 6, 6), (50, 5, 3, 4, 5)
            ]

        self._setup_universes()
        self._setup_membership_functions()
        self._setup_rules()
        self._create_control_system()

    def _setup_universes(self):
        self.speed = ctrl.Antecedent(np.arange(0, 141, 1), 'speed')
        self.weather = ctrl.Antecedent(np.arange(0, 11, 0.1), 'weather')
        self.focus = ctrl.Antecedent(np.arange(0, 11, 0.1), 'focus')
        self.risk = ctrl.Consequent(np.arange(0, 11, 1), 'risk')
        self.intervention = ctrl.Consequent(np.arange(0, 11, 1), 'intervention')

    def _setup_membership_functions(self):
        self.speed['low'] = fuzz.trapmf(self.speed.universe, [0, 0, 20, 50])
        self.speed['medium'] = fuzz.trimf(self.speed.universe, [30, 70, 110])
        self.speed['high'] = fuzz.trapmf(self.speed.universe, [90, 120, 140, 140])

        self.weather['good'] = fuzz.gaussmf(self.weather.universe, 0, 1.5)
        self.weather['moderate'] = fuzz.gaussmf(self.weather.universe, 5, 1.5)
        self.weather['bad'] = fuzz.gaussmf(self.weather.universe, 9, 1.5)

        self.focus['low'] = fuzz.gaussmf(self.focus.universe, 0, 1.5)
        self.focus['medium'] = fuzz.gaussmf(self.focus.universe, 5, 1.5)
        self.focus['high'] = fuzz.gaussmf(self.focus.universe, 9, 1)

        self.risk['low'] = fuzz.trimf(self.risk.universe, [0, 0, 4])
        self.risk['medium'] = fuzz.trimf(self.risk.universe, [2, 5, 8])
        self.risk['high'] = fuzz.trimf(self.risk.universe, [6, 10, 10])
        
        self.intervention['none'] = fuzz.trapmf(self.intervention.universe, [0, 0, 2, 4])
        self.intervention['warning'] = fuzz.trimf(self.intervention.universe, [3, 5, 7])
        self.intervention['emergency'] = fuzz.trapmf(self.intervention.universe, [6, 8, 10, 10])

    def _setup_rules(self):
        self.rules = [
            ctrl.Rule(self.speed['low'] & self.weather['good'] & self.focus['high'],
                      (self.risk['low'], self.intervention['none'])),
            ctrl.Rule(self.speed['low'] & self.weather['good'] & self.focus['medium'],
                      (self.risk['low'], self.intervention['none'])),
            ctrl.Rule(self.speed['low'] & self.weather['good'] & self.focus['low'],
                      (self.risk['medium'], self.intervention['warning'])),
            ctrl.Rule(self.speed['low'] & self.weather['moderate'] & self.focus['high'],
                      (self.risk['low'], self.intervention['none'])),
            ctrl.Rule(self.speed['low'] & self.weather['moderate'] & self.focus['medium'],
                      (self.risk['medium'], self.intervention['warning'])),
            ctrl.Rule(self.speed['low'] & self.weather['moderate'] & self.focus['low'],
                      (self.risk['medium'], self.intervention['warning'])),
            ctrl.Rule(self.speed['low'] & self.weather['bad'] & self.focus['high'],
                      (self.risk['medium'], self.intervention['warning'])),
            ctrl.Rule(self.speed['low'] & self.weather['bad'] & self.focus['medium'],
                      (self.risk['medium'], self.intervention['warning'])),
            ctrl.Rule(self.speed['low'] & self.weather['bad'] & self.focus['low'],
                      (self.risk['high'], self.intervention['emergency'])),
            ctrl.Rule(self.speed['medium'] & self.weather['good'] & self.focus['high'],
                      (self.risk['low'], self.intervention['none'])),
            ctrl.Rule(self.speed['medium'] & self.weather['good'] & self.focus['medium'],
                      (self.risk['medium'], self.intervention['warning'])),
            ctrl.Rule(self.speed['medium'] & self.weather['good'] & self.focus['low'],
                      (self.risk['medium'], self.intervention['warning'])),
            ctrl.Rule(self.speed['medium'] & self.weather['moderate'] & self.focus['high'],
                      (self.risk['medium'], self.intervention['warning'])),
            ctrl.Rule(self.speed['medium'] & self.weather['moderate'] & self.focus['medium'],
                      (self.risk['medium'], self.intervention['warning'])),
            ctrl.Rule(self.speed['medium'] & self.weather['moderate'] & self.focus['low'],
                      (self.risk['high'], self.intervention['emergency'])),
            ctrl.Rule(self.speed['medium'] & self.weather['bad'] & self.focus['high'],
                      (self.risk['medium'], self.intervention['warning'])),
            ctrl.Rule(self.speed['medium'] & self.weather['bad'] & self.focus['medium'],
                      (self.risk['high'], self.intervention['emergency'])),
            ctrl.Rule(self.speed['medium'] & self.weather['bad'] & self.focus['low'],
                      (self.risk['high'], self.intervention['emergency'])),
            ctrl.Rule(self.speed['high'] & self.weather['good'] & self.focus['high'],
                      (self.risk['medium'], self.intervention['warning'])),
            ctrl.Rule(self.speed['high'] & self.weather['good'] & self.focus['medium'],
                      (self.risk['medium'], self.intervention['warning'])),
            ctrl.Rule(self.speed['high'] & self.weather['good'] & self.focus['low'],
                      (self.risk['high'], self.intervention['emergency'])),
            ctrl.Rule(self.speed['high'] & self.weather['moderate'] & self.focus['high'],
                      (self.risk['medium'], self.intervention['warning'])),
            ctrl.Rule(self.speed['high'] & self.weather['moderate'] & self.focus['medium'],
                      (self.risk['high'], self.intervention['emergency'])),
            ctrl.Rule(self.speed['high'] & self.weather['moderate'] & self.focus['low'],
                      (self.risk['high'], self.intervention['emergency'])),
            ctrl.Rule(self.speed['high'] & self.weather['bad'] & self.focus['high'],
                      (self.risk['high'], self.intervention['emergency'])),
            ctrl.Rule(self.speed['high'] & self.weather['bad'] & self.focus['medium'],
                      (self.risk['high'], self.intervention['emergency'])),
            ctrl.Rule(self.speed['high'] & self.weather['bad'] & self.focus['low'],
                      (self.risk['high'], self.intervention['emergency']))
        ]

    def _create_control_system(self):
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)

    def evaluate(self, speed, weather, focus, expected_risk=None, expected_interv=None):
        simulation = ctrl.ControlSystemSimulation(self.control_system)
        simulation.input['speed'] = speed
        simulation.input['weather'] = weather
        simulation.input['focus'] = focus
        simulation.compute()

        return {
            'risk': float(simulation.output['risk']),
            'intervention': float(simulation.output['intervention'])
        }

  

    def get_membership_values(self, speed: float, weather: float, focus: float) -> Dict:
        return {
            'speed': {k: fuzz.interp_membership(self.speed.universe, self.speed[k].mf, speed) for k in self.speed.terms},
            'weather': {k: fuzz.interp_membership(self.weather.universe, self.weather[k].mf, weather) for k in self.weather.terms},
            'focus': {k: fuzz.interp_membership(self.focus.universe, self.focus[k].mf, focus) for k in self.focus.terms},
        }

    def create_membership_plot(self):
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        for ax, var, label in zip(
            [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0]],
            [self.speed, self.weather, self.focus, self.risk, self.intervention],
            ['Speed', 'Weather', 'Focus', 'Risk', 'Intervention']
        ):
            for term in var.terms:
                ax.plot(var.universe, var[term].mf, label=term)
            ax.set_title(label)
            ax.set_xlabel(label.lower())
            ax.set_ylabel('Membership')
            ax.legend()
            ax.grid(True)
        axes[2, 1].axis('off')
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close()
        return img_str

    def plot_risk_output(self, speed, weather, focus):
        self.simulation.input['speed'] = speed
        self.simulation.input['weather'] = weather
        self.simulation.input['focus'] = focus
        self.simulation.compute()
        risk_val = self.simulation.output['risk']
        fig, ax = plt.subplots(figsize=(7, 6))
        self.risk.view(sim=self.simulation, ax=ax)
        ax.set_title("Risk Output")
        ax.text(risk_val, 1.05, f"{risk_val:.2f}", ha='center', color='black', fontsize=10)
        plt.tight_layout(pad=2.0)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return img_str

    def plot_intervention_output(self, speed, weather, focus):
        self.simulation.input['speed'] = speed
        self.simulation.input['weather'] = weather
        self.simulation.input['focus'] = focus
        self.simulation.compute()
        intervention_val = self.simulation.output['intervention']
        fig, ax = plt.subplots(figsize=(7, 6))
        self.intervention.view(sim=self.simulation, ax=ax)
        ax.set_title("Intervention Output")
        ax.text(intervention_val, 1.05, f"{intervention_val:.2f}", ha='center', color='black', fontsize=10)
        plt.tight_layout(pad=2.0)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return img_str

    def validate_system(self) -> Dict:
        results = []
        for speed, weather, focus, expected_risk, expected_interv in self.test_cases:
            output = self.evaluate(speed, weather, focus)
            results.append({
                'speed': speed, 'weather': weather, 'focus': focus,
                'expected_risk': expected_risk, 'actual_risk': output['risk'],
                'expected_intervention': expected_interv, 'actual_intervention': output['intervention'],
                'risk_error': abs(expected_risk - output['risk']) if expected_risk is not None else None,
                'interv_error': abs(expected_interv - output['intervention']) if expected_interv is not None else None
            })
        df = pd.DataFrame(results)
        return {
            'mean_absolute_error_risk': df['risk_error'].mean() if df['risk_error'].notna().all() else None,
            'mean_absolute_error_intervention': df['interv_error'].mean() if df['interv_error'].notna().all() else None,
            'accuracy_risk_within_1': (df['risk_error'] <= 1).sum() / len(df) * 100 if df['risk_error'].notna().all() else 0,
            'accuracy_intervention_within_1': (df['interv_error'] <= 1).sum() / len(df) * 100 if df['interv_error'].notna().all() else 0,
            'test_cases': results
        }
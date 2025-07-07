"""
Fuzzy Expert System for Air Conditioning Control
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
import io
import base64


class FuzzyAirConditioningSystem:
    """Fuzzy Expert System for Air Conditioning Control"""
    
    def __init__(self):
        self._setup_universes()
        self._setup_membership_functions()
        self._setup_rules()
        self._create_control_system()
        
    def _setup_universes(self):
        """Define the universe of discourse for each variable."""
        self.temperature = ctrl.Antecedent(np.arange(15, 36, 1), 'temperature')
        self.humidity = ctrl.Antecedent(np.arange(30, 91, 1), 'humidity')
        self.time_of_day = ctrl.Antecedent(np.arange(0, 25, 1), 'time_of_day')
        self.ac_setting = ctrl.Consequent(np.arange(0, 101, 1), 'ac_setting')
        
    def _setup_membership_functions(self):
        """Define membership functions for all variables."""
        # Temperature membership functions
        self.temperature['cold'] = fuzz.trimf(self.temperature.universe, [15, 15, 20])
        self.temperature['cool'] = fuzz.trimf(self.temperature.universe, [18, 22, 26])
        self.temperature['warm'] = fuzz.trimf(self.temperature.universe, [24, 28, 32])
        self.temperature['hot'] = fuzz.trimf(self.temperature.universe, [30, 35, 35])
        
        # Humidity membership functions
        self.humidity['low'] = fuzz.trimf(self.humidity.universe, [30, 30, 50])
        self.humidity['medium'] = fuzz.trimf(self.humidity.universe, [45, 60, 75])
        self.humidity['high'] = fuzz.trimf(self.humidity.universe, [70, 90, 90])
        
        # Time of day membership functions
        self.time_of_day['night'] = fuzz.trimf(self.time_of_day.universe, [0, 0, 6])
        self.time_of_day['morning'] = fuzz.trimf(self.time_of_day.universe, [5, 8, 11])
        self.time_of_day['afternoon'] = fuzz.trimf(self.time_of_day.universe, [10, 14, 18])
        self.time_of_day['evening'] = fuzz.trimf(self.time_of_day.universe, [17, 20, 24])
        
        # AC Setting membership functions
        self.ac_setting['low'] = fuzz.trimf(self.ac_setting.universe, [0, 0, 30])
        self.ac_setting['medium'] = fuzz.trimf(self.ac_setting.universe, [20, 50, 80])
        self.ac_setting['high'] = fuzz.trimf(self.ac_setting.universe, [70, 100, 100])
        
    def _setup_rules(self):
        """Define the fuzzy rules for the system."""
        rule1 = ctrl.Rule(self.temperature['hot'] & self.humidity['high'], self.ac_setting['high'])
        rule2 = ctrl.Rule(self.temperature['hot'] & self.humidity['medium'], self.ac_setting['high'])
        rule3 = ctrl.Rule(self.temperature['warm'] & self.humidity['high'], self.ac_setting['medium'])
        rule4 = ctrl.Rule(self.temperature['warm'] & self.humidity['medium'], self.ac_setting['medium'])
        rule5 = ctrl.Rule(self.temperature['cool'] & self.humidity['low'], self.ac_setting['low'])
        rule6 = ctrl.Rule(self.temperature['cold'], self.ac_setting['low'])
        rule7 = ctrl.Rule(self.time_of_day['night'] & self.temperature['warm'], self.ac_setting['medium'])
        rule8 = ctrl.Rule(self.time_of_day['afternoon'] & self.temperature['hot'], self.ac_setting['high'])
        rule9 = ctrl.Rule(self.time_of_day['morning'] & self.temperature['cool'], self.ac_setting['low'])
        rule10 = ctrl.Rule(self.humidity['high'] & self.temperature['warm'], self.ac_setting['high'])
        
        self.rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10]
        
    def _create_control_system(self):
        """Create the fuzzy control system."""
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
        
    def evaluate(self, temperature: float, humidity: float, time_of_day: float) -> float:
        """Evaluate the fuzzy system with given inputs."""
        try:
            self.simulation.input['temperature'] = temperature
            self.simulation.input['humidity'] = humidity
            self.simulation.input['time_of_day'] = time_of_day
            self.simulation.compute()
            return float(self.simulation.output['ac_setting'])
        except Exception as e:
            print(f"Error in fuzzy evaluation: {e}")
            return 50.0
            
    def get_membership_values(self, temperature: float, humidity: float, time_of_day: float) -> Dict:
        """Get membership values for all fuzzy sets given the inputs."""
        # Calculate membership values for temperature
        temp_cold = fuzz.interp_membership(self.temperature.universe, 
                                         self.temperature['cold'].mf, temperature)
        temp_cool = fuzz.interp_membership(self.temperature.universe, 
                                         self.temperature['cool'].mf, temperature)
        temp_warm = fuzz.interp_membership(self.temperature.universe, 
                                         self.temperature['warm'].mf, temperature)
        temp_hot = fuzz.interp_membership(self.temperature.universe, 
                                         self.temperature['hot'].mf, temperature)
        
        # Calculate membership values for humidity
        hum_low = fuzz.interp_membership(self.humidity.universe, 
                                       self.humidity['low'].mf, humidity)
        hum_medium = fuzz.interp_membership(self.humidity.universe, 
                                          self.humidity['medium'].mf, humidity)
        hum_high = fuzz.interp_membership(self.humidity.universe, 
                                        self.humidity['high'].mf, humidity)
        
        # Calculate membership values for time of day
        time_night = fuzz.interp_membership(self.time_of_day.universe, 
                                          self.time_of_day['night'].mf, time_of_day)
        time_morning = fuzz.interp_membership(self.time_of_day.universe, 
                                            self.time_of_day['morning'].mf, time_of_day)
        time_afternoon = fuzz.interp_membership(self.time_of_day.universe, 
                                              self.time_of_day['afternoon'].mf, time_of_day)
        time_evening = fuzz.interp_membership(self.time_of_day.universe, 
                                            self.time_of_day['evening'].mf, time_of_day)
        
        return {
            'temperature': {
                'cold': temp_cold,
                'cool': temp_cool,
                'warm': temp_warm,
                'hot': temp_hot
            },
            'humidity': {
                'low': hum_low,
                'medium': hum_medium,
                'high': hum_high
            },
            'time_of_day': {
                'night': time_night,
                'morning': time_morning,
                'afternoon': time_afternoon,
                'evening': time_evening
            }
        }
        
    def create_membership_plot(self):
        """Create membership functions plot and return as base64 string for Streamlit."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Temperature membership functions
        self.temperature.view(ax=axes[0, 0])
        axes[0, 0].set_title('Temperature Membership Functions')
        axes[0, 0].set_xlabel('Temperature (°C)')
        axes[0, 0].set_ylabel('Membership Degree')
        
        # Humidity membership functions
        self.humidity.view(ax=axes[0, 1])
        axes[0, 1].set_title('Humidity Membership Functions')
        axes[0, 1].set_xlabel('Humidity (%)')
        axes[0, 1].set_ylabel('Membership Degree')
        
        # Time of day membership functions
        self.time_of_day.view(ax=axes[1, 0])
        axes[1, 0].set_title('Time of Day Membership Functions')
        axes[1, 0].set_xlabel('Time (hours)')
        axes[1, 0].set_ylabel('Membership Degree')
        
        # AC Setting membership functions
        self.ac_setting.view(ax=axes[1, 1])
        axes[1, 1].set_title('AC Setting Membership Functions')
        axes[1, 1].set_xlabel('AC Setting (%)')
        axes[1, 1].set_ylabel('Membership Degree')
        
        plt.tight_layout()
        
        # Convert plot to base64 string for Streamlit
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()  # Close the plot to free memory
        
        return img_str
        
    def validate_system(self) -> Dict:
        """Validate the fuzzy system using test data."""
        test_cases = [
            (35, 85, 14, 90), (25, 60, 8, 50), (18, 40, 22, 20),
            (15, 35, 2, 10), (30, 75, 15, 80), (22, 50, 10, 30),
            (28, 80, 20, 70), (20, 45, 6, 25), (32, 90, 13, 95), (16, 30, 0, 5)
        ]
        
        results = []
        for temp, hum, time, expected in test_cases:
            actual = self.evaluate(temp, hum, time)
            error = abs(expected - actual)
            results.append({
                'temperature': temp, 'humidity': hum, 'time_of_day': time,
                'expected': expected, 'actual': actual, 'error': error
            })
            
        df = pd.DataFrame(results)
        mae = df['error'].mean()
        rmse = np.sqrt((df['error'] ** 2).mean())
        accuracy = (df['error'] <= 10).sum() / len(df) * 100
        
        return {
            'mean_absolute_error': mae,
            'root_mean_square_error': rmse,
            'accuracy_within_10_percent': accuracy,
            'test_cases': results
        } 
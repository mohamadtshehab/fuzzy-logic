import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from src.fuzzy_system import FuzzyAirConditioningSystem
import base64
from io import BytesIO

# Configure page
st.set_page_config(
    page_title="Fuzzy AC Control System",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: gray;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = FuzzyAirConditioningSystem()
    st.session_state.plot_generated = False

# Main header
st.markdown('<h1 class="main-header">❄️ Fuzzy Expert System: Air Conditioning Control</h1>', unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    st.subheader("Input Parameters")
    temperature = st.slider("🌡️ Temperature (°C)", 15, 35, 25, help="Range: 15-35°C")
    humidity = st.slider("💧 Humidity (%)", 30, 90, 50, help="Range: 30-90%")
    time_of_day = st.slider("🕐 Time of Day (hour)", 0, 24, 12, help="Range: 0-24 hours")
    
    st.subheader("System Information")
    st.info("""
    **Problem Statement:**
    This fuzzy expert system determines optimal air conditioning settings based on environmental conditions and time of day to optimize comfort while minimizing energy consumption.
    
    **Input Variables:**
    - Temperature (°C): 15-35
    - Humidity (%): 30-90  
    - Time of Day (hour): 0-24
    
    **Output Variable:**
    - AC Setting (%): 0-100
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎯 System Evaluation")
    
    # Evaluation button
    if st.button("🚀 Evaluate AC Setting", type="primary"):
        with st.spinner("Processing fuzzy logic..."):
            ac_setting = st.session_state.system.evaluate(temperature, humidity, time_of_day)
            
            # Display result with styling
            st.markdown(f"""
            <div class="success-box">
                <h3>✅ Recommended AC Setting</h3>
                <h2 style="color: #28a745; text-align: center;">{ac_setting:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Display membership values
            st.subheader("📊 Membership Values")
            membership_values = st.session_state.system.get_membership_values(temperature, humidity, time_of_day)
            
            # Create a nice display for membership values
            col_temp, col_hum, col_time = st.columns(3)
            
            with col_temp:
                st.write("**Temperature Membership:**")
                for key, value in membership_values['temperature'].items():
                    st.write(f"• {key.capitalize()}: {value:.3f}")
                    
            with col_hum:
                st.write("**Humidity Membership:**")
                for key, value in membership_values['humidity'].items():
                    st.write(f"• {key.capitalize()}: {value:.3f}")
                    
            with col_time:
                st.write("**Time of Day Membership:**")
                for key, value in membership_values['time_of_day'].items():
                    st.write(f"• {key.capitalize()}: {value:.3f}")

with col2:
    st.subheader("📈 Current Inputs")
    
    # Display current inputs in a nice format
    st.metric("Temperature", f"{temperature}°C")
    st.metric("Humidity", f"{humidity}%")
    st.metric("Time of Day", f"{time_of_day}:00")

# Membership Functions Plot
st.subheader("📊 Membership Functions Visualization")
st.write("These plots show how the fuzzy system interprets different input values.")

# Generate plot only when needed
if st.button("🔄 Update Membership Plots") or not st.session_state.plot_generated:
    with st.spinner("Generating membership function plots..."):
        plot_img = st.session_state.system.create_membership_plot()
        st.image(f"data:image/png;base64,{plot_img}", use_container_width=True)
        st.session_state.plot_generated = True

# System Validation
st.subheader("🔍 System Validation")
st.write("The system is validated using expert-defined test cases.")

validation = st.session_state.system.validate_system()

# Display validation metrics in cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Mean Absolute Error</h4>
        <h3>{validation['mean_absolute_error']:.2f}</h3>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Root Mean Square Error</h4>
        <h3>{validation['root_mean_square_error']:.2f}</h3>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Accuracy (within 10%)</h4>
        <h3>{validation['accuracy_within_10_percent']:.1f}%</h3>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Test Cases</h4>
        <h3>{len(validation['test_cases'])}</h3>
    </div>
    """, unsafe_allow_html=True)

# Display test cases in an expandable section
with st.expander("📋 View Test Cases"):
    test_df = pd.DataFrame(validation['test_cases'])
    st.dataframe(test_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Built with ❤️ using Streamlit and scikit-fuzzy</p>
    <p>Fuzzy Expert System for Air Conditioning Control</p>
</div>
""", unsafe_allow_html=True) 
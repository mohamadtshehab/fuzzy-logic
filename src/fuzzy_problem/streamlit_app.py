import streamlit as st
import pandas as pd
from src.fuzzy_problem.fuzzy_system import FuzzyDrivingRiskSystem
import matplotlib as mpl

st.set_page_config(
    page_title="Fuzzy Driving Risk System",
    page_icon="🚗",
    layout="wide"
)

mpl.rcParams.update({
    'axes.facecolor': '#f9f9f9',
    'savefig.facecolor': '#f9f9f9',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'legend.edgecolor': 'black'
})

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #d6336c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: transparent;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 5px solid #6f42c1;
    }
    .success-box {
        border: 2px solid #198754;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        background-color: rgba(0, 0, 0, 0);
    }
</style>
""", unsafe_allow_html=True)

if 'system' not in st.session_state:
    st.session_state.system = FuzzyDrivingRiskSystem()
    st.session_state.val = st.session_state.system.validate_system() 
    st.session_state.plot_generated = False

st.markdown('<h1 class="main-header">🚗 Fuzzy Expert System: Driving Risk & Intervention</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Input Panel")
    speed = st.slider("Speed (km/h) (0=Low, 140=High)", 0, 140, 80)
    weather = st.slider("Weather Condition (0=Good, 10=Bad)", 0.0, 10.0, 5.0, step=0.1)
    focus = st.slider("Driver Focus (0=Low, 10=High)", 0.0, 10.0, 5.0, step=0.1)

if st.button("Evaluate Risk & Intervention"):
    with st.spinner("Evaluating..."):
        print(f"Evaluating with: speed={speed}, weather={weather}, focus={focus}")  # طباعة المدخلات
        result = st.session_state.system.evaluate(speed, weather, focus)
        print(f"Result: risk={result['risk']}, intervention={result['intervention']}")  # طباعة المخرجات
        risk = result['risk']
        intervention = result['intervention']

        val = st.session_state.system.validate_system()
        st.session_state.val = val  
        st.markdown(f"""
        <div class="success-box">
            <h4>Results</h4>
            <p><strong>Risk Level:</strong> {risk:.2f}</p>
            <p><strong>Intervention Level:</strong> {intervention:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("System Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h4>Risk MAE</h4><h3>{st.session_state.val['mean_absolute_error_risk']:.2f}</h3></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h4>Intervention MAE</h4><h3>{st.session_state.val['mean_absolute_error_intervention']:.2f}</h3></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h4>Risk Accuracy</h4><h3>{st.session_state.val['accuracy_risk_within_1']:.1f}%</h3></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><h4>Intervention Accuracy</h4><h3>{st.session_state.val['accuracy_intervention_within_1']:.1f}%</h3></div>", unsafe_allow_html=True)

    st.subheader("Membership Degrees")
    st.markdown("**Values between 0 and 1 indicating the degree of membership to each fuzzy set.**")
    membership_values = st.session_state.system.get_membership_values(speed, weather, focus)

    col_s, col_w, col_f = st.columns(3)
    with col_s:
        st.write("**Speed**")
        for key, val in membership_values['speed'].items():
            st.write(f"{key.capitalize()}: {val:.3f}")
            st.progress(min(1.0, val))

    with col_w:
        st.write("**Weather**")
        for key, val in membership_values['weather'].items():
            st.write(f"{key.capitalize()}: {val:.3f}")
            st.progress(min(1.0, val))

    with col_f:
        st.write("**Focus**")
        for key, val in membership_values['focus'].items():
            st.write(f"{key.capitalize()}: {val:.3f}")
            st.progress(min(1.0, val))

# عرض القيم الحالية
st.subheader("Current Input Values")
col1, col2, col3 = st.columns(3)
col1.metric("Speed", f"{speed} km/h")
col2.metric("Weather", f"{weather}/10")
col3.metric("Focus", f"{focus}/10")

# عرض نتائج 
st.subheader("Evaluation Results")
with st.expander("Show Test Cases", expanded=False):
    st.write("### Test Cases Results")
    if st.session_state.val['test_cases']:
        df = pd.DataFrame(st.session_state.val['test_cases'])
        styled_df = df.style\
            .set_properties(**{'text-align': 'center'})\
            .format({
                'speed': '{:.0f}', 'weather': '{:.1f}', 'focus': '{:.1f}',
                'expected_risk': '{:.1f}', 'actual_risk': '{:.1f}',
                'expected_intervention': '{:.1f}', 'actual_intervention': '{:.1f}',
                'risk_error': '{:.3f}', 'interv_error': '{:.3f}'
            })\
            .set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#6f42c1'), ('color', 'white')]},
                {'selector': 'td', 'props': [('border', '1px solid #ddd')]}
            ])
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.warning("no data fuzzy_data.csv!")

st.subheader("All Fuzzy Logic Plots")
col1, col2 = st.columns(2)

with col1:
    if st.button("Show Membership Functions"):
        with st.spinner("Generating Membership Functions..."):
            plot_img = st.session_state.system.create_membership_plot()
            st.image(f"data:image/png;base64,{plot_img}", use_container_width=True)

with col2:
    if st.button("Show Output Results with Crisp Values"):
        with st.spinner("Generating Output Result Plots..."):
            risk_img = st.session_state.system.plot_risk_output(speed, weather, focus)
            intervention_img = st.session_state.system.plot_intervention_output(speed, weather, focus)
            col_risk, col_intervention = st.columns(2)
            with col_risk:
                st.subheader("Risk Output")
                st.image(f"data:image/png;base64,{risk_img}", use_container_width=True)
            with col_intervention:
                st.subheader("Intervention Output")
                st.image(f"data:image/png;base64,{intervention_img}", use_container_width=True)

st.markdown("---")
st.markdown("""<div style="text-align:center; color: #666;">
    <p>Built with Streamlit & Fuzzy Logic | Driving Safety System</p>
</div>""", unsafe_allow_html=True)
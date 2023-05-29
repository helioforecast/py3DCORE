import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Function to generate time series data
def generate_time_series():
    # Generate time values
    dt = np.arange(0, 10, 0.1)

    # Generate component values
    component1 = np.sin(dt)
    component2 = np.cos(dt)
    component3 = np.exp(-dt)

    return dt, component1, component2, component3


# Streamlit app
def main():
    st.title('Polynomial Fit and Time Series')

    # Generate time series data
    dt, component1, component2, component3 = generate_time_series()

    # Plot time series
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dt, y=component1, name='Component 1'))
    fig.add_trace(go.Scatter(x=dt, y=component2, name='Component 2'))
    fig.add_trace(go.Scatter(x=dt, y=component3, name='Component 3'))

    # Slider for polynomial fit degree
    degree = st.slider('Polynomial Fit Degree', min_value=1, max_value=10, value=1)

    # Calculate polynomial fit
    polynomial_fit = np.polyfit(component1, component2, degree)
    polynomial_values = np.polyval(polynomial_fit, component1)

    # Plot polynomial fit
    fig.add_trace(go.Scatter(x=component1, y=polynomial_values, name='Polynomial Fit'))

    # Configure layout
    fig.update_layout(
        title='Time Series and Polynomial Fit',
        xaxis_title='Time',
        yaxis_title='Value'
    )

    # Display plot
    st.plotly_chart(fig)


if __name__ == '__main__':
    main()
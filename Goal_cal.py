import numpy as np
pip install scipy
from scipy.stats import norm
import pandas as pd
import streamlit as st


st.title('Goal Value Calculator for Retriment :sunglasses:')
st.write("เครื่องมือนี้สร้างขึ้นมาเพื่อ คำนวณความน่าจะเป็นในการบรรลุเป้าหมายทางการเงิน ของการเกษียณอย่างง่าย ")

# Title for the app
st.title("Goal Value Calculator")

# User inputs
time_horizon = st.number_input("ระยะเวลาลงทุน(ก่อนเกษียณ)", min_value=1, value=10)
pool = st.number_input("เงินลงทุน", min_value=100000, value=1000000)
annu_value = st.number_input("เงินใช้หลังเกษียณรายเดือน", min_value=1000, value=30000)
Time_to_Use = st.number_input("ระยะเวลาการใช้เงิน", min_value=1, value=20)

# Inputs for weights
weight_1 = st.number_input("Weight Stock ", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
weight_2 = st.number_input("Weight Bond", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
weight_3 = st.number_input("Weight Cash", min_value=0.0, max_value=1.0, value=0.30, step=0.01)


W = [weight_1, weight_2 , weight_3 ]



# Validation and calculation of Goal Value
if weight_1 + weight_2 + weight_3 == 1.0:
    Goal_value = annu_value * Time_to_Use * 12
    st.write(f"The calculated Goal Value is: {Goal_value}")
else:
    st.error("Please ensure that the sum of weights equals 1.0")

# Running the app
# To run this app, save the code in a file named `app.py`, and then run `streamlit run app.py` in your terminal.




def calculate_density(target_amount, current_amount, expected_return, volatility, time_horizon):
    # Calculate the required annual growth rate
    required_growth_rate = (target_amount / current_amount) ** (1 / time_horizon) - 1

    # Calculate the mean and standard deviation of the expected final amount
    mean = expected_return * time_horizon
    std_dev = volatility * np.sqrt(time_horizon)

    # Calculate the Z-score
    z_score = (np.log(1 + required_growth_rate) - mean) / std_dev

    # Calculate density using the PDF of the normal distribution
    density = norm.pdf(z_score)

    return density


# Additional function to calculate volatility
def calculate_volatility(volatility_forecast, correlations, W):
    covariances = np.zeros((len(volatility_forecast), len(volatility_forecast)))
    for i in range(len(volatility_forecast)):
        for j in range(len(volatility_forecast)):
            covariances[i, j] = volatility_forecast[i] * volatility_forecast[j] * correlations[i, j]
    volatility = np.sqrt(np.dot(W, np.dot(covariances, W)))
    return volatility


cme = pd.DataFrame({'Volatility_Forecast': [0.3, 0.2, 0.001], 'Return_Forecast': [0.09, 0.02, 0.015]})
correlations = np.array([[1, 0.2, 0.01], [0.2, 1, 0.01], [0.01, 0.01, 1]])  # Example correlations matrix

# Calculate the updated volatility
volatility_forecast = cme['Volatility_Forecast'].values
volatility = calculate_volatility(volatility_forecast, correlations, W)

# Calculate and display the updated density
if st.button("Calculate Updated Density"):
    expected_return = (cme["Return_Forecast"] * W * 100).sum()
    density = 1 - calculate_density(Goal_value, pool, expected_return, volatility, time_horizon)
    st.write(f"The expected_return: {expected_return}")
    st.subheader(f"The updated calculated density(โอกาสบรรลุเป้าหมาย) is: {density *100}")






#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# --- Data Simulation ---
time = pd.date_range(start='2025-01-01', periods=500, freq='H')
np.random.seed(42)
vibration = np.random.normal(0.5, 0.05, size=500)
temperature = np.random.normal(70, 2, size=500)
vibration[-50:] += np.linspace(0, 0.5, 50)
temperature[-50:] += np.linspace(0, 10, 50)
failure = [0]*450 + [1]*50
df = pd.DataFrame({'timestamp': time, 'vibration': vibration, 'temperature': temperature, 'failure': failure})

# --- Train Model ---
X = df[['vibration', 'temperature']]
y = df['failure']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
df['predicted_failure'] = model.predict(X)

# --- Streamlit App ---
st.title("Predictive Maintenance Demo")

st.write("### Sensor Data")
st.line_chart(df[['vibration','temperature']])

st.write("### Predicted Failures (Red)")
plt.figure(figsize=(10,4))
plt.plot(df['vibration'], label='Vibration')
plt.scatter(df.index[df['predicted_failure']==1], df['vibration'][df['predicted_failure']==1], color='red', label='Predicted Failure')
plt.legend()
st.pyplot(plt)


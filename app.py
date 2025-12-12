import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from sklearn.linear_model import LinearRegression
import time

st.title("💜 Student Score Prediction App (Purple Theme)")

# Load the dataset
df = pd.read_csv("student_scores.csv")

x = df[['Hours']]
y = df['Score']

model = LinearRegression()
model.fit(x, y)

hours = st.number_input("Enter study hours:", min_value=0.0, step=0.5)

# BUTTON
if st.button("Predict Score"):
    prediction = model.predict([[hours]])
    st.success(f"Predicted Score: {prediction[0]:.2f} marks")

    # Create animated graph
    fig, ax = plt.subplots(figsize=(8, 5))

    # --- DARK PURPLE THEME ---
    ax.set_facecolor("#2A003D")  # dark purple background
    fig.patch.set_facecolor("#1A0025")  # darker outer background

    ax.set_xlabel("Study Hours", color="white")
    ax.set_ylabel("Score", color="white")
    ax.set_title("Study Hours vs Score (Purple Theme)", color="violet")

    # Set white color for ticks
    ax.tick_params(colors="white")

    # Plot scatter points (lavender color)
    ax.scatter(x, y, color="#D6C7FF", s=70, label="Data Points")

    # Prepare regression line animation
    x_vals = np.linspace(x.min(), x.max(), 100)
    y_vals = model.predict(x_vals.reshape(-1, 1))

    # Initialize empty line
    line, = ax.plot([], [], color="#FF66CC", linewidth=2.5, label="Regression Line")  # pink line

    # Animate line drawing
    for i in range(1, len(x_vals)):
        line.set_data(x_vals[:i], y_vals[:i])
        st.pyplot(fig)
        time.sleep(0.01)  # speed of animation

    st.pyplot(fig)

    # Save graph for download
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)

    st.download_button(
        label="📥 Download Graph as PNG",
        data=buf,
        file_name="student_score_graph.png",
        mime="image/png"
    )

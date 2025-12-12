import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from sklearn.linear_model import LinearRegression

st.title(" Student Score Prediction App (Purple Theme)")

# Load dataset
df = pd.read_csv("student_scores.csv")

x = df[['Hours']]
y = df['Score']

model = LinearRegression()
model.fit(x, y)

hours = st.number_input("Enter study hours:", min_value=0.0, step=0.5)

if st.button("Predict Score"):
    prediction = model.predict([[hours]])
    st.success(f"Predicted Score: {prediction[0]:.2f} marks")

    # Create graph
    fig, ax = plt.subplots(figsize=(8, 5))

    # Purple theme background
    ax.set_facecolor("#2A003D")
    fig.patch.set_facecolor("#1A0025")

    # Labels
    ax.set_xlabel("Study Hours", color="white")
    ax.set_ylabel("Score", color="white")
    ax.set_title("Study Hours vs Score (Purple Theme)", color="violet")
    ax.tick_params(colors="white")

    # Scatter points
    ax.scatter(x, y, color="#D6C7FF", s=80, label="Existing Data")

    # Regression line
    x_vals = np.linspace(x.min(), x.max(), 100)
    y_vals = model.predict(x_vals.reshape(-1, 1))
    ax.plot(x_vals, y_vals, color="#FF66CC", linewidth=2.5, label="Regression Line")

    # Mark predicted point
    ax.scatter(hours, prediction, color="yellow", s=150, marker="*", label="Predicted Score")

    ax.legend(facecolor="#2A003D", edgecolor="white", labelcolor="white")

    # Show graph once
    st.pyplot(fig)

    # Download button
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)

    st.download_button(
        label="📥 Download Graph as PNG",
        data=buf,
        file_name="student_score_graph.png",
        mime="image/png"
    )

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("Student Score Prediction App")

df=pd.read_csv("Student_scores.csv")

x=df[['Hours']]
y=df['Score']

model=LinearRegression()
model.fit(x,y)

hours=st.number_input('Enter study hours:',min_value=0.0,step=0.5)

if st.button("Predict Score"):
    prediction=model.predict([[hours]])
    st.success(f"Predicted Score:{prediction[0]:.2f}marks")


    fig,ax=plt.subplots()
    ax.scatter(x,y)
    ax.plot(x,model.predict(x))
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Score")
    ax.set_title("Study Hours vs Score")
    st.pyplot(fig)
    
    
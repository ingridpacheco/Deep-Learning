import streamlit as st
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

st.write("""
Predicao do dataset de Boston
""")

st.sidebar.header("Vamos colocar coisas aqui")

def usuario():
    crim = st.sidebar.slider("Taxa de crime per capta", 0.0,89.0,4.0,0.1)
    indus = st.sidebar.slider("Proporcao de acres de negocios nao varejistas por cidade", 0.0,28.0,11.0,0.5)
    chas = st.sidebar.slider("Variavel Charles River", 0.0,1.0,0.1,0.1)
    nox = st.sidebar.slider("Concentracao de acidos nitricos", 0.0,0.9,0.5,0.1)
    rm = st.sidebar.slider("Quantidade de quartos", 3.0,9.0,6.0,0.5)
    ptratio = st.sidebar.slider("Taxa de aluno-professor por cidade", 12.0,22.0,18.5)
    b = st.sidebar.slider("Proporcao de pessoas negras por cidade", 0.0,397.0,356.5)
    data = {"CRIM":crim,
            "INDUS":indus,
            "CHAS":chas,
            "NOX":nox,
            "RM":rm,
            "PTRATIO":ptratio,
            "B":b
    }
    features = pd.DataFrame(data,index=[0])
    return features

df = usuario()

st.subheader("Input do usuario")
st.write(df)

boston = load_boston()
data = pd.DataFrame(boston.data,columns=boston.feature_names)
data["MEDV"] = boston.target

X = data.drop(["RAD","TAX","MEDV","DIS","AGE","ZN","LSTAT"], axis = 1)
Y = data["MEDV"]

modelo = LinearRegression()
modelo.fit(X,Y)

y_predito = modelo.predict(df)

st.subheader("Predicao")
st.write(y_predito)
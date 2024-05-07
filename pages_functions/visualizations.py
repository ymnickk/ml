import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from functions.init_df import df
from pandas.plotting import scatter_matrix


def get_visualizations():
    numeric_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'Year', 'Month', 'Day']

    st.subheader("Тепловая карта корреляции")
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(14, 11))
    correlation_matrix = df[numeric_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5, ax=ax_heatmap)
    plt.savefig("heatmap.png")
    st.image("heatmap.png")

    st.subheader("Гистограммы")
    for feature in ['Rainfall', 'Evaporation', 'WindSpeed9am', 'WindSpeed3pm']:
        fig_hist = plt.figure()
        sns.histplot(data=df, x=feature, kde=True)
        plt.title(f"Гистограмма для {feature}")
        plt.savefig(f"hist_{feature}.png")
        st.image(f"hist_{feature}.png")

    st.subheader("Боксплоты №1")
    fig_boxplot, ax_boxplot = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df[numeric_features[0:9]], ax=ax_boxplot)
    plt.title("Боксплоты для числовых признаков")
    plt.savefig("boxplot.png")
    st.image("boxplot.png")

    st.subheader("Боксплоты №2")
    fig_boxplot, ax_boxplot = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df[numeric_features[9:18]], ax=ax_boxplot)
    plt.title("Боксплоты для числовых признаков")
    plt.savefig("boxplot_1.png")
    st.image("boxplot_1.png")

    st.subheader("Матрица диаграмм рассеяния №1")
    scatter_matrix_fig = scatter_matrix(df[numeric_features[0:9]], figsize=(12, 12), alpha=0.8, diagonal='hist')
    plt.savefig("scatter_matrix.png")
    st.image("scatter_matrix.png")

    st.subheader("Матрица диаграмм рассеяния №2")
    scatter_matrix_fig = scatter_matrix(df[numeric_features[9:18]], figsize=(12, 12), alpha=0.8, diagonal='hist')
    plt.savefig("scatter_matrix.png")
    st.image("scatter_matrix.png")

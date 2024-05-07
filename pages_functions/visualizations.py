import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from functions.init_df import df
from pandas.plotting import scatter_matrix


def get_visualizations():
    numeric_features = ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5', 'CNT']

    st.subheader("Тепловая карта корреляции")
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(14, 11))
    correlation_matrix = df[numeric_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5, ax=ax_heatmap)
    plt.savefig("heatmap.png")
    st.image("heatmap.png")

    st.subheader("Гистограммы")
    for feature in ['Temperature[C]', 'Humidity[%]']:
        fig_hist = plt.figure()
        sns.histplot(data=df, x=feature, kde=True)
        plt.title(f"Гистограмма для {feature}")
        plt.savefig(f"hist_{feature}.png")
        st.image(f"hist_{feature}.png")

    st.subheader("Боксплоты")
    fig_boxplot, ax_boxplot = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df[numeric_features[0:6]], ax=ax_boxplot)
    plt.title("Боксплоты для числовых признаков")
    plt.savefig("boxplot.png")
    st.image("boxplot.png")

    st.subheader("Боксплоты")
    fig_boxplot, ax_boxplot = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df[numeric_features[6:12]], ax=ax_boxplot)
    plt.title("Боксплоты для числовых признаков")
    plt.savefig("boxplot1.png")
    st.image("boxplot1.png")

    st.subheader("Матрица диаграмм рассеяния")
    scatter_matrix_fig = scatter_matrix(df[numeric_features[0:6]], figsize=(12, 12), alpha=0.8, diagonal='hist')
    plt.savefig("scatter_matrix.png")
    st.image("scatter_matrix.png")

    st.subheader("Матрица диаграмм рассеяния")
    scatter_matrix_fig = scatter_matrix(df[numeric_features[6:12]], figsize=(12, 12), alpha=0.8, diagonal='hist')
    plt.savefig("scatter_matrix1.png")
    st.image("scatter_matrix1.png")


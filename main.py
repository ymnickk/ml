import streamlit as st

from functions.run_function import run
from pages_functions.about_me import get_about_me
from pages_functions.info_data import get_info_data
from pages_functions.predictions import get_predictions
from pages_functions.visualizations import get_visualizations

st.title("Веб приложение")

pages = ["Информация о разработчике моделей ML",
         "Информация о наборе данных",
         "Визуализациями зависимостей",
         "Предсказания соответствующей модели ML"
]

functions_list = [get_about_me, get_info_data, get_visualizations, get_predictions]
page = st.sidebar.selectbox("Выбрать страницу", pages)

st.markdown(f'<h1 style="font-size:2em;">{page}</h1>', unsafe_allow_html=True)
pages_id = 0 if pages.index(page) is None else pages.index(page)
if not pages_id <= len(functions_list) - 1:
    pages_id = 0
run(functions_list[pages_id])

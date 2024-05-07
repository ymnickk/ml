import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, rand_score
from functions.init_df import X_test, y_test
from functions.load_model import get_models


def get_predictions():
    uploaded_file = st.file_uploader("Загрузите ваш файл CSV формата", type="csv")

    # Интерактивный ввод данных, если файл не загружен
    if uploaded_file is None:
        st.subheader("Введите данные для предсказания:")

        # Интерактивные поля для ввода данных
        input_data = {}

        feature_names_ru = {
            'MinTemp': 'Введите минимальную температуру в выбранный день: ',
            'MaxTemp': 'Введите максимальную температуру в выбранный день: ',
            'Rainfall': 'Введите количество осадков, выпавших за выбранный день (мм): ',
            'Evaporation': 'Введите количество испарений в поддоне класса А (мм) за 24 часа до 9 утра выбранного дня: ',
            'Sunshine': 'Введите количество часов яркого солнечного света в течение выбранного дня: ',
            'WindSpeed9am': 'Введите скорость ветра в 9 утра выбранного дня (км/ч): ',
            'WindSpeed3pm': 'Введите скорость ветра в 3 часа  после полудня выбранного дня (км/ч): ',
            'WindGustSpeed': 'Введите скорость самого сильного порыва ветра за 24 часа до полуночи выбранного дня (км/ч): ',
            'Humidity9am': 'Введите влажность в 9 утра выбранного дня (г/м3): ',
            'Humidity3pm': 'Введите влажность в 3 часа после полудня выбранного дня (г/м3): ',
            'Pressure9am': 'Введите давление в 9 утра выбранного дня (Па): ',
            'Pressure3pm': 'Введите давление в 3 часа после полудня выбранного дня (Па): ',
            'Cloud9am': 'Введите облачность в 9 утра выбранного дня (в % разделённых на 100): ',
            'Cloud3pm': 'Введите облачность в 3 часа после полудня выбранного дня (в % разделённых на 100): ',
            'Temp9am': 'Введите температуру в 9 утра выбранного дня: ',
            'Temp3pm': 'Введите температуру в 3 часа после полудня выбранного дня: ',
            'RainToday': 'Введите, был ли сегодня дождь или нет(1 - да, 0 - нет): ',
            'Year': 'Введите текущий год: ',
            'Month': 'Введите текущий месяц: ',
            'Day': 'Введите текущий день: ',
        }

        feature_names = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindSpeed9am', 'WindSpeed3pm',
                         'WindGustSpeed', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
                         'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'Year', 'Month', 'Day']

        input_data['Year'] = st.number_input(feature_names_ru.get('Year'), min_value=2000, max_value=2024, value=2024, step=1)
        input_data['Month'] = st.number_input(feature_names_ru.get('Month'), min_value=1, max_value=12, value=1, step=1)
        input_data['Day'] = st.number_input(feature_names_ru.get('Day'), min_value=1, max_value=31, value=15, step=1)
        input_data['RainToday'] = st.number_input(feature_names_ru.get('RainToday'), min_value=0, max_value=1, value=0, step=1)
        input_data['Rainfall'] = st.number_input(feature_names_ru.get('Rainfall'), min_value=0.0, value=0.0)
        input_data['MinTemp'] = st.number_input(feature_names_ru.get('MinTemp'), value=0)
        input_data['MaxTemp'] = st.number_input(feature_names_ru.get('MaxTemp'), value=0)
        input_data['Temp9am'] = st.number_input(feature_names_ru.get('Temp9am'), value=0)
        input_data['Temp3pm'] = st.number_input(feature_names_ru.get('Temp3pm'), value=0)
        input_data['Evaporation'] = st.number_input(feature_names_ru.get('Evaporation'), min_value=0.0, value=0.0)

        feature_names_unique_validate = ['RainToday', 'Year', 'Month', 'Day', 'Rainfall', 'MinTemp', 'MaxTemp',
                                         'Temp9am', 'Temp3pm']
        for feature in feature_names:
            if feature not in feature_names_unique_validate:
                input_data[feature] = st.number_input(f"{feature_names_ru.get(feature)}", min_value=0, value=10)




        if st.button('Сделать предсказание'):
            model1, model2, model3, model4, model5, model6 = get_models()
            input_df = pd.DataFrame([input_data])
            st.write("Входные данные:", input_df)

            # Сделать предсказания на тестовых данных
            predictions_ml1 = model1.predict(input_df)
            predictions_ml2 = model2.fit_predict(input_df)
            predictions_ml3 = model3.predict(input_df)
            predictions_ml4 = model4.predict(input_df)
            predictions_ml5 = model5.predict(input_df)
            probabilities_ml6 = model6.predict(input_df)
            predictions_ml6 = np.argmax(probabilities_ml6, axis=1)

            st.success(f"По предсказанию 1-й модели {'завтра будет дождь' if predictions_ml1 == 1 else 'завтра не будет дождя'}")
            st.success(f"По предсказанию 2-й модели {'завтра будет дождь' if predictions_ml2 == 1 else 'завтра не будет дождя'}")
            st.success(f"По предсказанию 3-й модели {'завтра будет дождь' if predictions_ml3 == 1 else 'завтра не будет дождя'}")
            st.success(f"По предсказанию 4-й модели {'завтра будет дождь' if predictions_ml4 == 1 else 'завтра не будет дождя'}")
            st.success(f"По предсказанию 5-й модели {'завтра будет дождь' if predictions_ml5 == 1 else 'завтра не будет дождя'}")
            st.success(f"По предсказанию 6-й модели {'завтра будет дождь' if predictions_ml6 == 1 else 'завтра не будет дождя'}")
            st.success(f"Более подробная информация: ")

            st.success(f"Предсказанние LogisticRegression: {predictions_ml1}")
            st.success(f"Предсказанние KMeans: {predictions_ml2}")
            st.success(f"Предсказанние GradientBoostingClassifier: {predictions_ml3}")
            st.success(f"Предсказанние BaggingClassifier: {predictions_ml4}")
            st.success(f"Предсказанние StackingClassifier: {predictions_ml5}")
            st.success(f"Предсказанние Tensorflow: {predictions_ml6}")
    else:
        try:
            model1, model2, model3, model4, model5, model6 = get_models()

            predictions_ml1 = model1.predict(X_test)
            predictions_ml2 = model2.fit_predict(X_test)
            predictions_ml3 = model3.predict(X_test)
            predictions_ml4 = model4.predict(X_test)
            predictions_ml5 = model5.predict(X_test)
            probabilities_ml6 = model6.predict(X_test)
            predictions_ml6 = np.argmax(probabilities_ml6, axis=1)

            # Оценить результаты
            accuracy_ml1 = accuracy_score(y_test, predictions_ml1)
            accuracy_ml2 = accuracy_score(y_test, predictions_ml2)
            rand_score_ml3 = round(rand_score(y_test, predictions_ml3))
            accuracy_ml4 = accuracy_score(y_test, predictions_ml4)
            accuracy_ml5 = accuracy_score(y_test, predictions_ml5)

            accuracy_ml6 = accuracy_score(y_test, predictions_ml6)
            st.success(f"Точность LogisticRegression: {accuracy_ml1}")
            st.success(f"Точность KMeans: {accuracy_ml2}")
            st.success(f"Rand Score GradientBoostingClassifier: {rand_score_ml3}")
            st.success(f"Точность BaggingClassifier: {accuracy_ml4}")
            st.success(f"Точность StackingClassifier: {accuracy_ml5}")
            st.success(f"Точность Tensorflow: {accuracy_ml6}")
        except Exception as e:
            st.error(f"Произошла ошибка при чтении файла: {e}")
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
            'Temperature[C]': 'Введите температуру в градусах Цельсия: ',
            'Humidity[%]': 'Введите влажность в процентах: ',
            'TVOC[ppb]': 'Введите концентрацию летучих органических соединений в частицах на миллиард (ppb): ',
            'eCO2[ppm]': 'Введите концентрацию диоксида углерода в частицах на миллион (ppm): ',
            'Raw H2': 'Введите сырые данные о содержании водорода: ',
            'Raw Ethanol': 'Введите сырые данные о содержании этанола: ',
            'Pressure[hPa]': 'Введите давление в гектопаскалях (гПа): ',
            'PM1.0': 'Введите PM1.0: ',
            'PM2.5': 'Введите PM2.5: ',
            'NC0.5': 'Введите NC0.5: ',
            'NC1.0': 'Введите NC1.0: ',
            'NC2.5': 'Введите NC2.5: ',
            'CNT': 'Введите CNT: ',
            'CNT1': 'Введите CNT1: ',
            'CNT2': 'Введите CNT2: ',
            'CNT3': 'Введите CNT3: ',
            'CNT4': 'Введите CNT4: ',
            'CNT5': 'Введите CNT5: ',
            'CNT6': 'Введите CNT6: ',
            'CNT7': 'Введите CNT7: ',
        }

        feature_names = ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5', 'CNT', 'CNT1', 'CNT2', 'CNT3', 'CNT4', 'CNT5', 'CNT6', 'CNT7']

        feature_names_unique_validate = ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5', 'CNT', 'CNT1', 'CNT2', 'CNT3', 'CNT4', 'CNT5', 'CNT6', 'CNT7']
        for feature in feature_names_unique_validate:
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

            st.success(f"По предсказанию 1-й модели {'сработала сигнализация' if predictions_ml1 == 1 else 'не сработала сигнализация'}")
            st.success(f"По предсказанию 2-й модели {'сработала сигнализация' if predictions_ml2 == 1 else 'не сработала сигнализация'}")
            st.success(f"По предсказанию 3-й модели {'сработала сигнализация' if predictions_ml3 == 1 else 'не сработала сигнализация'}")
            st.success(f"По предсказанию 4-й модели {'сработала сигнализация' if predictions_ml4 == 1 else 'не сработала сигнализация'}")
            st.success(f"По предсказанию 5-й модели {'сработала сигнализация' if predictions_ml5 == 1 else 'не сработала сигнализация'}")
            st.success(f"По предсказанию 6-й модели {'сработала сигнализация' if predictions_ml6 == 1 else 'не сработала сигнализация'}")
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

import streamlit as st


def get_about_me():
    st.markdown(f'<h1 style="font-size:2em;">Тема РГР</h1>', unsafe_allow_html=True)
    st.write("Разработка Web-приложения для инференса моделей ML и анализа данных")
    st.markdown(f'<h1 style="font-size:2em;">Обо мне</h1>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/me.jpg", width=170)
    with col2:
        st.write("Евтушевский Максим Сергеевич")
        st.write("ФИТ-221")

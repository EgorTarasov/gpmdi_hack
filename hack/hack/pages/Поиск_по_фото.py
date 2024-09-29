import streamlit as st
import time
import cv2
from PIL import Image
import numpy as np

from utils.photo_search import Similar
from utils.person_clustering import clustering
# Заголовок приложения
st.title("Поиск по фото")

name_db = 'db'

s = Similar(name_db=name_db)

col1, col2 = st.columns(2)
with col1:
    way = st.selectbox('Выберите способ', ('По своему фото', 'По людям в видео'))

video_dict = {
    'Silicon Valley S01E01.2014.KvK.BDRip.mp4': '0',
    'Silicon Valley S06E01.2019.KvK.WEB-DLRip.avi': '1',
    'Silicon Valley S06E05.2019.KvK.WEB-DLRip.mp4': '2'
}


if way == 'По своему фото':
    uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns(2)
    # Проверка, было ли загружено изображение
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        with col1:
            st.image(image, width=400)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('project/data/temp.jpg', image)

        with col2:
            video_name = st.selectbox("Выберите видео", list(video_dict.keys()))
        video_name_for_db = video_dict[video_name]
        video_path = f'project/data/{video_name}'
        with col2:
            button = st.button('Найти')
        if button:
            frames_idx, timings = s.get_timings('project/data/temp.jpg', video_path, video_name_for_db, 60)
            st.title("Результат")
            if not timings:
                st.write("Ничего не найдено")
            else:
                dict_col = {i: col for i, col in enumerate(st.columns(3))}
                for i, (frame, time) in enumerate(zip(frames_idx, timings)):
                    frame = s.get_frame(video_path, frame)
                    with dict_col[i % 3]:
                        st.image(frame, width=400)
                        st.write(f"{int(time // 60):02}:{int(time % 60):02}")

if way == 'По людям в видео':

    if "images_path_list" not in st.session_state:
        st.session_state.images_path_list = clustering(st.session_state.whisper_data, st.session_state.frames, "project/data")
    
    if not len(st.session_state.images_path_list):
        st.write("Людей не найдено")
    if len(st.session_state.images_path_list):
        dict_col = {i: col for i, col in enumerate(st.columns(5))}
        for i, image_path in enumerate(st.session_state.images_path_list):
            with dict_col[i % 5]:
                image = np.array(Image.open(image_path))
                st.image(image, width=200)
                st.write(f'photo {i + 1:02}')
        
        col1, col2 = st.columns(2)
        with col1:
            photo_idx = st.selectbox('Выберите фото', [''] + [f"{i + 1:02}" for i in range(len(st.session_state.images_path_list))])
        if photo_idx:
            image = np.array(Image.open(st.session_state.images_path_list[int(photo_idx) - 1]))
            st.image(image, width=400)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite('project/data/temp.jpg', image)

            with col2:
                video_name = st.selectbox("Выберите видео", list(video_dict.keys()))
            video_name_for_db = video_dict[video_name]
            video_path = f'project/data/{video_name}'
            with col2:
                button = st.button('Найти')
            if button:
                frames_idx, timings = s.get_timings('project/data/temp.jpg', video_path, video_name_for_db, 60)
                st.title("Результат")
                if not timings:
                    st.write("Ничего не найдено")
                else:
                    dict_col = {i: col for i, col in enumerate(st.columns(3))}
                    for i, (frame, time) in enumerate(zip(frames_idx, timings)):
                        frame = s.get_frame(video_path, frame)
                        with dict_col[i % 3]:
                            st.image(frame, width=400)
                            st.write(f"{int(time // 60):02}:{int(time % 60):02}")


        


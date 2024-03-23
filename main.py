import streamlit as st
from PIL import Image
import numpy as np
from annotated_text import annotated_text
import tensorflow as tf
import keras

new_size = (60, 80)

def preprocess_image(image):

    resized_image = image.resize(new_size)
    img_array = np.array(resized_image)
    img_array = keras.utils.img_to_array(img_array)
    img_array = keras.ops.expand_dims(img_array, 0) 

    return img_array


st.title('Phân loại đồ gia dụng')
Class_names = ['ao', 'balo,tui xach,vi', 'do dien tu', 'dong ho', 'giay dep', 'mi pham', 'phu kien', 'quan', 'vay']

tab1, tab2 = st.tabs(['Camera', 'load image'])

checkpoint_path = '.\save_at_7.keras'
model = tf.keras.models.load_model(checkpoint_path)

with tab1:
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:

        img = Image.open(img_file_buffer)
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)
        st.write(predictions)
        label_index = int(np.argmax(predictions[0]))
        score = predictions[0][label_index]
        label_name = Class_names[label_index]
        st.write(label_name,score)

with tab2:
    
    st.title('Phân loại đồ gia dụng image')
    # File uploader for image
    uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"],accept_multiple_files=True)
    if uploaded_files is not None:
        try:
            list_images = []
            for uploaded_file in uploaded_files:
                img = Image.open(uploaded_file)
                img_array = preprocess_image(img)
                list_images.append(img_array)
            image_np = keras.ops.squeeze(np.array(list_images))
            predictions = model.predict(image_np)
            label_indexs = np.argmax(predictions,axis = 1)
            # st.write(predictions,label_indexs)
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                for i in range(0,len(label_indexs),5):
                    container = st.container(border=True)
                    container.write(Class_names[label_indexs[i]])
                    container.image(uploaded_files[i])


            with col2:
                for i in range(1,len(label_indexs),5):
                    container = st.container(border=True)
                    container.write(Class_names[label_indexs[i]])
                    container.image(uploaded_files[i])

            with col3:
                for i in range(2,len(label_indexs),5):
                    container = st.container(border=True)
                    container.write(Class_names[label_indexs[i]])
                    container.image(uploaded_files[i])
            with col4:
                for i in range(3,len(label_indexs),5):
                    container = st.container(border=True)
                    container.write(Class_names[label_indexs[i]])
                    container.image(uploaded_files[i])
            with col5:
                for i in range(4,len(label_indexs),5):
                    container = st.container(border=True)
                    container.write(Class_names[label_indexs[i]])
                    container.image(uploaded_files[i])
        except:
            pass

       


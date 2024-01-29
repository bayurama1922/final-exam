import streamlit as st
from streamlit_option_menu import option_menu
from googletrans import Translator
import moviepy.editor as mp
import speech_recognition as sr
from transformers import BertTokenizer, BertForSequenceClassification,pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image as PILImage
from datetime import datetime
import os
import re

selected = option_menu(
        menu_title=None,
        options=["Home", "Go Terjemahan", "Video to Text"],
        icons=['house'],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )
selected

uploads_dir = "proses_data"
os.makedirs(uploads_dir, exist_ok=True)
timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")

def secure_filename(filename):
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "", filename)
    filename = filename[:255]
    return filename

def analyze_sentiment(text):
    pretrained = "mdhugol/indonesia-bert-sentiment-classification"
    model = BertForSequenceClassification.from_pretrained(pretrained)
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}
    max_length = 512
    text = text[:max_length]
    result = sentiment_analysis(text)
    status = label_index[result[0]['label']]
    score = result[0]['score']
    return status, score

def create_tagcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")

    img_stream = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(img_stream, format='png')
    plt.close()
    img_stream.seek(0)

    # Menyimpan gambar tag cloud ke dalam direktori uploads dengan timestamp
    img_path = os.path.join(uploads_dir, f'hasil_tagcloud_{timestamp_str}.png')
    plt.savefig(img_path, format='png')
    plt.close()

    tagcloud_image = PILImage.open(img_stream)
    return tagcloud_image

def recognize_speech(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_file = recognizer.record(source)
    try:
        text_result = recognizer.recognize_google(audio_file, language="id-ID")
    except sr.UnknownValueError:
        st.warning("Pengenalan suara tidak dapat memahami audio.")
        text_result = ""
    return text_result

if (selected == 'Go Terjemahan') :
    st.header('Go Translate')
    # Daftar kode bahasa yang dapat dipilih
    language_options = {
        'Indonesia': 'id',
        'English': 'en',
        'Spanish': 'es',
        'French': 'fr',
        'German': 'de',
        'Italian': 'it',
        'Japanese': 'ja',
        'Korean': 'ko',
        'Chinese (Simplified)': 'zh-CN',
        'Arabic': 'ar'
    }

    # Pilih bahasa sumber
    source_language = st.selectbox('Pilih Bahasa Sumber', list(language_options.keys()))
    # Pilih bahasa tujuan
    target_language = st.selectbox('Pilih Bahasa Tujuan', list(language_options.keys()))

    input_text = st.text_area('Masukkan teks untuk diterjemahkan')

    if st.button('Terjemahkan'):
        if input_text:
            translator = Translator()
            translated_text = translator.translate(input_text, src=language_options[source_language], dest=language_options[target_language])
            st.subheader(f'Hasil Terjemahan dari {source_language} ke {target_language}:')
            st.write(translated_text.text)
        else:
            st.warning('Masukkan teks terlebih dahulu.')

if (selected == 'Home') :
    st.header('Selamat Datang di RamaProject.co!')
    st.subheader('Jelajahi Keajaiban AI Terjemah dan Konversi Video')

    st.write(
        "Selamat datang di RamaProject.co, platform pintar yang menggabungkan kecerdasan buatan "
        "dalam pengalaman pengguna sehari-hari Anda. 🚀 Temukan fitur menarik seperti Go Terjemahan untuk "
        "menerjemahkan teks ke berbagai bahasa, serta Converter Video untuk merubah percakapan video menjadi "
        "teks, dilengkapi dengan analisis sentimen dan tag cloud. 💬"
    )

    st.write(
        "Kemudahan dan kecanggihan dalam genggaman Anda. Mulai eksplorasi sekarang! 🌐"
    )

    st.subheader("Alur Kerja Go Terjemahan:")
    st.write(
        "- Pilih bahasa sumber dan bahasa tujuan menggunakan dropdown. 🌍\n"
        "- Masukkan teks yang ingin Anda terjemahkan dalam area teks. ✍️\n"
        "- Klik tombol 'Terjemahkan'. 🚀\n"
        "- Teks hasil terjemahan akan ditampilkan di bawah tombol. 📄"
    )

    st.subheader("Alur Kerja Converter Video to Text:")
    st.write(
        "- Unggah video yang ingin Anda konversi menggunakan tombol 'Unggah video untuk konversi'. 🎥\n"
        "- Klik tombol 'Konversi ke Teks'. 🚀\n"
        "- Proses konversi video akan dimulai, dan progress bar akan menunjukkan kemajuan. 🔄\n"
        "- Hasil teks konversi, analisis sentimen, dan tag cloud akan ditampilkan di bawahnya. 💬"
    )

if (selected == 'Video to Text') :
    st.header('Converter Video to Text, Sentiment Analysis and Tag Cloud')
    video_file = st.file_uploader('Unggah video untuk konversi', type=['mp4'])

    if st.button('Konversi ke Teks') and video_file:
        video_path = f'proses_data/{timestamp_str}_video.mp4'
        with open(video_path, 'wb') as f:
            f.write(video_file.read())

        audio_path = f'proses_data/{timestamp_str}_audio.wav'
        video_clip = mp.VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path)

        text_result = recognize_speech(audio_path)

        text_path = f'proses_data/{timestamp_str}_text_result.txt'
        with open(text_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text_result)

        # Display the uploaded video
        st.subheader('Video Hasil Konversi:')
        st.video(video_path, format='video/mp4')

        # Analisis sentimen
        sentiment, score = analyze_sentiment(text_result)
        st.subheader('Hasil Konversi Teks:')
        st.write(text_result)
        st.subheader('Analisis Sentimen:')
        st.write(f'Sentimennya adalah "{sentiment}"')
        st.write(f'Skornya adalah "{score}"')

        # Buat dan tampilkan tag cloud dari teks hasil konversi
        tagcloud_image = create_tagcloud(text_result)
        st.subheader('Tag Cloud:')
        st.image(tagcloud_image, caption='Tag Cloud', use_column_width=True)

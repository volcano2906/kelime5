import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from collections import Counter
from itertools import islice
from nltk.util import ngrams

# Stopwords'leri yükle
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Sayfa ayarlarını tam ekran yap
st.set_page_config(layout="wide")

# Başlık
st.title("Uygulama ID'lerine Göre Rank Edilmiş Anahtar Kelimeler ve Puanlama")

# Kullanıcıdan Title, Subtitle ve KW girişi
st.subheader("Anahtar Kelime Karşılaştırma")
title = st.text_input("Title (Maksimum 30 karakter)", max_chars=30)
subtitle = st.text_input("Subtitle (Maksimum 30 karakter)", max_chars=30)
kw_input = st.text_input("Keyword Alanı (Maksimum 100 karakter, space veya comma ile ayırın)", max_chars=100)

# Girilen alanları birleştir ve temizle
all_keywords = set(re.split(r'[ ,]+', title + ' ' + subtitle + ' ' + kw_input))
all_keywords = {word.lower().strip() for word in all_keywords if word and word.lower() not in stop_words}

# CSV dosyalarını yükleme
uploaded_files = st.file_uploader("CSV dosyanızı yükleyin", type=["csv"], accept_multiple_files=True)

# Anahtar kelime hacmi 5 olanları filtreleme seçeneği
drop_low_volume = st.checkbox("Exclude Keywords with Volume 5")
include_volume_5_in_analysis = st.checkbox("Include Volume 5 Keywords in Analysis", value=True)

if uploaded_files:
    # Dosyaları oku ve birleştir
    df_list = [pd.read_csv(file) for file in uploaded_files]
    df = pd.concat(df_list, ignore_index=True)
    df = df.drop_duplicates()
    
    # Anahtar kelime hacmi 5 olanları analizden hariç tutma
    df_analysis = df.copy()
    if not include_volume_5_in_analysis:
        df_analysis = df_analysis[df_analysis["Volume"] != 5]
    
    keywords_list_filtered = ' '.join(df_analysis["Keyword"].dropna()).lower().split()
    monograms_filtered = Counter(keywords_list_filtered)
    bigrams_filtered = Counter(ngrams(keywords_list_filtered, 2))
    trigrams_filtered = Counter(ngrams(keywords_list_filtered, 3))
    
    # Anahtar kelime hacmi 5 olanları filtrele
    if drop_low_volume:
        df = df[df["Volume"] != 5]
    
    # Rank değerlerini sayıya çevir ve puan hesapla
    df["Rank"] = df["Rank"].astype(str)  # Rank sütunu string olmalı
    
    # Sonuçları gösterme
    st.write("### En Çok Tekrar Eden Kelimeler")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Monogram (Tek Kelimeler)**")
        monogram_data = monograms_filtered.most_common(10)
        for word, count in monogram_data:
            st.write(f"{word}: {count}")
    
    with col2:
        st.write("**Bigram (İki Kelimeli Öbekler)**")
        bigram_data = bigrams_filtered.most_common(10)
        for bigram, count in bigram_data:
            st.write(f"{' '.join(bigram)}: {count}")
    
    with col3:
        st.write("**Trigram (Üç Kelimeli Öbekler)**")
        trigram_data = trigrams_filtered.most_common(10)
        for trigram, count in trigram_data:
            st.write(f"{' '.join(trigram)}: {count}")
    
    # CSV olarak indirme butonu
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Dönüştürülmüş CSV'yi İndir",
        data=csv,
        file_name="converted_keywords_with_scores.csv",
        mime="text/csv"
    )

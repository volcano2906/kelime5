import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
from collections import Counter
import nltk

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
all_keywords = set(re.split(r'[ ,]+', f"{title} {subtitle} {kw_input}".strip().lower()))
all_keywords = {word for word in all_keywords if word and word not in stop_words}

# CSV dosyalarını yükleme
uploaded_files = st.file_uploader("CSV dosyanızı yükleyin", type=["csv"], accept_multiple_files=True)

# Anahtar kelime hacmi 5 olanları filtreleme seçeneği
drop_low_volume = st.checkbox("Exclude Keywords with Volume 5")

def update_rank(rank):
    try:
        rank = int(float(rank))  # Önce float, sonra int dönüşümü
    except ValueError:
        return 1
    return 5 if rank <= 10 else 4 if rank <= 30 else 3 if rank <= 50 else 2 if rank <= 249 else 1

if uploaded_files:
    # Dosyaları oku ve birleştir
    df_list = [pd.read_csv(file) for file in uploaded_files]
    df = pd.concat(df_list, ignore_index=True).drop_duplicates()
    
    # Anahtar kelime hacmi 5 olanları filtrele
    if drop_low_volume:
        df = df[df["Volume"] != 5]
    
    # Rank değerlerini sayıya çevir ve puan hesapla
    df["Rank"] = df["Rank"].fillna("250").astype(str)
    df["Score"] = df["Rank"].apply(update_rank)
    
    # Eksik kelimeleri bul
    def find_missing_keywords(keyword):
        words = set(re.split(r'[ ,]+', keyword.lower()))
        missing_words = words - all_keywords
        return ', '.join(missing_words) if missing_words else "-"

    df["Missing Keywords"] = df["Keyword"].apply(find_missing_keywords)
    
    ### **Eksik Kelimeler Üzerinden Frekans Analizi** ###
    st.subheader("Eksik Kelimeler İçin Frekans Analizi")

    # Missing keywords'in tamamını tek bir string listesi olarak al
    missing_text = " ".join(df["Missing Keywords"].dropna().tolist())

    # Kelime ayrıştırma fonksiyonları
    def extract_words(text):
        words = re.split(r'[ ,]+', text.lower())
        return [word.strip() for word in words if word and word not in stop_words]

    def extract_ngrams(text, n):
        words = extract_words(text)
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

    # Missing keywords üzerinden frekans analizi
    missing_words = extract_words(missing_text)
    missing_bigrams = extract_ngrams(missing_text, 2)
    missing_trigrams = extract_ngrams(missing_text, 3)

    # Frekansları hesapla
    missing_word_freq = pd.DataFrame(Counter(missing_words).items(), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False)
    missing_bigram_freq = pd.DataFrame(Counter(missing_bigrams).items(), columns=["Bigram", "Frequency"]).sort_values(by="Frequency", ascending=False)
    missing_trigram_freq = pd.DataFrame(Counter(missing_trigrams).items(), columns=["Trigram", "Frequency"]).sort_values(by="Frequency", ascending=False)

    # Sonuçları yatay olarak gösterme
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Eksik Tek Kelimeler (Unigrams)**")
        st.dataframe(missing_word_freq, use_container_width=True)

    with col2:
        st.write("**Eksik İki Kelimelik Kombinasyonlar (Bigrams)**")
        st.dataframe(missing_bigram_freq, use_container_width=True)

    with col3:
        st.write("**Eksik Üç Kelimelik Kombinasyonlar (Trigrams)**")
        st.dataframe(missing_trigram_freq, use_container_width=True)

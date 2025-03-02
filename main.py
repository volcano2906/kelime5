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
    
    # Tüm eksik kelimeleri set olarak al
    missing_keywords_set = set(df["Missing Keywords"].str.lower().str.split(', ').explode())

    # Kelime ayrıştırma fonksiyonları
    def extract_words(text):
        words = re.split(r'[ ,]+', text.lower())
        return [word.strip() for word in words if word and word not in stop_words]

    def extract_ngrams(text, n):
        words = extract_words(text)
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

    # Frekans analizi için verileri hazırla
    all_words = []
    all_bigrams = []
    all_trigrams = []

    for keyword in df["Keyword"]:
        words = extract_words(keyword)
        all_words.extend(words)
        all_bigrams.extend(extract_ngrams(keyword, 2))
        all_trigrams.extend(extract_ngrams(keyword, 3))

    # Frekansları hesapla
    word_freq = pd.DataFrame(Counter(all_words).items(), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False)
    bigram_freq = pd.DataFrame(Counter(all_bigrams).items(), columns=["Bigram", "Frequency"]).sort_values(by="Frequency", ascending=False)
    trigram_freq = pd.DataFrame(Counter(all_trigrams).items(), columns=["Trigram", "Frequency"]).sort_values(by="Frequency", ascending=False)

    # "Missing Keywords" analizi yap
    word_freq["Missing"] = word_freq["Word"].apply(lambda x: "Yes" if x in missing_keywords_set else "No")
    bigram_freq["Missing"] = bigram_freq["Bigram"].apply(lambda x: "Yes" if x in missing_keywords_set else "No")
    trigram_freq["Missing"] = trigram_freq["Trigram"].apply(lambda x: "Yes" if x in missing_keywords_set else "No")

    # Sonuçları yatay olarak gösterme
    st.write("### Eksik Kelimeler İçin Frekans Analizi")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Tek Kelimeler (Unigrams)**")
        st.dataframe(word_freq, use_container_width=True)

    with col2:
        st.write("**İki Kelimelik Kombinasyonlar (Bigrams)**")
        st.dataframe(bigram_freq, use_container_width=True)

    with col3:
        st.write("**Üç Kelimelik Kombinasyonlar (Trigrams)**")
        st.dataframe(trigram_freq, use_container_width=True)

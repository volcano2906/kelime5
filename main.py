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
kw_input = st.text_input("Keyword Alanı (Maksimum 100 karakter, space veya comma ile ayırın)", max_chars=400)

# Girilen kelimeleri temizle ve set olarak sakla
user_input_text = f"{title} {subtitle} {kw_input}".strip().lower()
user_words = set(re.split(r'[ ,]+', user_input_text))
user_words = {word for word in user_words if word and word not in stop_words}

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
        missing_words = words - user_words
        return ','.join(missing_words) if missing_words else "-"

    df["Missing Keywords"] = df["Keyword"].apply(find_missing_keywords)
    
    # Veriyi uygun formata dönüştürme
    pivot_df = df.pivot_table(
        index=["Keyword", "Volume"], 
        columns="Application Id", 
        values="Rank", 
        aggfunc='first'
    ).reset_index()
    
    # Puanları toplama ve Rank sayısını hesaplama
    summary_df = df.groupby("Keyword").agg(
        Total_Score=("Score", "sum"),
        Rank_Count=("Rank", "count"),
        Missing_Keywords=("Missing Keywords", "first")
    ).reset_index()

    # Tabloları birleştir
    pivot_df = pivot_df.merge(summary_df, on="Keyword", how="left")

    # Boş değerleri "null" olarak değiştir
    pivot_df.fillna("null", inplace=True)

    # Sonuçları gösterme
    st.write("### Dönüştürülmüş Veri Tablosu ve Puanlar")
    st.dataframe(pivot_df, use_container_width=True)

    # CSV olarak indirme butonu
    csv = pivot_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Dönüştürülmüş CSV'yi İndir",
        data=csv,
        file_name="converted_keywords_with_scores.csv",
        mime="text/csv"
    )

    ### Ek Alan: Frekans Analizi ###
    st.subheader("Anahtar Kelime Frekans Analizi")

    # Ek filtreleme seçenekleri
    exclude_low_volume_freq = st.checkbox("Exclude Keywords with Volume 5 in Frequency Analysis")
    exclude_single_app_keywords = st.checkbox("Exclude Keywords Ranked by Only One App in Frequency Analysis")

    # Filtreleme uygulama
    freq_df = df.copy()
    if exclude_low_volume_freq:
        freq_df = freq_df[freq_df["Volume"] != 5]
    if exclude_single_app_keywords:
        freq_df = freq_df[freq_df.groupby("Keyword")["Application Id"].transform("nunique") > 1]

    # Kelime ayrıştırma fonksiyonları
    def extract_words(text):
        words = re.split(r'[ ,]+', text.lower())
        return [word.strip() for word in words if word and word not in stop_words]

    def extract_ngrams(text, n):
        words = extract_words(text)
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

    # Tüm kelimeleri içeren liste
    all_words = []
    all_bigrams = []
    all_trigrams = []

    for keyword in freq_df["Keyword"]:
        words = extract_words(keyword)
        all_words.extend(words)
        all_bigrams.extend(extract_ngrams(keyword, 2))
        all_trigrams.extend(extract_ngrams(keyword, 3))

    # Eksik kelimeleri bul (word-level check)
    def find_missing_items(ngram):
        ngram_words = set(ngram.split())  # Bütün kelimeleri böl ve set olarak al
        missing_words = ngram_words - user_words  # Kullanıcının girmediği kelimeleri bul
        return ','.join(missing_words) if missing_words else "-"

    # Frekansları hesapla ve eksik kelimeleri ekle
    word_freq = pd.DataFrame(Counter(all_words).items(), columns=["Word", "Frequency"])
    word_freq["Missing Keywords"] = word_freq["Word"].apply(find_missing_items)
    
    bigram_freq = pd.DataFrame(Counter(all_bigrams).items(), columns=["Bigram", "Frequency"])
    bigram_freq["Missing Keywords"] = bigram_freq["Bigram"].apply(find_missing_items)
    
    trigram_freq = pd.DataFrame(Counter(all_trigrams).items(), columns=["Trigram", "Frequency"])
    trigram_freq["Missing Keywords"] = trigram_freq["Trigram"].apply(find_missing_items)

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

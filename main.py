import streamlit as st
import pandas as pd
import re
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import stopwords
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

# Tek bir uygulamada rank edilen kelimeleri filtreleme seçeneği
drop_single_app_rank = st.checkbox("Exclude Keywords Ranked in Only One App")

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

    # Kelime analizini yapmadan önce tek bir uygulamada geçen kelimeleri çıkartalım
    if drop_single_app_rank:
        app_counts = df.groupby("Keyword")["Application Id"].nunique()
        df = df[df["Keyword"].isin(app_counts[app_counts > 1].index)]

    # Eksik kelimeleri bul
    def find_missing_keywords(keyword):
        words = set(re.split(r'[ ,]+', keyword.lower()))
        missing_words = words - all_keywords
        return ', '.join(missing_words) if missing_words else "-"

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

    # ---- N-gram Analysis ----
    st.subheader("Kelime Frekans Analizi")

    # Tek kelime, biword ve treeword analizlerini yapalım
    all_text = ' '.join(df["Keyword"]).lower()
    words = [word for word in re.split(r'\W+', all_text) if word and word not in stop_words]
    
    # Kelime frekansları
    word_counts = Counter(words)
    biword_counts = Counter(ngrams(words, 2))
    triword_counts = Counter(ngrams(words, 3))

    # Tek kelime frekans tablosu
    word_df = pd.DataFrame(word_counts.items(), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False)
    biword_df = pd.DataFrame(biword_counts.items(), columns=["Biword", "Frequency"]).sort_values(by="Frequency", ascending=False)
    triword_df = pd.DataFrame(triword_counts.items(), columns=["Triword", "Frequency"]).sort_values(by="Frequency", ascending=False)

    # Verileri göster
    st.write("#### Tek Kelime Frekansı")
    st.dataframe(word_df, use_container_width=True)

    st.write("#### Biword (İki Kelime) Frekansı")
    st.dataframe(biword_df, use_container_width=True)

    st.write("#### Triword (Üç Kelime) Frekansı")
    st.dataframe(triword_df, use_container_width=True)

    # CSV indirme butonları
    word_csv = word_df.to_csv(index=False).encode('utf-8')
    biword_csv = biword_df.to_csv(index=False).encode('utf-8')
    triword_csv = triword_df.to_csv(index=False).encode('utf-8')

    st.download_button("Tek Kelime Frekanslarını İndir", data=word_csv, file_name="word_frequency.csv", mime="text/csv")
    st.download_button("Biword Frekanslarını İndir", data=biword_csv, file_name="biword_frequency.csv", mime="text/csv")
    st.download_button("Triword Frekanslarını İndir", data=triword_csv, file_name="triword_frequency.csv", mime="text/csv")

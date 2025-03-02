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
include_volume_5_in_analysis = st.checkbox("Include Volume 5 Keywords in Analysis")

# Yeniden hesaplama düğmesi
treemap_recalculate = st.button("Tekrar Eden Kelimeleri Yeniden Hesapla")

def update_rank(rank):
    try:
        rank = float(rank)
    except:
        rank = 250.0
    if 1 <= rank <= 10:
        return 5
    elif 11 <= rank <= 30:
        return 4
    elif 31 <= rank <= 50:
        return 3
    elif 51 <= rank <= 249:
        return 2
    else:
        return 1

if uploaded_files:
    # Dosyaları oku ve birleştir
    df_list = [pd.read_csv(file) for file in uploaded_files]
    df = pd.concat(df_list, ignore_index=True)
    df = df.drop_duplicates()
    
    # Anahtar kelime hacmi 5 olanları filtrelemeden önce monogram, bigram ve trigram hesapla
    df_analysis = df.copy()
    if not include_volume_5_in_analysis:
        df_analysis = df_analysis[df_analysis["Volume"] != 5]
    
    keywords_list_full = ' '.join(df_analysis["Keyword"].dropna()).lower().split()
    monograms_full = Counter(keywords_list_full)
    bigrams_full = Counter(ngrams(keywords_list_full, 2))
    trigrams_full = Counter(ngrams(keywords_list_full, 3))
    
    # Anahtar kelime hacmi 5 olanları filtrele
    if drop_low_volume:
        df = df[df["Volume"] != 5]
    
    # Eğer düğmeye basılırsa tekrar eden kelimeleri hesapla
    if treemap_recalculate:
        df_analysis = df.copy()
        if not include_volume_5_in_analysis:
            df_analysis = df_analysis[df_analysis["Volume"] != 5]
        
        keywords_list_filtered = ' '.join(df_analysis["Keyword"].dropna()).lower().split()
        monograms_filtered = Counter(keywords_list_filtered)
        bigrams_filtered = Counter(ngrams(keywords_list_filtered, 2))
        trigrams_filtered = Counter(ngrams(keywords_list_filtered, 3))
    else:
        monograms_filtered = monograms_full
        bigrams_filtered = bigrams_full
        trigrams_filtered = trigrams_full
    
    # Rank değerlerini sayıya çevir ve puan hesapla
    df["Rank"] = df["Rank"].astype(str)  # Rank sütunu string olmalı
    df["Score"] = df["Rank"].apply(update_rank)
    
    # Eksik kelimeleri bul
    def find_missing_keywords(keyword):
        words = set(re.split(r'[ ,]+', keyword.lower()))
        missing_words = words - all_keywords - stop_words
        return ', '.join(missing_words) if missing_words else "-"
    
    df["Missing Keywords"] = df["Keyword"].apply(find_missing_keywords)
    
    # Veriyi uygun formata dönüştürme (Keyword'ler satır, Application Id'ler sütun, Rank değerleri hücrede)
    pivot_df = df.pivot_table(index=["Keyword", "Volume"], columns="Application Id", values="Rank", aggfunc=lambda x: ', '.join(map(str, set(x)))).reset_index()
    
    # Puanları toplama
    score_pivot = df.groupby("Keyword")["Score"].sum().reset_index()
    
    # Null olmayan Rank değerlerini sayma
    rank_count = df.groupby("Keyword")["Rank"].count().reset_index()
    rank_count.rename(columns={"Rank": "Rank Count"}, inplace=True)
    
    # Puanları ve Rank sayısını tabloya ekleme
    pivot_df = pivot_df.merge(score_pivot, on="Keyword", how="left")
    pivot_df = pivot_df.merge(rank_count, on="Keyword", how="left")
    pivot_df = pivot_df.merge(df[["Keyword", "Missing Keywords"]].drop_duplicates(), on="Keyword", how="left")
    
    # Sütun adlarını güncelle (Application Id'leri doğrudan koru)
    pivot_df.columns = ["Keyword", "Volume"] + list(pivot_df.columns[2:-3]) + ["Total Score", "Rank Count", "Missing Keywords"]
    
    # Boş değerleri null olarak değiştir
    pivot_df = pivot_df.fillna("null")
    
    st.write("### En Çok Tekrar Eden Kelimeler")
    st.write("**Monogram (Tek Kelimeler)**")
    st.write(monograms_filtered.most_common(10))
    
    st.write("**Bigram (İki Kelimeli Öbekler)**")
    st.write([' '.join(bigram) for bigram, _ in bigrams_filtered.most_common(10)])
    
    st.write("**Trigram (Üç Kelimeli Öbekler)**")
    st.write([' '.join(trigram) for trigram, _ in trigrams_filtered.most_common(10)])
    
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

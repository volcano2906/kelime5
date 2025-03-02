import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
from collections import Counter
import nltk
from deep_translator import GoogleTranslator

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

# Çeviri önbelleği (daha hızlı çeviri için)
translation_cache = {}

def translate_word(word):
    """ Translates a word only if it's not already in the cache. """
    if word in translation_cache:
        return translation_cache[word]
    
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(word)
        translation_cache[word] = translated
        return translated
    except:
        return "Translation Error"

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

    # Önce tüm benzersiz kelimeleri toplayarak tek seferde çeviri yap
    unique_keywords = df["Keyword"].unique()
    translations = {word: translate_word(word) for word in unique_keywords}

    # İngilizce çeviri ekle
    df["English Translation"] = df["Keyword"].map(translations)

    # Veriyi uygun formata dönüştürme
    pivot_df = df.pivot_table(
        index=["Keyword", "English Translation", "Volume"], 
        columns="Application Id", 
        values="Rank", 
        aggfunc='first'
    ).reset_index()
    
    # Puanları toplama ve Rank sayısını hesaplama
    summary_df = df.groupby("Keyword").agg(
        Total_Score=("Score", "sum"),
        Rank_Count=("Rank", "count")
    ).reset_index()

    # İngilizce çevirileri ekle
    summary_df["English Translation"] = summary_df["Keyword"].map(translations)

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

    ### **Frekans Analizi** ###
    st.subheader("Anahtar Kelime Frekans Analizi")

    # Filtreleme seçenekleri
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

    # Frekansları hesapla
    word_freq = pd.DataFrame(Counter(all_words).items(), columns=["Word", "Frequency"])
    bigram_freq = pd.DataFrame(Counter(all_bigrams).items(), columns=["Bigram", "Frequency"])
    trigram_freq = pd.DataFrame(Counter(all_trigrams).items(), columns=["Trigram", "Frequency"])

    # **Tek seferde çeviri yaparak hızlandır**
    all_unique_terms = set(word_freq["Word"]) | set(bigram_freq["Bigram"]) | set(trigram_freq["Trigram"])
    term_translations = {term: translate_word(term) for term in all_unique_terms}

    # Çevirileri ekleyelim
    word_freq["English"] = word_freq["Word"].map(term_translations)
    bigram_freq["English"] = bigram_freq["Bigram"].map(term_translations)
    trigram_freq["English"] = trigram_freq["Trigram"].map(term_translations)

    # Sonuçları yatay olarak gösterme
    st.write("### Kelime Frekans Analizi")

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

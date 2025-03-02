import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from collections import Counter
from itertools import islice
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

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

# Tekrar eden kelimeleri yeniden hesaplama seçeneği
recalculate_treemap = st.checkbox("Tekrar Eden Kelimeleri Yeniden Hesapla")

if uploaded_files:
    # Dosyaları oku ve birleştir
    df_list = [pd.read_csv(file) for file in uploaded_files]
    df = pd.concat(df_list, ignore_index=True)
    df = df.drop_duplicates()
    
    # Anahtar kelime hacmi 5 olanları filtrele
    if drop_low_volume:
        df = df[df["Volume"] != 5]
    
    # Parent-Child analizi
    if "Keyword" in df.columns:
        keywords = df["Keyword"].dropna().unique()
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(keywords)
        n_clusters = max(3, int(len(keywords) ** 0.5))
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit(X.toarray())
        keyword_clusters = pd.DataFrame({"Keyword": keywords, "Cluster": clustering.labels_})
        parent_nodes = keyword_clusters.groupby("Cluster")["Keyword"].apply(lambda x: x.value_counts().idxmax())
        parent_child_mapping = keyword_clusters.merge(parent_nodes.rename("Parent_Node"), on="Cluster")
        
        # Sonuçları göster
        st.subheader("Detected Parent Nodes & Child Keywords")
        st.dataframe(parent_child_mapping)
        csv_data = parent_child_mapping.to_csv(index=False).encode('utf-8')
        st.download_button("Download Parent-Child Mapping CSV", csv_data, "parent_nodes.csv", "text/csv")
    
    # Sonuçları gösterme
    st.write("### Dönüştürülmüş Veri Tablosu ve Puanlar")
    st.dataframe(df, use_container_width=True)
    
    # CSV olarak indirme butonu
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Dönüştürülmüş CSV'yi İndir",
        data=csv,
        file_name="converted_keywords_with_scores.csv",
        mime="text/csv"
    )

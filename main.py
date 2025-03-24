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

# Show the uploader inside the placeholder
uploaded_files = st.file_uploader("CSV dosyanızı yükleyin", type=["csv"], accept_multiple_files=True)

# Kullanıcıdan 4 Title, 4 Subtitle ve KW girişi
st.subheader("Anahtar Kelime Karşılaştırma")
col1, col2 = st.columns([1, 2])

title1 = col1.text_input("Title 1 (Maksimum 30 karakter)", max_chars=30)
subtitle1 = col1.text_input("Subtitle 1 (Maksimum 30 karakter)", max_chars=30)

kw_input = col2.text_input("Keyword Alanı (Maksimum 400 karakter, space veya comma ile ayırın)", max_chars=400)
long_description = col2.text_input("Long Description (Maksimum 4000 karakter)", max_chars=4000)

# Girilen kelimeleri temizle ve set olarak sakla
user_input_text = f"{title1} {subtitle1} {kw_input} {long_description}".lower()
user_input_text = re.sub(r'[^a-zA-Z\s]', ' ', user_input_text).strip()
user_words = re.split(r'[ ,]+', user_input_text)
user_words = {word for word in user_words if word and word not in stop_words}

# Create a placeholder for the uploader
uploader_placeholder = st.empty()


# Anahtar kelime hacmi 5 olanları filtreleme seçeneği
drop_low_volume = st.checkbox("Exclude Keywords with Volume 5")
drop_rank_more = st.checkbox("Exclude Keywords with Rank More Than 11")
drop_rank_count = st.checkbox("Exclude When Rank Count with 1")

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
    dfCopyAnaliz=df.copy()
    
    # Anahtar kelime hacmi 5 olanları filtrele
    if drop_low_volume:
        df = df[df["Volume"] != 5]

    if drop_rank_more:
        df = df[df["Rank"] < 11]
    
    # Rank değerlerini sayıya çevir ve puan hesapla
    df["Rank"] = df["Rank"].fillna("250").astype(str)
    df["Score"] = df["Rank"].apply(update_rank)
    
    # Eksik kelimeleri bul
    #def find_missing_keywords(keyword):
     #   words = set(re.split(r'[ ,]+', keyword.lower()))
      #  missing_words = words - user_words
      #  return ','.join(missing_words) if missing_words else "-"


    def find_missing_keywords(keyword):
        words = set(re.split(r'[ ,]+', keyword.lower()))
        missing_words = {word for word in words - user_words if word not in stop_words}
        return ','.join(missing_words) if missing_words else "-"    
        # Eksik kelimeleri bul
    def check_exact_match(keyword):
        # Regex ile exact match kontrolü yap
        pattern = r'(^|[\s,])' + re.escape(keyword) + r'($|[\s,])'
        return "Yes" if re.search(pattern, user_input_text) else "No"

    df["Missing Keywords"] = df["Keyword"].apply(find_missing_keywords)

    
    # Step 3: Apply to DataFrame
    
    

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
    pivot_df["Exact Match"] = pivot_df["Keyword"].apply(check_exact_match)

    if drop_rank_count:
       pivot_df = pivot_df[pivot_df["Rank_Count"] != 1]
    # Boş değerleri "null" olarak değiştir
    pivot_df.fillna("null", inplace=True)
        # Kolonları yeniden sıralama

    #
    competitor_count = df["Application Id"].nunique()
    keyword_rank_counts = df.groupby("Keyword")["Application Id"].nunique()
    keywords_in_all_competitors = keyword_rank_counts[keyword_rank_counts == competitor_count].index.tolist()
    unique_words = set()
    for keyword in keywords_in_all_competitors:
            words = re.split(r'\s+', keyword.lower())  # Split by spaces
            unique_words.update([word for word in words if word not in stop_words])

    # Convert unique words to a comma-separated string
    result_string = ", ".join(sorted(unique_words))
    # Display result
    st.write(result_string)

    # Step: Generate extra words per keyword
    def find_extra_words_not_in_shared_set(keyword, reference_words):
        keyword_words = set(re.split(r'\s+', keyword.lower()))
        keyword_words = {w for w in keyword_words if w and w not in stop_words}
        not_in_result = keyword_words - reference_words
        return ', '.join(sorted(not_in_result)) if not_in_result else "-"
    
    # Apply on original df
    df["missFromCommon"] = df["Keyword"].apply(
        lambda k: find_extra_words_not_in_shared_set(k, unique_words)
    )
    
    # Merge into pivot_df
    pivot_df = pivot_df.merge(
        df[["Keyword", "missFromCommon"]].drop_duplicates(),
        on="Keyword",
        how="left"
    )

    
    first_columns = ["Keyword","Volume", "Total_Score", "Rank_Count", "Missing_Keywords", "Exact Match","missFromCommon"]
    remaining_columns = [col for col in pivot_df.columns if col not in first_columns]
    pivot_df = pivot_df[first_columns + remaining_columns]
    for col in pivot_df.columns[7:]:  # İlk 2 sütun (Keyword, Volume) hariç diğerlerine uygula
        pivot_df[col] = pd.to_numeric(pivot_df[col], errors='coerce').fillna(250).astype(int)

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
    keyword_filter_text = st.text_input("Include only keywords containing (case-insensitive):", "")

    # 1. Clean original keywords from df for exact match lookup
    df["Keyword_cleaned"] = df["Keyword"].str.lower().str.replace(r'[^a-zA-Z\\s]', '', regex=True).str.strip()
    df["Keyword_cleaned"] = df["Keyword_cleaned"].str.replace(r'\\s+', ' ', regex=True)
    
    volume_lookup = df[["Keyword_cleaned", "Volume"]].drop_duplicates()
    

    
    # 2. Filtreleme uygulama
    freq_df = dfCopyAnaliz.copy()
    if exclude_low_volume_freq:
        freq_df = freq_df[freq_df["Volume"] != 5]
    if exclude_single_app_keywords:
        freq_df = freq_df[freq_df.groupby("Keyword")["Application Id"].transform("nunique") > 1]
    if keyword_filter_text:
        freq_df = freq_df[freq_df["Keyword"].str.contains(keyword_filter_text, case=False, na=False)]
    
    # 3. Word splitting functions
    def extract_words(text):
        words = re.split(r'[ ,]+', text.lower())
        return [word.strip() for word in words if word and word not in stop_words]
    
    def extract_ngrams(text, n):
        words = extract_words(text)
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    # 4. Extract all keywords
    all_words = []
    all_bigrams = []
    all_trigrams = []
    
    for keyword in freq_df["Keyword"]:
        words = extract_words(keyword)
        all_words.extend(words)
        all_bigrams.extend(extract_ngrams(keyword, 2))
        all_trigrams.extend(extract_ngrams(keyword, 3))
    
    # 5. Clean ngrams the same way as df["Keyword_cleaned"]
    def clean_ngram(ngram):
        return re.sub(r'[^a-zA-Z\\s]', '', ngram.lower()).strip()
    
    def find_missing_items(keyword):
        words = set(re.split(r'[ ,]+', keyword.lower()))
        missing_words = {word for word in words - user_words if word not in stop_words}
        return ','.join(missing_words) if missing_words else "-"
    
    # 6. Create cleaned version of ngrams for matching
    word_freq = pd.DataFrame(Counter(all_words).items(), columns=["Word", "Frequency"])
    word_freq["Keyword_cleaned"] = word_freq["Word"].apply(clean_ngram)
    word_freq = word_freq.merge(volume_lookup, how="left", on="Keyword_cleaned")
    word_freq["Volume"] = word_freq["Volume"].fillna("none")
    word_freq["Missing Keywords"] = word_freq["Word"].apply(find_missing_items)
    word_freq.drop(columns=["Keyword_cleaned"], inplace=True)
    
    bigram_freq = pd.DataFrame(Counter(all_bigrams).items(), columns=["Bigram", "Frequency"])
    bigram_freq["Keyword_cleaned"] = bigram_freq["Bigram"].apply(clean_ngram)
    bigram_freq = bigram_freq.merge(volume_lookup, how="left", on="Keyword_cleaned")
    bigram_freq["Volume"] = bigram_freq["Volume"].fillna("none")
    bigram_freq["Missing Keywords"] = bigram_freq["Bigram"].apply(find_missing_items)
    bigram_freq.drop(columns=["Keyword_cleaned"], inplace=True)
    
    trigram_freq = pd.DataFrame(Counter(all_trigrams).items(), columns=["Trigram", "Frequency"])
    trigram_freq["Keyword_cleaned"] = trigram_freq["Trigram"].apply(clean_ngram)
    trigram_freq = trigram_freq.merge(volume_lookup, how="left", on="Keyword_cleaned")
    trigram_freq["Volume"] = trigram_freq["Volume"].fillna("none")
    trigram_freq["Missing Keywords"] = trigram_freq["Trigram"].apply(find_missing_items)
    trigram_freq.drop(columns=["Keyword_cleaned"], inplace=True)

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


    # Step 1: Get shared words across all competitors (same as before)
    dfCommonAnaliz=df.copy()
    competitor_count = df["Application Id"].nunique()
    keyword_rank_counts = df.groupby("Keyword")["Application Id"].nunique()
    keywords_in_all_competitors = keyword_rank_counts[keyword_rank_counts == competitor_count].index.tolist()
    # Add Rank_Count column to df (based on how many times each keyword appears)
    df["Rank_Count"] = df.groupby("Keyword")["Application Id"].transform("count")
    shared_words = set()
    for keyword in keywords_in_all_competitors:
        words = re.split(r'\s+', keyword.lower())
        shared_words.update([word for word in words if word and word not in stop_words])
    
    def get_miss_from_common(keyword, shared_words):
        keyword_words = set(re.split(r'\s+', keyword.lower()))
        keyword_words = {w for w in keyword_words if w and w not in stop_words}
        return keyword_words - shared_words
    
    # Generate result string per app
    app_results = {}
    
    for app_id in df["Application Id"].unique():
        app_df = df[df["Application Id"] == app_id]
        app_word_set = set(shared_words)  # start with shared words
    
        for _, row in app_df.iterrows():
            if int(row["Rank"]) != 250 and row["Rank_Count"] != 1:
                keyword = row["Keyword"]
                miss_words = get_miss_from_common(keyword, shared_words)
                app_word_set.update(miss_words)
        
        app_results[app_id] = ", ".join(sorted(app_word_set))
    
    # Display result
    st.write("### Result Strings by Competitor (Application Id)")
    for app_id, word_string in app_results.items():
        words = word_string.split(", ")
        highlighted_words = [
            f"<span style='color:green'>{word}</span>" if word in user_words else word
            for word in words
        ]
        highlighted_string = ", ".join(highlighted_words)
        st.markdown(f"**{app_id}**: {highlighted_string}", unsafe_allow_html=True)



# Anaiz2

    target_app_id = st.text_input("Enter Application ID to inspect keywords and ranks", "")
    pivot_df.columns = pivot_df.columns.astype(str)
    # Proceed only if target ID is valid
    if target_app_id and target_app_id.strip() in pivot_df.columns:
        target_app_id = target_app_id.strip()
    
        # Step 1: Get keywords where this app has Rank = 250
        keywords_with_250 = pivot_df[pivot_df[target_app_id] == 250]["Keyword"]
    
        st.write("🎯 Keywords with Rank = 250 from pivot_df:")
        st.write(keywords_with_250)
    
        # Step 2: Extract words from those keywords
        app_250_words = set()
        for kw in keywords_with_250:
            words = re.split(r'\s+', kw.lower())
            app_250_words.update([w for w in words if w and w not in stop_words])
    
        # Step 3: Get words from app_results[target_app_id] if available
        existing_app_words = set()
        if target_app_id in app_results:
            existing_app_words = set(re.split(r'[,\s]+', app_results[target_app_id].lower()))
            existing_app_words = {w for w in existing_app_words if w and w not in stop_words}
    
        # Step 4: Find new relevant words
        st.write(existing_app_words)
        new_common_words = app_250_words & user_words - existing_app_words
        
        # Step 5: Display
        if new_common_words:
            st.success("✅ Common words (not already in app_results):")
            st.write(", ".join(new_common_words))
        else:
            st.warning("🚫 No new common words found.")
    else:
        if target_app_id:
            st.warning("❌ Application ID not found in pivot_df columns.")

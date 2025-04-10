import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
from collections import Counter
import nltk
import chardet
from collections import defaultdict

# Stopwords'leri yükle
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Sayfa ayarlarını tam ekran yap
st.set_page_config(layout="wide")

# Başlık
st.title("Uygulama ID'lerine Göre Rank Edilmiş Anahtar Kelimeler ve Puanlama")

# Show the uploader inside the placeholder
#uploaded_files = st.file_uploader("CSV dosyanızı yükleyin", type=["csv"], accept_multiple_files=True)
uploaded_files = st.file_uploader(
    "Dosyanızı yükleyin (.csv veya .xlsx destekleniyor)", 
    type=["csv", "xlsx"], 
    accept_multiple_files=True
)

# Kullanıcıdan 4 Title, 4 Subtitle ve KW girişi
st.subheader("Anahtar Kelime Karşılaştırma")
col1, col2 = st.columns([1, 2])

title1 = col1.text_input("Title 1 (Maksimum 30 karakter)", max_chars=30)
subtitle1 = col1.text_input("Subtitle 1 (Maksimum 30 karakter)", max_chars=30)

kw_input = col2.text_input("Keyword Alanı (Maksimum 400 karakter, space veya comma ile ayırın)", max_chars=400)
long_description = col2.text_input("Long Description (Maksimum 4000 karakter)", max_chars=4000)

# Girilen kelimeleri temizle ve set olarak sakla
user_input_text = f"{title1} {subtitle1} {kw_input} {long_description}".lower()
user_input_text = re.sub(r'[^\w\s]', ' ', user_input_text, flags=re.UNICODE).strip()
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
    df_list = []
    
    for file in uploaded_files:
        if file.name.endswith(".csv"):
            # Dosyanın encoding'ini otomatik algıla
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result["encoding"] if result["encoding"] else "utf-8"
    
            # Dosyayı tekrar oku (çünkü read() yukarıda bitirdi)
            file.seek(0)
            try:
                df_read = pd.read_csv(file, encoding=encoding)
            except Exception as e:
                st.error(f"CSV dosyası okunamadı: {file.name} ({str(e)})")
                continue
    
        elif file.name.endswith(".xlsx"):
            try:
                df_read = pd.read_excel(file, engine="openpyxl")
            except Exception as e:
                st.error(f"Excel dosyası okunamadı: {file.name} ({str(e)})")
                continue
    
        else:
            st.warning(f"❌ Desteklenmeyen dosya formatı: {file.name}")
            continue
    
        df_list.append(df_read)
    
    # Dosyaları birleştir
    if df_list:
        df = pd.concat(df_list, ignore_index=True).drop_duplicates()
    else:
        st.stop()
    dfCopyAnaliz=df.copy()
    
    # Anahtar kelime hacmi 5 olanları filtrele
    if drop_low_volume:
        df = df[df["Volume"] != 5]

    if drop_rank_more:
        df = df[df["Rank"] < 11]
    
    # Rank değerlerini sayıya çevir ve puan hesapla
    df["Rank"] = df["Rank"].fillna("250").astype(str)
    df["Score"] = df["Rank"].apply(update_rank)

    # 1️⃣ Kullanıcıdan exact match için filtre kelimeleri al — key ekliyoruz
    exclude_exact_words_raw = st.text_input(
        "❌ Exact Match ile Elemek İstediğiniz Kelimeler (boşlukla ayırın)", 
        "", 
        key="exact_filter_input"
    )
    
    # 2️⃣ Eğer kullanıcı bir şey girdiyse → hem virgül hem boşluğa göre böl
    if exclude_exact_words_raw.strip():
        # 🔁 Re ile split: boşluk, virgül, virgül+boşluk
        exclude_words = set(
            word.strip().lower()
            for word in re.split(r'[,\s]+', exclude_exact_words_raw)
            if word.strip()
        )
    
        def contains_excluded_word(keyword, exclude_set):
            keyword_words = set(keyword.lower().split())
            return not keyword_words.isdisjoint(exclude_set)
    
        # 3️⃣ Uygula (df üzerinde)
        df = df[
            ~df["Keyword"].astype(str).apply(lambda kw: contains_excluded_word(kw, exclude_words))
        ]
    
        st.success(f"❌ Filtrelenen kelimeler (tam eşleşme): {', '.join(exclude_words)}")
    else:
        st.info("ℹ️ Exact match filtresi uygulanmadı. Kelime girilmedi.")


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
    # 1️⃣ Tüm rakiplerde geçen anahtar kelimeleri bul
    competitor_count = df["Application Id"].nunique()
    keyword_rank_counts = df.groupby("Keyword")["Application Id"].nunique()
    keywords_in_all_competitors = keyword_rank_counts[keyword_rank_counts == competitor_count].index.tolist()
    
    # 2️⃣ unique_words seti oluştur (stopwords hariç)
    unique_words = set()
    for keyword in keywords_in_all_competitors:
        words = re.split(r'\s+', keyword.lower())  # boşluklara göre ayır
        unique_words.update([word for word in words if word and word not in stop_words])
    
    # 3️⃣ user_words ile karşılaştırıp renkli hale getir
    highlighted_result_words = []
    for word in sorted(unique_words):
        if word in user_words:
            highlighted_result_words.append(f"<span style='color:green'>{word}</span>")
        else:
            highlighted_result_words.append(word)
    
    # 4️⃣ result_string oluştur (renkli)
    result_string = ", ".join(highlighted_result_words)
    
    # 5️⃣ ekranda göster
    st.markdown("📌 Ortak Kelimeler (Tüm Rakiplerde Geçenler)")
    st.markdown(result_string, unsafe_allow_html=True)

    # unique_words içindeki her kelime için df'de arama (duplikatsız)
    word_to_keywords = {}
    
    for word in unique_words:
        # Anahtar kelimelerde geçenleri bul (case insensitive)
        matching_rows = df[
            df["Keyword"].str.contains(rf'\b{re.escape(word)}\b', flags=re.IGNORECASE, regex=True)
            & (df["Volume"] > 5)
        ]
    
        if not matching_rows.empty:
            # Duplikatsız olarak (keyword, volume) çiftlerini set'e al
            entries = {
                f'{row["Keyword"]} ({int(row["Volume"])})'
                for _, row in matching_rows.iterrows()
            }
            word_to_keywords[word] = sorted(entries)
    
    # Gösterim
    st.write("📌 Kelime Geçen Anahtar Kelimeler ve Hacimleri (App Count, Volume, A-Z)")

    for word in sorted(word_to_keywords.keys()):
        display_word = f"<span style='color:green'>{word}</span>" if word in user_words else word
    
        entries = []
        for kw_text in word_to_keywords[word]:
            keyword_only = re.sub(r'\s*\(\d+\)$', '', kw_text).strip().lower()
            matches = df[df["Keyword"].str.lower() == keyword_only]
            if not matches.empty:
                volume = matches["Volume"].iloc[0]
                app_count = matches["Application Id"].nunique()
                entries.append({
                    "keyword": keyword_only,
                    "volume": volume,
                    "app_count": app_count
                })
    
        # 🔁 Sıralama: app_count > volume > A-Z
        sorted_entries = sorted(
            entries,
            key=lambda x: (-x["app_count"], -x["volume"], x["keyword"])
        )
    
        # Gösterim
        highlighted_keywords = []
        for item in sorted_entries:
            words = item["keyword"].split()
            highlighted_words = [
                f"<span style='color:green'>{w}</span>" if w in user_words else w
                for w in words
            ]
            label = f"{' '.join(highlighted_words)} ({item['volume']}, {item['app_count']})"
            highlighted_keywords.append(label)
    
        if highlighted_keywords:
            st.markdown(
                f"<b>{display_word}</b> → {', '.join(highlighted_keywords)}",
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"<span style='color:gray'>{display_word}</span> → eşleşme bulunamadı.", unsafe_allow_html=True)
   

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
        # 🔍 Her satırda user_words'ten kaç kelime geçtiğini hesapla
    def count_user_word_matches(keyword, user_words_set):
        keyword_lower = keyword.lower()
        return sum(1 for w in user_words_set if w in keyword_lower)
    
    # ⚡ Uygula
    pivot_df["matchCount"] = pivot_df["Keyword"].astype(str).apply(
        lambda kw: count_user_word_matches(kw, user_words)
    )

    
    first_columns = ["Keyword","Volume", "Total_Score", "Rank_Count", "Missing_Keywords", "Exact Match","missFromCommon","matchCount"]
    remaining_columns = [col for col in pivot_df.columns if col not in first_columns]
    pivot_df = pivot_df[first_columns + remaining_columns]
    for col in pivot_df.columns[8:]:  # İlk 2 sütun (Keyword, Volume) hariç diğerlerine uygula
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
    # Daha temiz ve eşleşebilir versiyon
    df["Keyword_cleaned"] = df["Keyword"].astype(str).str.lower()
    df["Keyword_cleaned"] = df["Keyword_cleaned"].str.replace(r"[^\w\s]", "", regex=True).str.strip()
    df["Keyword_cleaned"] = df["Keyword_cleaned"].str.replace(r"\s+", " ", regex=True)
    
    volume_lookup = df[["Keyword_cleaned", "Volume"]].drop_duplicates()
    

    
    # 2. Filtreleme uygulama
    #freq_df = dfCopyAnaliz.copy()
    freq_df = df.copy()
    if exclude_low_volume_freq:
        freq_df = freq_df[freq_df["Volume"] != 5]
    if exclude_single_app_keywords:
        freq_df = freq_df[freq_df.groupby("Keyword")["Application Id"].transform("nunique") > 1]
    if keyword_filter_text:
        keyword_filters = [kw.strip().lower() for kw in re.split(r'[,\n]+', keyword_filter_text) if kw.strip()]
        pattern = '|'.join([re.escape(kw) for kw in keyword_filters])  # regex pattern: baby|baby generator|kids
        freq_df = freq_df[freq_df["Keyword"].str.lower().str.contains(pattern, na=False)]
    
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
    def clean_ngram(text):
        return re.sub(r"[^\w\s]", "", text.lower()).strip()
    
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
   # 1️⃣ Filter Volume > 5
    df_filtered = df[df["Volume"] <= 5].copy()
    df_filtered["Keyword"] = df_filtered["Keyword"].astype(str)
    
    # 2️⃣ Yardımcı fonksiyonlar
    def extract_words(text):
        return re.findall(r'\b\w+\b', text.lower())
    
    def rank_to_score(rank):
        try:
            rank = int(float(rank))
        except:
            rank = 250
        if 1 <= rank <= 10:
            return 0.9
        elif 11 <= rank <= 20:
            return 0.8
        elif 21 <= rank <= 40:
            return 0.7
        elif 41 <= rank <= 60:
            return 0.5
        else:
            return 0.3
    
    # 3️⃣ Skorları topla
    competitor_word_scores = defaultdict(lambda: defaultdict(list))
    
    for _, row in df_filtered.iterrows():
        app_id = row["Application Id"]
        keyword = row["Keyword"]
        words = extract_words(keyword)
        score = rank_to_score(row["Rank"])
        
        for word in words:
            competitor_word_scores[app_id][word].append(score)
    
    # 4️⃣ HTML ile sıralı ve yeşil vurgulu çıktı hazırla
    app_word_result = {}
    
    for app_id, word_dict in competitor_word_scores.items():
        word_scores = []
        for word, scores in word_dict.items():
            avg_score = round(sum(scores) / len(scores), 3)
            word_scores.append((word, avg_score))
        
        # Skora göre sırala (büyükten küçüğe)
        word_scores.sort(key=lambda x: -x[1])
        
        # Görsel çıktı için hazırla
        display_items = []
        for word, avg_score in word_scores:
            display_word = f"<span style='color:green'>{word}</span>" if word in user_words else word
            display_items.append(f"{display_word} ({avg_score})")
        
        app_word_result[app_id] = ", ".join(display_items)
    
    # 5️⃣ Gösterim
    st.write("### 🔢 Kelimeler (Skora Göre Azalan) – User Word'ler Yeşil")
    for app_id, word_dict in competitor_word_scores.items():
        word_scores = []
        for word, scores in word_dict.items():
            avg_score = round(sum(scores) / len(scores), 3)
            count = len(scores)
            word_scores.append((word, avg_score, count))
            if word == "davetiyesi":
               st.write(word_dict)
        
        # Skora göre sırala (büyükten küçüğe)
        word_scores.sort(key=lambda x: -x[1])
    
        # Hazırla ve user_words'te varsa yeşil boya
        display_items = []
        for word, avg_score, count in word_scores:
            word_display = f"<span style='color:green'>{word}</span>" if word in user_words else word
            display_items.append(f"{word_display} ({avg_score} / {count})")
        
        st.markdown(f"**{app_id}** → {', '.join(display_items)}", unsafe_allow_html=True)




    st.subheader("🔍 User Words Analizi: Hangi Kelimelerle Birlikte Geçiyor? (Sadece 2 ve 3Kelimelik Keyword'ler)")
    for user_word in sorted(user_words):
        # 1. user_word içeren 2-3 kelimelik keyword'leri filtrele
        filtered_df = df[df["Keyword"].str.contains(rf'\b{re.escape(user_word)}\b', case=False, regex=True)]
        filtered_df = filtered_df[filtered_df["Keyword"].str.split().str.len().isin([2, 3])]
    
        # 2. En az 2 farklı app'te rank edilenleri bul
        app_counts = filtered_df.groupby("Keyword")["Application Id"].nunique()
        valid_keywords = app_counts[app_counts > 1].index.tolist()
        filtered_df = filtered_df[filtered_df["Keyword"].isin(valid_keywords)]
    
        # 3. Frekansları say
        keyword_list = filtered_df["Keyword"].str.lower().tolist()
        keyword_freq = Counter(keyword_list)
    
        # 4. Frekansa göre gruplama yap
        freq_groups = defaultdict(list)
        for kw, freq in keyword_freq.items():
            freq_groups[freq].append(kw)
    
        # 5. Grupları büyükten küçüğe sırala, içindekileri A-Z sırala
        grouped_output = []
        for freq in sorted(freq_groups.keys(), reverse=True):
            group_words = sorted(freq_groups[freq])
            # user_words içindekileri yeşile boya
            highlighted = []
            for word in group_words:
                parts = [
                    f"<span style='color:green'>{w}</span>" if w in user_words else w
                    for w in word.split()
                ]
                highlighted.append(" ".join(parts))
            grouped_output.append(f"{freq} ({', '.join(highlighted)})")
    
        # 6. Final çıktı
        if grouped_output:
            st.markdown(
                f"<b><span style='color:green'>{user_word}</span></b> → {', '.join(grouped_output)}",
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"<span style='color:gray'>{user_word}</span> → eşleşme bulunamadı.", unsafe_allow_html=True)

    
    # Anaiz2
    previousMeta = st.text_input("Please write previous all metadata", "")
    user_input_text_2 = f"{previousMeta}".lower()
    user_input_text_2 = re.sub(r'[^\w\s]', ' ', user_input_text_2,flags=re.UNICODE).strip()
    user_words_2 = re.split(r'[ ,]+', user_input_text_2)
    user_words_2 = {word for word in user_words_2 if word and word not in stop_words}
    target_app_id = st.text_input("Enter Application ID to inspect keywords and ranks", "")
    pivot_df.columns = pivot_df.columns.astype(str)
    # Proceed only if target ID is valid
    if target_app_id and target_app_id.strip() in pivot_df.columns:
        target_app_id = target_app_id.strip()
    
        # Step 1: Get keywords where this app has Rank = 250
        keywords_with_250 = pivot_df[pivot_df[target_app_id] == 250]["Keyword"]

    
        # Step 2: Extract words from those keywords
        app_250_words = set()
        for kw in keywords_with_250:
            words = re.split(r'\s+', kw.lower())
            app_250_words.update([w for w in words if w and w not in stop_words])
    
        # Step 3: Get words from app_results[target_app_id] if available
        existing_app_words = set()
        app_results = {str(app_id): result for app_id, result in app_results.items()}
        if target_app_id in app_results:
            result_str = app_results[target_app_id].lower()
            existing_app_words = set(re.split(r'[,\s]+', result_str))
            existing_app_words = {w for w in existing_app_words if w and w not in stop_words}
    
        # Step 4: Find new relevant words
        new_common_words = app_250_words & user_words_2 - existing_app_words
        
        # Step 5: Display
        if new_common_words:
            st.success("✅ Used but not ranked:")
            st.write(", ".join(new_common_words))
        else:
            st.warning("🚫 No new common words found.")
    else:
        if target_app_id:
            st.warning("❌ Application ID not found in pivot_df columns.")


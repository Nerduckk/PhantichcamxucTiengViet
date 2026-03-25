import pandas as pd
import json
import re
from underthesea import word_tokenize

# 1. Load các bộ từ điển của bạn
with open('resources/teencode.json', 'r', encoding='utf-8') as f:
    teencode_dict = json.load(f)

with open('resources/stopword.json', 'r', encoding='utf-8') as f:
    stopword_data = json.load(f)
    # Stopwords thực sự = (danh sách xóa) trừ đi (danh sách giữ cho cảm xúc)
    final_stopwords = set(stopword_data['remove_always']) - set(stopword_data['keep_for_sentiment'])

def clean_text(text):
    if not isinstance(text, str): return ""
    
    # Lowercase & Xóa ký tự đặc biệt
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text) # Chỉ giữ lại chữ cái và số
    text = re.sub(r'\s+', ' ', text).strip() # Xóa khoảng trắng thừa

    # Thay thế Teencode
    words = text.split()
    words = [teencode_dict.get(w, w) for w in words]
    
    # Tách từ (Word Segmentation)
    text = " ".join(words)
    text = word_tokenize(text, format="text") # ví dụ: 'sản phẩm' -> 'sản_phẩm'
    
    # Xóa Stopwords
    words = text.split()
    words = [w for w in words if w not in final_stopwords]
    
    return " ".join(words)


def preprocess_pipeline(input_path, output_path_csv, output_path_txt):
    df = pd.read_csv(input_path)
    print(f"Đang xử lý file: {input_path}")
    
    # 1. Áp dụng hàm làm sạch (giữ nguyên logic của bạn)
    df['comments_clean'] = df['comments'].apply(clean_text)
    
    # 2. Loại bỏ các dòng bị trống sau khi làm sạch 
    df = df.dropna(subset=['comments_clean'])
    df = df[df['comments_clean'].str.strip() != ""]

    # 3. Lưu file .csv để dùng cho Notebook/Pandas
    df.to_csv(output_path_csv, index=False, encoding='utf-8-sig')
    
    # 4. Tạo định dạng FastText: __label__<flag> <văn_bản>
    df_fasttext = "__label__" + df['flag'].astype(str) + " " + df['comments_clean']
    
    # 5. Lưu file .txt dành riêng cho FastText
    df_fasttext.to_csv(output_path_txt, index=False, header=False, encoding='utf-8')
    
    print(f"Đã lưu CSV tại: {output_path_csv}")
    print(f"Đã lưu TXT tại: {output_path_txt}")

# Chạy lại pipeline với 3 tham số
preprocess_pipeline('data/preprocessed/train.csv', 
                    'data/processed/train_cleaned.csv', 
                    'data/processed/train_fasttext.txt')

preprocess_pipeline('data/preprocessed/test.csv', 
                    'data/processed/test_cleaned.csv', 
                    'data/processed/test_fasttext.txt')

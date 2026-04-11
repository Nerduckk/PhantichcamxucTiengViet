import pandas as pd
import json
import re
from underthesea import word_tokenize
from sklearn.model_selection import train_test_split

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


def save_fasttext_format(df, filepath):
    """Hàm hỗ trợ lưu dữ liệu theo định dạng chuẩn của FastText"""
    fasttext_data = "__label__" + df['flag'].astype(str) + " " + df['comments_clean']
    # Lưu file .txt không có header, không có index, không có dấu ngoặc kép
    fasttext_data.to_csv(filepath, index=False, header=False, encoding='utf-8')

def preprocess_pipeline(input_train_path, input_test_path, output_dir):
    # 1. Xử lý tập Train gốc (để chia thành Train mới và Valid)
    df_train_full = pd.read_csv(input_train_path)
    print(f"Đang xử lý và chia tập Train: {input_train_path}")
    df_train_full['comments_clean'] = df_train_full['comments'].apply(clean_text)
    
    # Loại bỏ dòng trống sau khi làm sạch
    df_train_full = df_train_full.dropna(subset=['comments_clean'])
    df_train_full = df_train_full[df_train_full['comments_clean'].str.strip() != ""]

    # Chia 90/10 để lấy tập Validation phục vụ Autotune
    df_train_new, df_val = train_test_split(df_train_full, test_size=0.1, random_state=42)

    # 2. Xử lý tập Test gốc
    df_test = pd.read_csv(input_test_path)
    print(f"Đang xử lý tập Test: {input_test_path}")
    df_test['comments_clean'] = df_test['comments'].apply(clean_text)
    df_test = df_test.dropna(subset=['comments_clean'])

    # 3. Lưu toàn bộ kết quả ra thư mục data/processed/
    # Lưu các file CSV (để soi lỗi và EDA)
    df_train_new.to_csv(f'{output_dir}/train_new_cleaned.csv', index=False, encoding='utf-8-sig')
    df_val.to_csv(f'{output_dir}/val_cleaned.csv', index=False, encoding='utf-8-sig')
    df_test.to_csv(f'{output_dir}/test_cleaned.csv', index=False, encoding='utf-8-sig')

    # Lưu các file TXT (để nạp vào FastText Autotune)
    save_fasttext_format(df_train_new, f'{output_dir}/train_fasttext.txt')
    save_fasttext_format(df_val, f'{output_dir}/val_fasttext.txt')
    save_fasttext_format(df_test, f'{output_dir}/test_fasttext.txt')

    print(f"Hoàn tất! Tất cả file đã sẵn sàng trong {output_dir}")

if __name__ == "__main__":
    preprocess_pipeline(
        input_train_path='data/preprocessed/train.csv', 
        input_test_path='data/preprocessed/test.csv',
        output_dir='data/processed'
    )

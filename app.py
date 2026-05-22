import fasttext
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# -------------------------------------------------------------------
# LOAD MÔ HÌNH THẬT (Chỉ load 1 lần khi khởi động app)
# -------------------------------------------------------------------
@st.cache_resource # Lệnh này giúp Streamlit lưu model vào RAM, không bị load lại mỗi khi nhập chữ
def load_models():
    # 1. Load FastText
    # Thay bằng đường dẫn file .bin của bạn
    ft_model = fasttext.load_model("models/fasttext_v1.bin") 
    
    # 2. Load PhoBERT
    # Thay bằng thư mục chứa model PhoBERT sau khi train xong
    phobert_path = "models/phobert_finetuned" 
    tokenizer = AutoTokenizer.from_pretrained(phobert_path)
    phobert_model = AutoModelForSequenceClassification.from_pretrained(phobert_path)
    
    return ft_model, tokenizer, phobert_model

# Khởi tạo mô hình
ft_model, phobert_tokenizer, phobert_model = load_models()

# -------------------------------------------------------------------
# HÀM DỰ ĐOÁN THẬT (Inference)
# -------------------------------------------------------------------
def predict_fasttext_real(text):
    # FastText trả về dạng (('__label__2',), array([0.92]))
    result = ft_model.predict(text)
    label_str = result[0][0].replace('__label__', '')
    prob = result[1][0]
    return int(label_str), prob

def predict_phobert_real(text):
    # PhoBERT cần mã hóa text thành tensor trước khi dự đoán
    inputs = phobert_tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    
    with torch.no_grad(): # Không tính đạo hàm lúc dự đoán để chạy nhanh hơn
        outputs = phobert_model(**inputs)
        
    # Lấy xác suất bằng Softmax
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Lấy nhãn có xác suất cao nhất
    prob, label_id = torch.max(probs, dim=1)
    
    return label_id.item(), prob.item()

# Dưới phần giao diện, bạn chỉ cần gọi predict_fasttext_real() và predict_phobert_real()
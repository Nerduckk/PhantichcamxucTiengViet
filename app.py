import json
import re
from pathlib import Path

import fasttext
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from underthesea import word_tokenize


PROJECT_ROOT = Path(__file__).resolve().parent
FASTTEXT_MODEL_PATH = PROJECT_ROOT / "outputs" / "fasttext" / "fasttext_sentiment.bin"
PHOBERT_MODEL_PATH = PROJECT_ROOT / "outputs" / "phobert-tune-c"
TEENCODE_PATH = PROJECT_ROOT / "resources" / "teencode.json"
STOPWORD_PATH = PROJECT_ROOT / "resources" / "stopword.json"
FASTTEXT_METRICS_PATH = PROJECT_ROOT / "outputs" / "fasttext" / "metrics.json"
PHOBERT_METRICS_PATH = PROJECT_ROOT / "outputs" / "phobert-tune-c" / "metrics.json"
PHOBERT_MAX_LENGTH = 128
SAMPLE_TEXTS = [
    "Quán ăn này khá ổn, phục vụ nhanh và lịch sự.",
    "Đồ ăn nguội, nhân viên khó chịu, mình sẽ không quay lại nữa.",
    "Giao hàng đúng hẹn, đóng gói cẩn thận.",
]

LABEL_DISPLAY = {
    0: "Tiêu cực",
    1: "Trung lập",
    2: "Tích cực",
}
LABEL_COLORS = {
    0: "#ff6b6b",
    1: "#ffd166",
    2: "#06d6a0",
}


@st.cache_resource
def load_text_resources():
    with open(TEENCODE_PATH, "r", encoding="utf-8") as file:
        teencode_dict = json.load(file)
    with open(STOPWORD_PATH, "r", encoding="utf-8") as file:
        stopword_data = json.load(file)
    final_stopwords = set(stopword_data["remove_always"]) - set(
        stopword_data["keep_for_sentiment"]
    )
    return teencode_dict, final_stopwords


def clean_text(text: str) -> str:
    teencode_dict, final_stopwords = load_text_resources()
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [teencode_dict.get(word, word) for word in text.split()]
    text = word_tokenize(" ".join(words), format="text")
    words = [word for word in text.split() if word not in final_stopwords]
    return " ".join(words).strip()


@st.cache_resource
def load_models():
    missing_paths = []
    if not FASTTEXT_MODEL_PATH.exists():
        missing_paths.append(str(FASTTEXT_MODEL_PATH))
    if not PHOBERT_MODEL_PATH.exists():
        missing_paths.append(str(PHOBERT_MODEL_PATH))

    if missing_paths:
        raise FileNotFoundError(
            "Khong tim thay model tai:\n- " + "\n- ".join(missing_paths)
        )

    ft_model = fasttext.load_model(str(FASTTEXT_MODEL_PATH))
    tokenizer = AutoTokenizer.from_pretrained(str(PHOBERT_MODEL_PATH))
    phobert_model = AutoModelForSequenceClassification.from_pretrained(
        str(PHOBERT_MODEL_PATH)
    )
    phobert_model.eval()
    return ft_model, tokenizer, phobert_model


@st.cache_data
def load_demo_metrics():
    metrics = {}
    if FASTTEXT_METRICS_PATH.exists():
        metrics["fasttext"] = json.loads(FASTTEXT_METRICS_PATH.read_text(encoding="utf-8"))
    if PHOBERT_METRICS_PATH.exists():
        phobert_data = json.loads(PHOBERT_METRICS_PATH.read_text(encoding="utf-8"))
        metrics["phobert"] = phobert_data.get("test", {})
    return metrics


def predict_fasttext_real(ft_model, text: str):
    labels, probs = ft_model.predict(text, k=3)
    label_probs = {}
    for label, prob in zip(labels, probs):
        label_id = int(label.replace("__label__", ""))
        label_probs[label_id] = float(prob)
    top_label = max(label_probs, key=label_probs.get)
    return top_label, label_probs[top_label], label_probs


def predict_phobert_real(tokenizer, phobert_model, text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=PHOBERT_MAX_LENGTH,
    )

    with torch.no_grad():
        outputs = phobert_model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prob, label_id = torch.max(probs, dim=1)
    label_probs = {
        idx: float(score) for idx, score in enumerate(probs[0].detach().cpu().tolist())
    }
    return label_id.item(), prob.item(), label_probs


def build_probability_table(label_probs: dict[int, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Nhãn": LABEL_DISPLAY[label_id],
                "Xác suất": round(score, 4),
            }
            for label_id, score in sorted(label_probs.items())
        ]
    )


def format_pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.2%}"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .hero-card {
            background: linear-gradient(135deg, #1f2937 0%, #111827 65%, #0f172a 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            padding: 28px;
            margin-bottom: 18px;
        }
        .hero-title {
            font-size: 2.35rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 10px;
        }
        .hero-text {
            color: #d1d5db;
            font-size: 1rem;
            line-height: 1.7;
            margin-bottom: 0;
        }
        .mini-card {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 18px 18px 14px;
            min-height: 148px;
        }
        .mini-card h4 {
            margin: 0 0 8px 0;
            font-size: 1rem;
        }
        .mini-card p {
            margin: 0;
            color: #d1d5db;
            line-height: 1.6;
            font-size: 0.95rem;
        }
        .section-label {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            color: #dbeafe;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .result-pill {
            display: inline-block;
            padding: 8px 12px;
            border-radius: 999px;
            font-weight: 700;
            color: #0b1020;
            margin-bottom: 8px;
        }
        .summary-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_probability_bars(label_probs: dict[int, float]) -> None:
    for label_id, score in sorted(label_probs.items()):
        st.write(f"{LABEL_DISPLAY[label_id]}: **{score:.2%}**")
        st.progress(min(max(score, 0.0), 1.0))


def render_prediction(
    title: str,
    label_id: int,
    score: float,
    label_probs: dict[int, float],
    overall_accuracy: float | None,
):
    st.subheader(title)
    color = LABEL_COLORS.get(label_id, "#e5e7eb")
    st.markdown(
        f"<div class='result-pill' style='background:{color};'>Nhãn dự đoán: {LABEL_DISPLAY.get(label_id, label_id)}</div>",
        unsafe_allow_html=True,
    )
    st.write(f"Độ tin cậy của câu này: **{score:.4f}**")
    if overall_accuracy is not None:
        st.caption(f"Độ chính xác trên tập test: {overall_accuracy:.2%}")
    render_probability_bars(label_probs)
    with st.expander("Xem bảng xác suất chi tiết"):
        st.dataframe(
            build_probability_table(label_probs),
            hide_index=True,
            use_container_width=True,
        )


def render_overview(metrics: dict) -> None:
    st.markdown("<div class='section-label'>Tổng quan dự án</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">So sánh FastText và PhoBERT cho bài toán phân tích cảm xúc tiếng Việt</div>
            <p class="hero-text">
                Ứng dụng này minh họa một pipeline AI/ML đơn giản nhưng đủ rõ ràng để báo cáo:
                tiền xử lý dữ liệu, huấn luyện hai mô hình, đánh giá trên tập test và suy luận trực tiếp
                trên câu người dùng nhập vào. FastText đóng vai trò baseline gọn, còn PhoBERT là mô hình
                ngữ cảnh mạnh hơn để cải thiện chất lượng dự đoán.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("FastText Accuracy", format_pct(metrics.get("fasttext", {}).get("accuracy")))
    with metric_cols[1]:
        st.metric("PhoBERT Accuracy", format_pct(metrics.get("phobert", {}).get("accuracy")))
    with metric_cols[2]:
        phobert = metrics.get("phobert", {})
        st.metric("PhoBERT Macro F1", format_pct(phobert.get("macro_f1")))


def render_model_cards(metrics: dict) -> None:
    st.markdown("<div class='section-label'>Hai hướng tiếp cận</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="mini-card">
                <h4>FastText Baseline</h4>
                <p>
                    Mô hình nhẹ, huấn luyện nhanh, phù hợp làm mốc so sánh ban đầu.
                    Accuracy trên tập test hiện tại là <strong>{format_pct(metrics.get("fasttext", {}).get("accuracy"))}</strong>.
                    Điểm mạnh là tốc độ và độ ổn định khi demo.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="mini-card">
                <h4>PhoBERT Fine-tuning</h4>
                <p>
                    Mô hình transformer hiểu ngữ cảnh tốt hơn, dùng để nâng chất lượng dự đoán.
                    Accuracy trên tập test hiện tại là <strong>{format_pct(metrics.get("phobert", {}).get("accuracy"))}</strong>,
                    cao hơn baseline và phù hợp để trình bày phần mô hình chính.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_pipeline_section() -> None:
    st.markdown("<div class='section-label'>Quy trình xử lý</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    steps = [
        ("1. Tiền xử lý", "Chuẩn hóa chữ thường, thay teencode, tách từ và làm sạch văn bản."),
        ("2. Chia dữ liệu", "Dùng cùng một tập train, validation, test cho cả hai mô hình để so sánh công bằng."),
        ("3. Huấn luyện", "FastText làm baseline, PhoBERT được fine-tune trên dữ liệu tiếng Việt."),
        ("4. Suy luận", "App nhận câu mới, tiền xử lý giống lúc train và trả về nhãn cùng xác suất."),
    ]
    for col, (title, body) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="mini-card">
                    <h4>{title}</h4>
                    <p>{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_demo_intro() -> None:
    st.markdown("<div class='section-label'>Khu vực trải nghiệm</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="summary-card">
            <strong>Cách đọc kết quả:</strong> nhãn là dự đoán cuối cùng của mô hình, còn
            <em>độ tin cậy của câu này</em> là xác suất riêng cho câu bạn vừa nhập, không phải độ chính xác chung trên tập test.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_final_summary(ft_label: int, phobert_label: int, metrics: dict) -> None:
    better_model = "PhoBERT" if metrics.get("phobert", {}).get("accuracy", 0) >= metrics.get("fasttext", {}).get("accuracy", 0) else "FastText"
    st.markdown("<div class='section-label'>Nhận xét nhanh</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="summary-card">
            <p style="margin:0; line-height:1.7;">
                Với câu vừa nhập, <strong>FastText</strong> dự đoán là <strong>{LABEL_DISPLAY[ft_label]}</strong> và
                <strong>PhoBERT</strong> dự đoán là <strong>{LABEL_DISPLAY[phobert_label]}</strong>.
                Xét trên toàn bộ tập test, <strong>{better_model}</strong> đang là mô hình nhỉnh hơn trong project này.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Demo Phân Tích Cảm Xúc", page_icon="💬")
    inject_styles()
    metrics = load_demo_metrics()
    render_overview(metrics)
    render_model_cards(metrics)
    render_pipeline_section()

    try:
        ft_model, phobert_tokenizer, phobert_model = load_models()
    except Exception as exc:
        st.error(f"Khong the load model: {exc}")
        st.stop()

    if "input_text" not in st.session_state:
        st.session_state.input_text = "Quán ăn này khá ổn, phục vụ nhanh và lịch sự."

    render_demo_intro()
    st.markdown("**Câu mẫu để demo nhanh**")
    sample_cols = st.columns(len(SAMPLE_TEXTS))
    for index, sample_text in enumerate(SAMPLE_TEXTS):
        with sample_cols[index]:
            if st.button(f"Mẫu {index + 1}", use_container_width=True):
                st.session_state.input_text = sample_text

    text = st.text_area(
        "Nhập câu cần phân tích cảm xúc",
        key="input_text",
        height=120,
    )

    if st.button("Dự đoán", type="primary"):
        cleaned_text = clean_text(text)
        if not cleaned_text:
            st.warning("Nội dung sau khi làm sạch đang rỗng. Hãy nhập câu đầy đủ hơn.")
            st.stop()

        ft_label, ft_score, ft_probs = predict_fasttext_real(ft_model, cleaned_text)
        phobert_label, phobert_score, phobert_probs = predict_phobert_real(
            phobert_tokenizer, phobert_model, cleaned_text
        )

        st.caption(f"Văn bản sau khi tiền xử lý: `{cleaned_text}`")

        col1, col2 = st.columns(2)
        with col1:
            render_prediction(
                "FastText",
                ft_label,
                ft_score,
                ft_probs,
                metrics.get("fasttext", {}).get("accuracy"),
            )
        with col2:
            render_prediction(
                "PhoBERT",
                phobert_label,
                phobert_score,
                phobert_probs,
                metrics.get("phobert", {}).get("accuracy"),
            )

        render_final_summary(ft_label, phobert_label, metrics)

        st.info(
            "Lưu ý: 'Độ tin cậy của câu này' là xác suất của mô hình cho đúng câu bạn vừa nhập, "
            "không phải độ chính xác chung của mô hình."
        )


if __name__ == "__main__":
    main()

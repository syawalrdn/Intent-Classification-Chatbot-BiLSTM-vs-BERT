import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from streamlit_option_menu import option_menu


st.set_page_config(
    page_title="Intent Classification Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
BILSTM_DIR = BASE_DIR / "models" / "bilstm"
BERT_DIR = BASE_DIR / "models" / "bert"

BILSTM_MODEL_PATH = BILSTM_DIR / "bilstm_model.keras"
BILSTM_TOKENIZER_PATH = BILSTM_DIR / "bilstm_tokenizer.json"
BILSTM_META_PATH = BILSTM_DIR / "label_mapping.json"

BERT_META_PATH = BERT_DIR / "label_mapping.json"

MAX_LEN_DEFAULT = 32
MAX_BERT_LEN_DEFAULT = 64


def preprocess_text(text: str) -> str:
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=True)
def load_bilstm_assets():
    if not BILSTM_MODEL_PATH.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {BILSTM_MODEL_PATH}")
    if not BILSTM_TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {BILSTM_TOKENIZER_PATH}")
    if not BILSTM_META_PATH.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {BILSTM_META_PATH}")

    model = load_model(BILSTM_MODEL_PATH)

    with open(BILSTM_TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())

    meta = load_json(BILSTM_META_PATH)
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    max_len = int(meta.get("max_len", MAX_LEN_DEFAULT))

    return model, tokenizer, id2label, max_len


@st.cache_resource(show_spinner=True)
def load_bert_assets():
    if not BERT_DIR.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {BERT_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR)

    if BERT_META_PATH.exists():
        meta = load_json(BERT_META_PATH)
        max_bert_len = int(meta.get("max_bert_len", MAX_BERT_LEN_DEFAULT))
    else:
        max_bert_len = MAX_BERT_LEN_DEFAULT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return model, tokenizer, max_bert_len, device


def predict_bilstm_topk(text: str, top_k: int = 5):
    model, tokenizer, id2label, max_len = load_bilstm_assets()

    clean_text = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([clean_text])
    pad = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    probs = model.predict(pad, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]

    results = [{"intent": id2label[int(idx)], "confidence": float(probs[idx])} for idx in top_indices]

    return {
        "predicted_intent": results[0]["intent"],
        "confidence": results[0]["confidence"],
        "top_k": results,
    }


def predict_bert_topk(text: str, top_k: int = 5):
    model, tokenizer, max_bert_len, device = load_bert_assets()

    clean_text = preprocess_text(text)
    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_bert_len,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu().numpy()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    top_indices = np.argsort(probs)[::-1][:top_k]

    results = [{"intent": id2label[int(idx)], "confidence": float(probs[idx])} for idx in top_indices]

    return {
        "predicted_intent": results[0]["intent"],
        "confidence": results[0]["confidence"],
        "top_k": results,
    }


def show_prediction_block(title: str, result: dict):
    st.markdown(f"### {title}")
    st.metric("Predicted intent", result["predicted_intent"])
    st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
    st.dataframe(pd.DataFrame(result["top_k"]), use_container_width=True, hide_index=True)


with st.sidebar:
    st.markdown("## 🤖 Menu")

    page = option_menu(
        menu_title=None,
        options=["Tentang Chatbot", "Chatbot"],
        icons=["info-circle", "chat-dots"],
        default_index=0,
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "#f8fafc",
            },
            "icon": {
                "color": "#2563eb",
                "font-size": "18px",
            },
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "6px 0",
                "padding": "12px 14px",
                "border-radius": "10px",
                "color": "#111827",
                "--hover-color": "#e5e7eb",
            },
            "nav-link-selected": {
                "background-color": "#dbeafe",
                "color": "#1d4ed8",
                "font-weight": "600",
            },
        },
    )

    st.markdown("---")
    st.caption("Intent classification\nBiLSTM vs BERT")

st.title("🤖 Intent Classification Chatbot")
st.caption("Perbandingan prediksi BiLSTM dan BERT untuk data in-scope.")

if page == "Tentang Chatbot":
    st.subheader("Deskripsi")
    st.write(
        "Aplikasi ini membandingkan dua model intent classification, yaitu BiLSTM dan BERT. "
        "Keduanya memprediksi intent dari kalimat yang dimasukkan pengguna, lalu menampilkan "
        "confidence score dan top-k intent paling mungkin."
    )

    st.subheader("Cara menggunakan")
    st.markdown(
        '''
        1. Pilih menu **Chatbot** di sidebar kiri.  
        2. Ketik kalimat pada kolom input.  
        3. Tekan **Kirim**.  
        4. Lihat hasil prediksi **BiLSTM** dan **BERT** beserta confidence-nya.  
        5. Riwayat percakapan akan tetap tampil selama sesi masih aktif.  
        '''
    )

    st.subheader("Contoh kalimat input")
    st.markdown(
    """
    Berikut beberapa contoh kalimat yang bisa Anda coba:

    - **How much money do I have in my bank account?**
    - **Set a timer for 10 minutes.**
    - **Translate thank you to French.**
    - **What time is it in London?**
    - **Remind me to call my mom tomorrow.**
    """
    )

    st.subheader("Catatan penting")
    st.info(
        "Model ini dilatih hanya pada data in-scope. Jika Anda memasukkan pertanyaan di luar domain, "
        "model tetap akan memilih intent yang paling mirip, bukan label out-of-scope."
    )

    with st.expander("File model yang harus tersedia"):
        st.code(
            '''
models/
├── bilstm/
│   ├── bilstm_model.keras
│   ├── bilstm_tokenizer.json
│   └── label_mapping.json
└── bert/
    ├── config.json
    ├── model.safetensors  (atau pytorch_model.bin)
    ├── tokenizer.json / vocab.txt / tokenizer_config.json
    └── label_mapping.json   (opsional, untuk max_bert_len)
            '''
        )

else:
    st.subheader("Uji Chatbot")

    if "history" not in st.session_state:
        st.session_state.history = []

    col_a, col_b = st.columns([4, 1])
    with col_a:
        user_input = st.text_input("Masukkan kalimat", placeholder="Contoh: set a timer for 10 minutes")
    with col_b:
        top_k = st.number_input("Top-K", min_value=1, max_value=10, value=5, step=1)

    col_send, col_clear = st.columns([1, 1])
    send_clicked = col_send.button("Kirim", use_container_width=True)
    clear_clicked = col_clear.button("Hapus riwayat", use_container_width=True)

    if clear_clicked:
        st.session_state.history = []
        st.rerun()

    if send_clicked and user_input.strip():
        bilstm_result = predict_bilstm_topk(user_input, int(top_k))
        bert_result = predict_bert_topk(user_input, int(top_k))

        st.session_state.history.append(
            {
                "text": user_input,
                "bilstm": bilstm_result,
                "bert": bert_result,
            }
        )

    if not st.session_state.history:
        st.info("Belum ada input. Masukkan kalimat lalu klik Kirim.")
    else:
        for i, item in enumerate(reversed(st.session_state.history), start=1):
            st.markdown("---")
            st.markdown(f"#### Input #{len(st.session_state.history) - i + 1}")
            st.write(item["text"])

            col1, col2 = st.columns(2)
            with col1:
                show_prediction_block("BiLSTM", item["bilstm"])
            with col2:
                show_prediction_block("BERT", item["bert"])

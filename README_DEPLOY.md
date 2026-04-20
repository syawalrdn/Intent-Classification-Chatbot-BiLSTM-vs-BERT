# Deploy Chatbot Intent Classification (In-Scope)

Project ini berisi aplikasi **Streamlit** untuk membandingkan prediksi **BiLSTM** dan **BERT** pada task **intent classification in-scope**.

## Struktur file

```text
.
├── app.py
├── requirements.txt
└── models
    ├── bilstm
    │   ├── bilstm_model.keras
    │   ├── bilstm_tokenizer.json
    │   └── label_mapping.json
    └── bert
        ├── config.json
        ├── model.safetensors  (atau pytorch_model.bin)
        ├── tokenizer files
        └── label_mapping.json  (opsional)
```

## Ekspor model dari notebook training

### 1) Simpan BiLSTM

Jalankan setelah training BiLSTM selesai:

```python
import json
from pathlib import Path

deploy_dir = Path("models/bilstm")
deploy_dir.mkdir(parents=True, exist_ok=True)

bilstm_model.save(deploy_dir / "bilstm_model.keras")

with open(deploy_dir / "bilstm_tokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer_bilstm.to_json())

with open(deploy_dir / "label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "id2label": {str(k): v for k, v in id2label.items()},
            "label2id": label2id,
            "max_len": MAX_LEN
        },
        f,
        ensure_ascii=False,
        indent=2
    )
```

### 2) Simpan BERT

Jalankan setelah training BERT selesai:

```python
import json
from pathlib import Path

deploy_dir = Path("models/bert")
deploy_dir.mkdir(parents=True, exist_ok=True)

trainer.save_model(deploy_dir)
bert_tokenizer.save_pretrained(deploy_dir)

with open(deploy_dir / "label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "id2label": {str(k): v for k, v in id2label.items()},
            "label2id": label2id,
            "max_bert_len": MAX_BERT_LEN
        },
        f,
        ensure_ascii=False,
        indent=2
    )
```

## Jalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy online

Opsi termudah untuk app seperti ini adalah **Streamlit Community Cloud** karena platform ini memang untuk deploy aplikasi Streamlit, terhubung ke GitHub, dan cocok untuk aplikasi pendidikan/non-komersial. Hugging Face Spaces juga mendukung hosting demo ML dan dokumentasinya menyediakan dukungan untuk Streamlit Spaces. citeturn909804search4turn909804search0turn909804search1turn909804search3

### Langkah cepat di Streamlit Community Cloud
1. Upload project ini ke repository GitHub.
2. Pastikan `app.py`, `requirements.txt`, dan folder `models/` ikut ter-push.
3. Buka Streamlit Community Cloud.
4. Pilih **Create app**.
5. Pilih repository, branch, dan file entrypoint `app.py`.
6. Deploy. Streamlit akan membuat URL di domain `streamlit.app`. citeturn909804search0turn909804search14

## Catatan
Karena ini model **in-scope only**, input di luar domain tetap akan dipetakan ke intent paling mirip, bukan `oos`.

# PhantichcamxucTiengViet

Project phan tich cam xuc tieng Viet don gian cho bai toan 3 nhan:

- `0`: tieu cuc
- `1`: tich cuc
- `2`: trung lap

Repo hien co 2 huong:

- FastText: du lieu da duoc lam sach va xuat ra `.txt` tai `data/processed/`
- PhoBERT: pipeline train/evaluate/predict toi gian trong `src/`
- Notebook 1 file: `notebooks/phobert_sentiment_pipeline.ipynb` co ca PhoBERT va FastText

## Cau truc chinh

- `src/preprocess.py`: lam sach du lieu, thay teencode, tach tu, xoa stopword
- `src/train_phobert.py`: train PhoBERT tren file da preprocess
- `src/evaluate_phobert.py`: danh gia model tren tap test
- `src/predict_phobert.py`: du doan nhanh 1 cau
- `data/preprocessed/`: train/test goc
- `data/processed/`: du lieu da lam sach, san sang cho FastText va PhoBERT

## Cai dat

```bash
pip install -r requirements.txt
```

## 1. Tien xu ly du lieu

Neu can tao lai file cleaned:

```bash
python src/preprocess.py
```

Sau buoc nay se co:

- `data/processed/train_new_cleaned.csv`
- `data/processed/val_cleaned.csv`
- `data/processed/test_cleaned.csv`

## 2. Train PhoBERT

Lenh mac dinh:

```bash
python src/train_phobert.py
```

Lenh co tham so:

```bash
python src/train_phobert.py ^
  --train-file data/processed/train_new_cleaned.csv ^
  --val-file data/processed/val_cleaned.csv ^
  --output-dir outputs/phobert-sentiment ^
  --epochs 2 ^
  --batch-size 8 ^
  --learning-rate 2e-5
```

Model mac dinh la `vinai/phobert-base`.

## 3. Danh gia

```bash
python src/evaluate_phobert.py ^
  --model-dir outputs/phobert-sentiment ^
  --test-file data/processed/test_cleaned.csv
```

Script se in ra:

- accuracy
- macro f1
- weighted f1
- classification report

## 4. Du doan nhanh

```bash
python src/predict_phobert.py ^
  --model-dir outputs/phobert-sentiment ^
  --text "quan an nay kha on, phuc vu nhanh va lich su"
```

## Ghi chu

- PhoBERT hoat dong tot hon khi input da duoc tach tu, vi vay script train mac dinh doc cot `comments_clean`.
- Neu may yeu GPU, hay giam `--batch-size` xuong `4` hoac `2`.
- De scope do an gon nhe, pipeline hien tai uu tien huan luyen, danh gia va demo du doan thay vi toi uu sau.

# %% [markdown]
# # Twitter Sentiment Analysis bằng BERT Tiny
#
# Notebook này dùng riêng dataset Kaggle:
#
# https://www.kaggle.com/datasets/dunyajasim/twitter-dataset-for-sentiment-analysis
#
# Mục tiêu:
#
# - Đọc dữ liệu Twitter sentiment.
# - Lấy cột text và cột sentiment.
# - Làm sạch text ở mức đơn giản.
# - Fine-tune BERT tiny để phân loại cảm xúc/sentiment.
# - Đánh giá bằng Accuracy, F1-score, classification report và confusion matrix.

# %% [markdown]
# ## 1. Import thư viện

# %%
import os
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup


# %% [markdown]
# ## 2. Cố định seed
#
# Seed giúp kết quả ổn định hơn giữa các lần chạy.

# %%
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# %% [markdown]
# ## 3. Kiểm tra dataset trên Kaggle
#
# Sau khi Add Input dataset trên Kaggle, dữ liệu thường nằm trong:
#
# ```text
# /kaggle/input/datasets/dunyajasim/twitter-dataset-for-sentiment-analysis
# ```

# %%
print("Các folder trong /kaggle/input:")
if Path("/kaggle/input").exists():
    print(os.listdir("/kaggle/input"))
else:
    print("Không chạy trên Kaggle hoặc chưa có /kaggle/input")

data_root = Path("/kaggle/input/datasets/dunyajasim/twitter-dataset-for-sentiment-analysis")

if not data_root.exists():
    print("\nKhông thấy path mặc định:", data_root)
    print("Đang thử các path thường gặp khác...")

    possible_roots = [
        Path("/kaggle/input/twitter-dataset-for-sentiment-analysis"),
    ]

    if Path("/kaggle/input").exists():
        for path in Path("/kaggle/input").rglob("*"):
            if path.is_dir() and "twitter" in path.name.lower():
                possible_roots.append(path)

    possible_roots = [path for path in possible_roots if path.exists()]

    if len(possible_roots) == 0:
        raise FileNotFoundError(
            "Chưa tìm thấy dataset. Hãy Add Input dataset trên Kaggle trước."
        )

    data_root = possible_roots[0]

print("Dataset root đang dùng:", data_root)


# %% [markdown]
# ## 4. Đọc file dữ liệu
#
# Dataset này có thể chứa file `.xlsx` thay vì `.csv`.
# Cell này in ra các file bảng tìm thấy rồi chọn file đầu tiên.
#
# Nếu Kaggle của bạn có nhiều file và bạn muốn dùng file khác, chỉ cần sửa biến `data_path`.

# %%
table_files = []
for extension in ["*.csv", "*.xlsx", "*.xls"]:
    table_files.extend(list(data_root.rglob(extension)))

table_files = sorted(table_files)

print("Các file dữ liệu tìm thấy:")
for i, path in enumerate(table_files):
    print(i, "->", path)

if len(table_files) == 0:
    raise FileNotFoundError("Không tìm thấy file CSV/XLSX/XLS trong dataset.")

data_path = table_files[0]
print("\nFile đang dùng:", data_path)

if data_path.suffix.lower() == ".csv":
    df = pd.read_csv(data_path)
else:
    df = pd.read_excel(data_path)

print("Shape:", df.shape)
print("Columns:")
print(list(df.columns))
df.head()


# %% [markdown]
# ## 5. Chọn cột text và cột label
#
# Với dataset Twitter sentiment, các tên cột hay gặp là:
#
# - Text: `text`, `tweet`, `clean_text`, `Tweet_content`, `Caption`
# - Label: `sentiment`, `category`, `label`, `Sentiment`, `LABEL`
#
# Nếu cell này chọn sai, bạn sửa trực tiếp `text_column` và `label_column`.

# %%
def find_column(columns, candidates):
    lower_to_original = {str(col).lower(): col for col in columns}

    for name in candidates:
        if name.lower() in lower_to_original:
            return lower_to_original[name.lower()]

    for col in columns:
        col_lower = str(col).lower()
        for name in candidates:
            if name.lower() in col_lower:
                return col

    return None


text_column = find_column(
    df.columns,
    ["Caption", "text", "tweet", "clean_text", "tweet_content", "Tweet_content", "content", "sentence"],
)

label_column = find_column(
    df.columns,
    ["LABEL", "sentiment", "category", "label", "target", "class", "polarity"],
)

print("text_column :", text_column)
print("label_column:", label_column)

if text_column is None or label_column is None:
    raise ValueError(
        "Không tự tìm được cột text/label. Hãy xem danh sách Columns ở trên rồi sửa text_column, label_column."
    )

raw_label_values = df[label_column].dropna().unique()
print("\nMột số label gốc:")
print(raw_label_values[:20])

numeric_labels = pd.to_numeric(df[label_column], errors="coerce").dropna()
numeric_label_set = set(numeric_labels.astype(int).unique())
print("Numeric label set:", numeric_label_set)


# %% [markdown]
# ## 6. Chuẩn hóa dữ liệu
#
# Cell này tạo dataframe mới chỉ gồm:
#
# ```text
# text, label_name
# ```
#
# Text được làm sạch nhẹ:
#
# - Bỏ link URL.
# - Bỏ username dạng `@user`.
# - Bỏ khoảng trắng thừa.
#
# Label được chuẩn hóa thành chữ thường.

# %%
def clean_tweet(text):
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_label(label):
    if pd.isna(label):
        return None

    numeric_value = pd.to_numeric(pd.Series([label]), errors="coerce").iloc[0]

    if pd.notna(numeric_value):
        numeric_value = int(numeric_value)

        if numeric_label_set.issubset({0, 1}):
            if numeric_value == 0:
                return "negative"
            if numeric_value == 1:
                return "positive"

        if numeric_label_set.issubset({-1, 0, 1}):
            if numeric_value == -1:
                return "negative"
            if numeric_value == 0:
                return "neutral"
            if numeric_value == 1:
                return "positive"

        if numeric_label_set.issubset({0, 2, 4}):
            if numeric_value == 0:
                return "negative"
            if numeric_value == 2:
                return "neutral"
            if numeric_value == 4:
                return "positive"

    label_text = str(label).strip().lower()

    label_text = label_text.replace(" ", "_")

    if label_text in ["neg", "negative"]:
        return "negative"
    if label_text in ["neu", "neutral"]:
        return "neutral"
    if label_text in ["pos", "positive"]:
        return "positive"

    return label_text


data = pd.DataFrame()
data["text"] = df[text_column].apply(clean_tweet)
data["label_name"] = df[label_column].apply(normalize_label)

data = data.dropna(subset=["text", "label_name"]).reset_index(drop=True)
data = data[data["text"].str.len() > 0].reset_index(drop=True)

print("Số dòng sau khi làm sạch:", len(data))
print(data["label_name"].value_counts())
data.head()


# %% [markdown]
# ## 7. Giới hạn số mẫu để notebook chạy nhanh
#
# Nếu dataset lớn, fine-tune toàn bộ sẽ khá lâu. Để notebook dễ chạy trên Kaggle, ta lấy tối đa 30000 mẫu.
#
# Nếu muốn train toàn bộ dataset, bạn có thể bỏ cell này.

# %%
if len(data) > 30000:
    data = data.sample(n=30000, random_state=42).reset_index(drop=True)

print("Số mẫu dùng để train:", len(data))
print(data["label_name"].value_counts())


# %% [markdown]
# ## 8. Bỏ lớp có quá ít mẫu
#
# Khi chia train/validation có `stratify`, mỗi lớp cần ít nhất 2 mẫu.
# Nếu một lớp chỉ có 1 mẫu, ta bỏ lớp đó để tránh lỗi khi split.

# %%
class_counts = data["label_name"].value_counts()
valid_classes = class_counts[class_counts >= 2].index
data = data[data["label_name"].isin(valid_classes)].reset_index(drop=True)

print("Số mẫu sau khi bỏ lớp quá ít:", len(data))
print(data["label_name"].value_counts())


# %% [markdown]
# ## 9. Mã hóa label thành số
#
# PyTorch cần label dạng số nguyên.
#
# Ví dụ:
#
# ```text
# negative -> 0
# neutral  -> 1
# positive -> 2
# ```

# %%
preferred_order = ["negative", "neutral", "positive", "irrelevant"]
existing_labels = list(data["label_name"].unique())

ordered_labels = []
for label in preferred_order:
    if label in existing_labels:
        ordered_labels.append(label)

for label in sorted(existing_labels):
    if label not in ordered_labels:
        ordered_labels.append(label)

label2id = {label: i for i, label in enumerate(ordered_labels)}
id2label = {i: label for label, i in label2id.items()}

data["label"] = data["label_name"].map(label2id)

print("label2id:", label2id)
data.head()


# %% [markdown]
# ## 10. Chia train/validation

# %%
train_df, val_df = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data["label"],
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("\nTrain label distribution:")
print(train_df["label_name"].value_counts())
print("\nValidation label distribution:")
print(val_df["label_name"].value_counts())


# %% [markdown]
# ## 10. Tokenizer và Dataset class
#
# Model dùng:
#
# ```text
# prajjwal1/bert-tiny
# ```
#
# Đây là bản BERT nhỏ, nhẹ hơn `bert-base-uncased`, phù hợp demo trên Kaggle.

# %%
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")


class TwitterSentimentDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        encoded = tokenizer(
            row["text"],
            max_length=96,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.long),
        }


train_dataset = TwitterSentimentDataset(train_df)
val_dataset = TwitterSentimentDataset(val_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
)

print("Số batch train:", len(train_loader))
print("Số batch validation:", len(val_loader))


# %% [markdown]
# ## 11. Tạo model BERT classifier
#
# `AutoModelForSequenceClassification` gồm:
#
# - BERT encoder để đọc câu tweet.
# - Classification head để dự đoán sentiment.

# %%
model = AutoModelForSequenceClassification.from_pretrained(
    "prajjwal1/bert-tiny",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

total_training_steps = len(train_loader) * 3
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_training_steps,
)

print(model)


# %% [markdown]
# ## 12. Hàm train và evaluate

# %%
def train_one_epoch(model, dataloader):
    model.train()

    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1


def evaluate(model, dataloader):
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1, all_preds, all_labels


# %% [markdown]
# ## 13. Train model

# %%
best_val_f1 = 0

for epoch in range(3):
    print(f"\nEpoch {epoch + 1}/3")

    train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader)
    val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, val_loader)

    print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
    print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f} | Val   F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        model.save_pretrained("/kaggle/working/twitter_sentiment_bert_model")
        tokenizer.save_pretrained("/kaggle/working/twitter_sentiment_bert_model")
        print("Đã lưu model tốt nhất.")

print("\nBest validation macro F1:", best_val_f1)


# %% [markdown]
# ## 14. Báo cáo kết quả

# %%
val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, val_loader)

target_names = [id2label[i] for i in range(len(id2label))]

print("Accuracy:", val_acc)
print("Macro F1:", val_f1)
print("\nClassification report:")
print(classification_report(val_labels, val_preds, target_names=target_names))


# %% [markdown]
# ## 15. Confusion matrix

# %%
cm = confusion_matrix(val_labels, val_preds)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# %% [markdown]
# ## 16. Dự đoán thử một câu tweet

# %%
def predict_sentiment(text):
    model.eval()

    encoded = tokenizer(
        clean_tweet(text),
        max_length=96,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).squeeze(0)
        pred_id = int(torch.argmax(probs).cpu())

    return {
        "text": text,
        "predicted_label": id2label[pred_id],
        "confidence": float(probs[pred_id].cpu()),
    }


predict_sentiment("I love this product, it works really well!")


# %% [markdown]
# ## 17. Nơi lưu model
#
# Model tốt nhất được lưu tại:
#
# ```text
# /kaggle/working/twitter_sentiment_bert_model
# ```
#
# Sau khi chạy xong trên Kaggle, bạn có thể tải folder này trong phần Output.

# %%
print("Model folder:")
print("/kaggle/working/twitter_sentiment_bert_model")

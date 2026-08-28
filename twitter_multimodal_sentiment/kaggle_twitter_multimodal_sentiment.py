# %% [markdown]
# # Twitter Multimodal Sentiment Analysis
#
# Notebook này dùng dataset Kaggle:
#
# https://www.kaggle.com/datasets/dunyajasim/twitter-dataset-for-sentiment-analysis
#
# Dataset có:
#
# - File `.xlsx` chứa text và label.
# - Folder ảnh chứa ảnh tương ứng.
#
# Ý tưởng:
#
# ```text
# Caption -> BERT tiny -> text feature
# Image   -> ResNet18  -> image feature
# text feature + image feature -> fusion -> classifier -> sentiment
# ```

# %% [markdown]
# ## 1. Import thư viện

# %%
import os
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import torchvision.transforms as transforms
from torchvision import models

from transformers import AutoTokenizer, AutoModel


ImageFile.LOAD_TRUNCATED_IMAGES = True


# %% [markdown]
# ## 2. Cố định seed

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
# ## 3. Đường dẫn dataset trên Kaggle
#
# Theo Kaggle của bạn, dataset nằm ở:
#
# ```text
# /kaggle/input/datasets/dunyajasim/twitter-dataset-for-sentiment-analysis
# ```

# %%
data_root = Path("/kaggle/input/datasets/dunyajasim/twitter-dataset-for-sentiment-analysis")

if not data_root.exists():
    print("Không thấy path chính:", data_root)
    print("Đang thử path Kaggle thường gặp khác...")

    candidate_roots = [
        Path("/kaggle/input/twitter-dataset-for-sentiment-analysis"),
    ]

    if Path("/kaggle/input").exists():
        for path in Path("/kaggle/input").rglob("*"):
            if path.is_dir() and "twitter" in path.name.lower():
                candidate_roots.append(path)

    candidate_roots = [path for path in candidate_roots if path.exists()]

    if len(candidate_roots) == 0:
        raise FileNotFoundError("Không tìm thấy dataset. Hãy Add Input dataset trên Kaggle.")

    data_root = candidate_roots[0]

print("Dataset root đang dùng:", data_root)


# %% [markdown]
# ## 4. Đọc file metadata `.xlsx`
#
# Dataset này có file Excel chứa 3 cột quan trọng:
#
# ```text
# File Name, Caption, LABEL
# ```

# %%
table_files = []
for pattern in ["*.xlsx", "*.xls", "*.csv"]:
    table_files.extend(list(data_root.rglob(pattern)))

table_files = sorted(table_files)

print("Các file bảng tìm thấy:")
for index, path in enumerate(table_files):
    print(index, "->", path)

if len(table_files) == 0:
    raise FileNotFoundError("Không tìm thấy file .xlsx/.xls/.csv trong dataset.")

metadata_path = table_files[0]
print("\nMetadata đang dùng:", metadata_path)

if metadata_path.suffix.lower() == ".csv":
    df = pd.read_csv(metadata_path)
else:
    df = pd.read_excel(metadata_path)

print("Shape:", df.shape)
print("Columns:")
print(list(df.columns))
df.head()


# %% [markdown]
# ## 5. Trỏ thẳng vào các cột cần dùng
#
# Dựa trên ảnh bạn gửi, ta dùng trực tiếp:
#
# - `File Name`: tên file text, ví dụ `1.txt`.
# - `Caption`: nội dung tweet/caption.
# - `LABEL`: nhãn sentiment.

# %%
file_column = "File Name"
text_column = "Caption"
label_column = "LABEL"

required_columns = [file_column, text_column, label_column]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Không tìm thấy cột bắt buộc: {column}")

print("Đã tìm thấy đủ cột:", required_columns)


# %% [markdown]
# ## 6. Tìm ảnh tương ứng với từng dòng
#
# Trong metadata, `File Name` có dạng:
#
# ```text
# 1.txt
# 10.txt
# 100.txt
# ```
#
# Còn ảnh có dạng:
#
# ```text
# 1.jpg
# 10.jpg
# 100.jpg
# ```
#
# Vì vậy ta lấy phần tên không đuôi, ví dụ `1`, rồi tìm ảnh có cùng stem.

# %%
image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

all_image_paths = []
for path in data_root.rglob("*"):
    if path.suffix.lower() in image_extensions:
        all_image_paths.append(path)

print("Số ảnh tìm thấy:", len(all_image_paths))

image_path_dict = {}
for path in all_image_paths:
    key = path.stem.lower()
    image_path_dict[key] = str(path)


def get_image_path(file_name):
    if pd.isna(file_name):
        return None

    stem = Path(str(file_name).strip()).stem.lower()
    return image_path_dict.get(stem)


# %% [markdown]
# ## 7. Làm sạch text và label

# %%
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_label(label):
    if pd.isna(label):
        return None

    label = str(label).strip().lower()
    label = label.replace(" ", "_")

    if label in ["negative", "neg"]:
        return "negative"
    if label in ["neutral", "neu"]:
        return "neutral"
    if label in ["positive", "pos"]:
        return "positive"

    return None


data = pd.DataFrame()
data["file_name"] = df[file_column].astype(str)
data["text"] = df[text_column].apply(clean_text)
data["label_name"] = df[label_column].apply(normalize_label)
data["image_path"] = df[file_column].apply(get_image_path)

print("Số dòng ban đầu:", len(data))
print("Số dòng tìm được ảnh:", data["image_path"].notna().sum())
print("Số dòng có label hợp lệ:", data["label_name"].notna().sum())

data = data.dropna(subset=["text", "label_name", "image_path"]).reset_index(drop=True)
data = data[data["text"].str.len() > 0].reset_index(drop=True)

print("Số dòng sau khi bỏ thiếu text/label/ảnh:", len(data))
print(data["label_name"].value_counts())
data.head()


# %% [markdown]
# ## 8. Lọc ảnh lỗi
#
# Một số ảnh trong dataset có thể bị lỗi. Ta kiểm tra trước khi train để DataLoader không bị dừng.

# %%
def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            img.convert("RGB")

        return True
    except Exception:
        return False


before_filter = len(data)
data = data[data["image_path"].apply(is_valid_image)].reset_index(drop=True)
after_filter = len(data)

print("Số ảnh lỗi đã bỏ:", before_filter - after_filter)
print("Số mẫu cuối cùng:", after_filter)
print(data["label_name"].value_counts())


# %% [markdown]
# ## 9. Mã hóa label thành số

# %%
label2id = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

id2label = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

data["label"] = data["label_name"].map(label2id)
data = data.dropna(subset=["label"]).reset_index(drop=True)
data["label"] = data["label"].astype(int)

print(data["label_name"].value_counts())
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
# ## 11. Tokenizer và transform ảnh
#
# - Text dùng tokenizer của `prajjwal1/bert-tiny`.
# - Ảnh resize về `224 x 224` để đưa vào ResNet18.

# %%
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

train_image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

val_image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# %% [markdown]
# ## 12. Dataset class
#
# Mỗi sample gồm:
#
# ```text
# input_ids, attention_mask, image, label
# ```

# %%
class TwitterMultimodalDataset(Dataset):
    def __init__(self, dataframe, image_transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_transform = image_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        encoded_text = tokenizer(
            row["text"],
            max_length=96,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        image = Image.open(row["image_path"]).convert("RGB")
        image = self.image_transform(image)

        return {
            "input_ids": encoded_text["input_ids"].squeeze(0),
            "attention_mask": encoded_text["attention_mask"].squeeze(0),
            "image": image,
            "label": torch.tensor(row["label"], dtype=torch.long),
        }


train_dataset = TwitterMultimodalDataset(train_df, train_image_transform)
val_dataset = TwitterMultimodalDataset(val_df, val_image_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
)

print("Số batch train:", len(train_loader))
print("Số batch validation:", len(val_loader))


# %% [markdown]
# ## 13. Model BERT + ResNet18 + Fusion
#
# Model có 3 phần:
#
# - Text branch: BERT tiny trích xuất đặc trưng text.
# - Image branch: ResNet18 trích xuất đặc trưng ảnh.
# - Fusion classifier: nối hai đặc trưng lại rồi phân loại.
#
# Kỹ thuật fusion trong notebook này là **concat fusion**.

# %%
class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        text_feature_size = self.text_encoder.config.hidden_size

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        image_feature_size = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.image_encoder = resnet

        self.text_projection = nn.Linear(text_feature_size, 128)
        self.image_projection = nn.Linear(image_feature_size, 128)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3),
        )

    def forward(self, input_ids, attention_mask, image):
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        text_cls = text_outputs.last_hidden_state[:, 0, :]
        text_feature = self.text_projection(text_cls)

        image_feature = self.image_encoder(image)
        image_feature = self.image_projection(image_feature)

        fused_feature = torch.cat([text_feature, image_feature], dim=1)
        logits = self.classifier(fused_feature)

        return logits


model = MultimodalSentimentModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

print(model)


# %% [markdown]
# ## 14. Hàm train và evaluate

# %%
def train_one_epoch(model, dataloader):
    model.train()

    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image=image,
        )

        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

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
            image = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=image,
            )

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1, all_preds, all_labels


# %% [markdown]
# ## 15. Train model

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
        torch.save(model.state_dict(), "/kaggle/working/twitter_multimodal_best_model.pt")
        print("Đã lưu model tốt nhất.")

print("\nBest validation macro F1:", best_val_f1)


# %% [markdown]
# ## 16. Báo cáo kết quả

# %%
val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, val_loader)

target_names = [id2label[i] for i in range(3)]

print("Accuracy:", val_acc)
print("Macro F1:", val_f1)
print("\nClassification report:")
print(classification_report(val_labels, val_preds, target_names=target_names))


# %% [markdown]
# ## 17. Confusion matrix

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
# ## 18. Dự đoán thử một mẫu trong validation

# %%
sample = val_df.iloc[0]

model.eval()

encoded_text = tokenizer(
    sample["text"],
    max_length=96,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)

image = Image.open(sample["image_path"]).convert("RGB")
image = val_image_transform(image).unsqueeze(0)

with torch.no_grad():
    logits = model(
        input_ids=encoded_text["input_ids"].to(device),
        attention_mask=encoded_text["attention_mask"].to(device),
        image=image.to(device),
    )
    pred_id = int(torch.argmax(logits, dim=1).cpu())

print("Text:", sample["text"])
print("Image:", sample["image_path"])
print("True label:", sample["label_name"])
print("Predicted label:", id2label[pred_id])


# %% [markdown]
# ## 19. File output
#
# Model tốt nhất được lưu ở:
#
# ```text
# /kaggle/working/twitter_multimodal_best_model.pt
# ```

# %%
print("Model saved to:")
print("/kaggle/working/twitter_multimodal_best_model.pt")

# %% [markdown]
# # Simple Multimodal Sentiment Classification on Memotion Dataset 7k
#
# File này là một bài code Kaggle đơn giản, chỉ dùng **1 dataset**:
#
# ```text
# Memotion Dataset 7k
# ```
#
# Ý tưởng mô hình:
#
# ```text
# Text  -> DistilBERT/BERT -> text feature  \
#                                           concat -> classifier -> sentiment
# Image -> ResNet          -> image feature /
# ```
#
# Mục tiêu của file này là dễ đọc, dễ hiểu, dễ sửa khi chạy trên Kaggle.

# %% [markdown]
# ## 1. Import thư viện

# %%
import os
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

import torchvision.transforms as transforms
from torchvision import models

from transformers import AutoTokenizer, AutoModel


# Một số ảnh trong dataset có thể bị lỗi/truncated.
# Dòng này giúp PIL cố đọc các ảnh đó thay vì dừng chương trình ngay.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# %% [markdown]
# ## 2. Config đơn giản
#
# Nếu Kaggle báo không thấy dataset, hãy chạy:
#
# ```python
# import os
# print(os.listdir("/kaggle/input"))
# ```
#
# rồi sửa lại `DATA_ROOT` cho đúng.

# %%
class CFG:
    seed = 42

    # Đường dẫn mặc định khi Add Input dataset trên Kaggle.
    # Nếu Kaggle của bạn để dataset trong /kaggle/input/datasets/..., hãy sửa lại dòng này.
    DATA_ROOT = Path("/kaggle/input/memotion-dataset-7k")

    # Model text. Dùng DistilBERT để nhẹ hơn BERT-base.
    # Nếu muốn dùng BERT chuẩn, đổi thành: "bert-base-uncased"
    TEXT_MODEL = "distilbert-base-uncased"

    max_len = 96
    image_size = 224
    batch_size = 16
    epochs = 3
    lr = 2e-5
    num_workers = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Device:", CFG.device)


# %% [markdown]
# ## 3. Cố định seed
#
# Cố định seed giúp kết quả ít thay đổi giữa các lần chạy.

# %%
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(CFG.seed)


# %% [markdown]
# ## 4. Kiểm tra dataset trên Kaggle

# %%
print("Các folder trong /kaggle/input:")
if Path("/kaggle/input").exists():
    print(os.listdir("/kaggle/input"))
else:
    print("Không chạy trên Kaggle hoặc chưa có /kaggle/input")

print("\nDATA_ROOT hiện tại:", CFG.DATA_ROOT)
print("DATA_ROOT tồn tại không?", CFG.DATA_ROOT.exists())


# %% [markdown]
# ## 5. Tìm file metadata và folder ảnh
#
# Memotion Dataset 7k thường có file `.csv` hoặc `.xlsx` chứa thông tin:
#
# - tên ảnh
# - text/OCR của meme
# - nhãn sentiment
#
# Để code dễ hiểu, ta tìm tất cả file bảng và in ra cho bạn xem.

# %%
def find_table_files(root):
    table_files = []
    for path in root.rglob("*"):
        if path.suffix.lower() in [".csv", ".xlsx", ".xls"]:
            table_files.append(path)
    return table_files


table_files = find_table_files(CFG.DATA_ROOT)

print("Các file metadata tìm thấy:")
for i, path in enumerate(table_files):
    print(i, "->", path)

if len(table_files) == 0:
    raise FileNotFoundError("Không tìm thấy file metadata. Hãy kiểm tra lại DATA_ROOT.")


# %% [markdown]
# ## 6. Đọc metadata
#
# Code tự chọn file đầu tiên có vẻ phù hợp. Nếu chọn sai, bạn có thể sửa `metadata_path`.

# %%
# Ưu tiên file có chữ "label", "train" hoặc "annotation" trong tên.
best_score = -999
metadata_path = table_files[0]

for path in table_files:
    name = path.name.lower()
    score = 0
    if "label" in name:
        score += 3
    if "train" in name:
        score += 2
    if "annotation" in name:
        score += 2
    if score > best_score:
        best_score = score
        metadata_path = path

print("Metadata được chọn:", metadata_path)

if metadata_path.suffix.lower() == ".csv":
    df = pd.read_csv(metadata_path)
else:
    df = pd.read_excel(metadata_path)

print("Shape:", df.shape)
print("Columns:")
print(list(df.columns))
df.head()


# %% [markdown]
# ## 7. Chọn cột image, text và label
#
# Memotion có thể đặt tên cột hơi khác giữa các bản. Cell này tự đoán cột.
# Nếu tự đoán sai, bạn sửa 3 biến:
#
# ```python
# image_col = "ten_cot_anh"
# text_col = "ten_cot_text"
# label_col = "ten_cot_label"
# ```

# %%
def choose_column(columns, keywords):
    for key in keywords:
        for col in columns:
            if key.lower() in str(col).lower():
                return col
    return None


image_col = choose_column(
    df.columns,
    ["image_name", "image", "file_name", "filename", "img"],
)

text_col = choose_column(
    df.columns,
    ["ocr_text", "ocr", "text", "caption", "sentence"],
)

label_col = choose_column(
    df.columns,
    ["overall_sentiment", "sentiment", "label", "class"],
)

print("image_col:", image_col)
print("text_col :", text_col)
print("label_col:", label_col)

if image_col is None or text_col is None or label_col is None:
    raise ValueError("Không tự đoán được đủ cột. Hãy sửa image_col, text_col, label_col thủ công.")


# %% [markdown]
# ## 8. Tìm đường dẫn ảnh thật
#
# Metadata thường chỉ lưu tên ảnh, ví dụ:
#
# ```text
# image_1.jpg
# ```
#
# Ta cần tìm đường dẫn đầy đủ trong Kaggle, ví dụ:
#
# ```text
# /kaggle/input/memotion-dataset-7k/images/image_1.jpg
# ```

# %%
image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

all_image_paths = []
for path in CFG.DATA_ROOT.rglob("*"):
    if path.suffix.lower() in image_extensions:
        all_image_paths.append(path)

print("Số ảnh tìm thấy:", len(all_image_paths))

# Tạo dictionary để tìm ảnh nhanh.
# Key là tên file ảnh viết thường.
image_path_dict = {}
for path in all_image_paths:
    image_path_dict[path.name.lower()] = str(path)
    image_path_dict[path.stem.lower()] = str(path)


def get_image_path(image_value):
    if pd.isna(image_value):
        return None

    image_value = str(image_value).strip()
    image_name = Path(image_value).name
    image_stem = Path(image_value).stem

    possible_keys = [
        image_value.lower(),
        image_name.lower(),
        image_stem.lower(),
    ]

    for key in possible_keys:
        if key in image_path_dict:
            return image_path_dict[key]

    # Nếu metadata chỉ lưu tên không có đuôi, thử thêm các đuôi ảnh phổ biến.
    for ext in image_extensions:
        key = f"{image_stem.lower()}{ext}"
        if key in image_path_dict:
            return image_path_dict[key]

    return None


# %% [markdown]
# ## 9. Chuẩn hóa label
#
# Memotion có thể có nhãn:
#
# ```text
# very_positive, positive, neutral, negative, very_negative
# ```
#
# Để đơn giản, ta gom về 3 lớp:
#
# ```text
# positive, neutral, negative
# ```

# %%
def normalize_sentiment(label):
    if pd.isna(label):
        return None

    label = str(label).strip().lower()
    label = label.replace("_", " ")

    if label in ["very positive", "positive", "pos"]:
        return "positive"
    if label in ["neutral", "neu"]:
        return "neutral"
    if label in ["very negative", "negative", "neg"]:
        return "negative"

    return None


# %% [markdown]
# ## 10. Tạo dataframe sạch cho bài toán
#
# Sau bước này, dữ liệu chỉ còn 3 cột chính:
#
# ```text
# text, image_path, label_name
# ```

# %%
data = pd.DataFrame()
data["text"] = df[text_col].fillna("").astype(str)
data["image_path"] = df[image_col].apply(get_image_path)
data["label_name"] = df[label_col].apply(normalize_sentiment)

print("Trước khi bỏ mẫu lỗi:", len(data))

# Bỏ mẫu thiếu ảnh hoặc thiếu label.
data = data.dropna(subset=["image_path", "label_name"]).reset_index(drop=True)

print("Sau khi bỏ mẫu thiếu ảnh/label:", len(data))
print(data["label_name"].value_counts())
data.head()


# %% [markdown]
# ## 11. Bỏ ảnh bị lỗi
#
# Một số ảnh trong dataset có thể bị hỏng. Ta kiểm tra trước để tránh lỗi khi train.

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


before = len(data)
data = data[data["image_path"].apply(is_valid_image)].reset_index(drop=True)
after = len(data)

print("Số ảnh lỗi đã bỏ:", before - after)
print("Số mẫu cuối cùng:", after)
print(data["label_name"].value_counts())


# %% [markdown]
# ## 12. Mã hóa label thành số
#
# PyTorch không train trực tiếp với label dạng chữ, nên ta đổi sang số.

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
data.head()


# %% [markdown]
# ## 13. Chia train/validation

# %%
train_df, val_df = train_test_split(
    data,
    test_size=0.2,
    random_state=CFG.seed,
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
# ## 14. Tokenizer và transform ảnh
#
# Text dùng tokenizer của DistilBERT/BERT.
# Image được resize 224x224 và normalize theo ImageNet để hợp với ResNet pretrained.

# %%
tokenizer = AutoTokenizer.from_pretrained(CFG.TEXT_MODEL)

train_image_transform = transforms.Compose(
    [
        transforms.Resize((CFG.image_size, CFG.image_size)),
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
        transforms.Resize((CFG.image_size, CFG.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# %% [markdown]
# ## 15. Dataset class
#
# Class này quyết định mỗi lần lấy một sample thì trả về những gì.
#
# Một sample gồm:
#
# ```text
# input_ids, attention_mask, image, label
# ```

# %%
class MemotionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, image_transform, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        text = row["text"]
        image_path = row["image_path"]
        label = row["label"]

        # Xử lý text bằng tokenizer.
        encoded_text = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Xử lý image.
        # Nếu ảnh lỗi, thay bằng ảnh đen để tránh crash.
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (CFG.image_size, CFG.image_size), color=(0, 0, 0))

        image = self.image_transform(image)

        return {
            "input_ids": encoded_text["input_ids"].squeeze(0),
            "attention_mask": encoded_text["attention_mask"].squeeze(0),
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }


# %% [markdown]
# ## 16. DataLoader

# %%
train_dataset = MemotionDataset(
    dataframe=train_df,
    tokenizer=tokenizer,
    image_transform=train_image_transform,
    max_len=CFG.max_len,
)

val_dataset = MemotionDataset(
    dataframe=val_df,
    tokenizer=tokenizer,
    image_transform=val_image_transform,
    max_len=CFG.max_len,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=CFG.num_workers,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
)

print("Số batch train:", len(train_loader))
print("Số batch validation:", len(val_loader))


# %% [markdown]
# ## 17. Model đơn giản: Text branch + Image branch + Fusion
#
# Model gồm:
#
# - `text_encoder`: DistilBERT/BERT
# - `image_encoder`: ResNet18 pretrained
# - `classifier`: phân loại sau khi concat feature
#
# Dùng ResNet18 thay vì ResNet50 để code chạy nhẹ và dễ demo hơn.
# Nếu muốn mạnh hơn, có thể đổi sang ResNet50.

# %%
class SimpleMultimodalModel(nn.Module):
    def __init__(self, text_model_name, num_classes):
        super().__init__()

        # Branch 1: Text encoder.
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_feature_size = self.text_encoder.config.hidden_size

        # Branch 2: Image encoder.
        # Dùng ResNet18 pretrained để nhẹ hơn ResNet50.
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        image_feature_size = resnet.fc.in_features

        # Bỏ classifier gốc của ResNet.
        # Sau dòng này, ResNet chỉ trả về feature ảnh.
        resnet.fc = nn.Identity()
        self.image_encoder = resnet

        # Đưa text feature và image feature về cùng kích thước 256.
        self.text_projection = nn.Linear(text_feature_size, 256)
        self.image_projection = nn.Linear(image_feature_size, 256)

        # Classifier sau fusion.
        # Sau concat: 256 text + 256 image = 512.
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, image):
        # Text branch.
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Lấy vector của token đầu tiên làm đại diện cho cả câu.
        text_feature = text_output.last_hidden_state[:, 0, :]

        # Image branch.
        image_feature = self.image_encoder(image)

        # Projection.
        text_feature = self.text_projection(text_feature)
        image_feature = self.image_projection(image_feature)

        # Fusion bằng concat.
        fused_feature = torch.cat([text_feature, image_feature], dim=1)

        # Classifier.
        logits = self.classifier(fused_feature)
        return logits


model = SimpleMultimodalModel(
    text_model_name=CFG.TEXT_MODEL,
    num_classes=3,
)

model = model.to(CFG.device)

print(model)


# %% [markdown]
# ## 18. Loss và optimizer

# %%
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CFG.lr,
)


# %% [markdown]
# ## 19. Hàm train một epoch

# %%
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    all_predictions = []
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
        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average="macro")

    return avg_loss, acc, f1


# %% [markdown]
# ## 20. Hàm evaluate

# %%
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0
    all_predictions = []
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

            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average="macro")

    return avg_loss, acc, f1, all_predictions, all_labels


# %% [markdown]
# ## 21. Training loop

# %%
best_val_f1 = 0
history = []

for epoch in range(CFG.epochs):
    print("=" * 60)
    print(f"Epoch {epoch + 1}/{CFG.epochs}")

    train_loss, train_acc, train_f1 = train_one_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=CFG.device,
    )

    val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=CFG.device,
    )

    print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
    print(f"Val loss  : {val_loss:.4f} | Val acc  : {val_acc:.4f} | Val F1  : {val_f1:.4f}")

    history.append(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
        }
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_simple_memotion_model.pt")
        print("Saved best model!")


history_df = pd.DataFrame(history)
history_df


# %% [markdown]
# ## 22. Kết quả cuối cùng

# %%
model.load_state_dict(torch.load("best_simple_memotion_model.pt", map_location=CFG.device))

val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
    model=model,
    dataloader=val_loader,
    criterion=criterion,
    device=CFG.device,
)

print("Best validation accuracy:", val_acc)
print("Best validation macro-F1 :", val_f1)
print()

target_names = ["negative", "neutral", "positive"]
print(classification_report(val_labels, val_preds, target_names=target_names))

print("Confusion matrix:")
print(confusion_matrix(val_labels, val_preds))


# %% [markdown]
# ## 23. Inference thử một mẫu

# %%
def predict_one_sample(text, image_path):
    model.eval()

    encoded_text = tokenizer(
        text,
        max_length=CFG.max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        image = Image.new("RGB", (CFG.image_size, CFG.image_size), color=(0, 0, 0))

    image = val_image_transform(image).unsqueeze(0)

    input_ids = encoded_text["input_ids"].to(CFG.device)
    attention_mask = encoded_text["attention_mask"].to(CFG.device)
    image = image.to(CFG.device)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image=image,
        )

        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_id = torch.argmax(probabilities).item()

    return {
        "predicted_label": id2label[predicted_id],
        "probabilities": {
            id2label[i]: float(probabilities[i].cpu())
            for i in range(3)
        },
    }


sample = val_df.iloc[0]

print("Text:", sample["text"])
print("True label:", sample["label_name"])
print("Image path:", sample["image_path"])
print()

predict_one_sample(sample["text"], sample["image_path"])


# %% [markdown]
# ## 24. Nếu muốn chỉnh model
#
# ### Dùng BERT chuẩn thay DistilBERT
#
# Sửa trong `CFG`:
#
# ```python
# TEXT_MODEL = "bert-base-uncased"
# ```
#
# ### Dùng ResNet50 thay ResNet18
#
# Trong class `SimpleMultimodalModel`, đổi:
#
# ```python
# resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# ```
#
# thành:
#
# ```python
# resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# ```
#
# Các dòng còn lại giữ nguyên.
#
# ### Nếu bị CUDA out of memory
#
# Giảm:
#
# ```python
# batch_size = 8
# ```
#
# hoặc:
#
# ```python
# batch_size = 4
# ```
#
# ### Nếu DataLoader bị lỗi
#
# Sửa:
#
# ```python
# num_workers = 0
# ```
#
# ### Nếu muốn train lâu hơn
#
# Tăng:
#
# ```python
# epochs = 5
# ```
#
# hoặc:
#
# ```python
# epochs = 8
# ```

# %% [markdown]
# ## 25. Tóm tắt
#
# Code này chỉ dùng **Memotion Dataset 7k**.
#
# Kiến trúc:
#
# ```text
# Text  -> DistilBERT -> text feature
# Image -> ResNet18   -> image feature
# text feature + image feature -> concat fusion
# concat feature -> classifier -> negative/neutral/positive
# ```
#
# Đây là phiên bản đơn giản hơn bản 2 dataset, phù hợp để học, giải thích và demo trên Kaggle.

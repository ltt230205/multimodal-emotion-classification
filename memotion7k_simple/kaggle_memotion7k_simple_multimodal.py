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


seed_everything(42)


# %% [markdown]
# ## 4. Kiểm tra dataset trên Kaggle

# %%
print("Các folder trong /kaggle/input:")
if Path("/kaggle/input").exists():
    print(os.listdir("/kaggle/input"))
else:
    print("Không chạy trên Kaggle hoặc chưa có /kaggle/input")

print("\nDataset path:", Path("/kaggle/input/memotion-dataset-7k"))
print("Dataset path exists:", Path("/kaggle/input/memotion-dataset-7k").exists())


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


table_files = find_table_files(Path("/kaggle/input/memotion-dataset-7k"))

print("Các file metadata tìm thấy:")
for i, path in enumerate(table_files):
    print(i, "->", path)

if len(table_files) == 0:
    raise FileNotFoundError("Không tìm thấy file metadata. Hãy kiểm tra lại dataset path.")


# %% [markdown]
# ## 6. Đọc metadata
#


# %%

# ??c th?ng file labels.csv.
# N?u Kaggle c?a b?n c? ???ng d?n kh?c, s?a tr?c ti?p path b?n d??i.
metadata_path = Path(
    "/kaggle/input/datasets/williamscott701/memotion-dataset-7k/"
    "memotion_dataset_7k/labels.csv"
)

if not metadata_path.exists():
    raise FileNotFoundError(f"Kh?ng t?m th?y metadata: {metadata_path}")

df = pd.read_csv(metadata_path)

print("Metadata ?ang d?ng:", metadata_path)
print("Shape:", df.shape)
print("Columns:")
print(list(df.columns))
df.head()


# %% [markdown]
# ## 7. Ki?m tra c?c c?t c?n d?ng
#
# V? ta ?? bi?t c?u tr?c `labels.csv`, code d?ng th?ng 3 c?t:
#
# - `image_name`: t?n file ?nh
# - `text_ocr`: text/OCR ??a v?o BERT
# - `overall_sentiment`: nh?n sentiment

# %%
required_columns = ["image_name", "text_ocr", "overall_sentiment"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Kh?ng t?m th?y c?t b?t bu?c: {col}")

print("C?c c?t c?n d?ng ??u t?n t?i:", required_columns)

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
for path in Path("/kaggle/input/memotion-dataset-7k").rglob("*"):
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
data["text"] = df["text_ocr"].fillna("").astype(str)
data["image_path"] = df["image_name"].apply(get_image_path)
data["label_name"] = df["overall_sentiment"].apply(normalize_sentiment)

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
# ## 14. Tokenizer và transform ảnh
#
# Text dùng tokenizer của DistilBERT/BERT.
# Image được resize 224x224 và normalize theo ImageNet để hợp với ResNet pretrained.

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
    def __init__(self, dataframe, tokenizer, image_transform, max_len, image_size):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_len = max_len
        self.image_size = image_size

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
            image = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))

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
    max_len=96,
    image_size=224,
)

val_dataset = MemotionDataset(
    dataframe=val_df,
    tokenizer=tokenizer,
    image_transform=val_image_transform,
    max_len=96,
    image_size=224,
)

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
# ## 17. Model: Text branch + Image branch + Cross-Attention Fusion
#
# Model gồm:
#
# - `text_encoder`: BERT nano/tiny
# - `image_encoder`: ResNet18 pretrained
# - `cross_attention`: cho text tokens attend vào image tokens
# - `classifier`: phân loại sau cross-attention fusion
#
# Khác bản concat cũ:
#
# - Concat fusion chỉ nối `text_feature` và `image_feature`.
# - Cross-attention fusion cho đặc trưng text học cách chú ý tới vùng ảnh liên quan.

# %%
class SimpleMultimodalModel(nn.Module):
    def __init__(self, text_model_name, num_classes):
        super().__init__()

        # Branch 1: Text encoder.
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_feature_size = self.text_encoder.config.hidden_size

        # Branch 2: Image encoder.
        # Dùng ResNet18 pretrained để nhẹ và dễ chạy trên Kaggle.
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Bỏ avgpool và fc cuối để giữ feature map 7x7.
        # Nếu input image là 224x224, output thường có shape:
        # batch_size x 512 x 7 x 7
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-2])
        image_feature_size = 512

        # Đưa text token và image token về cùng kích thước để attention được.
        # MultiheadAttention yêu cầu query/key/value có cùng embed_dim.
        self.fusion_dim = 256
        self.text_projection = nn.Linear(text_feature_size, 256)
        self.image_projection = nn.Linear(image_feature_size, 256)

        # Cross-attention fusion.
        # Query  : text tokens
        # Key    : image tokens
        # Value  : image tokens
        #
        # Nghĩa là text sẽ học cách "nhìn" vào các vùng ảnh liên quan.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(self.fusion_dim)

        # Classifier sau cross-attention fusion.
        # Ta ghép 3 vector:
        # - text_cls: đặc trưng text gốc
        # - attended_cls: text sau khi attend vào ảnh
        # - image_global: đặc trưng ảnh trung bình
        #
        # Tổng kích thước: 256 * 3 = 768.
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask, image):
        # Text branch.
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Lấy toàn bộ token text, không chỉ lấy CLS.
        # Shape: batch_size x max_len x text_hidden
        text_tokens = text_output.last_hidden_state

        # Image branch.
        # Shape: batch_size x 512 x 7 x 7
        image_feature_map = self.image_encoder(image)

        # Đổi feature map thành chuỗi image tokens.
        # batch_size x 512 x 7 x 7 -> batch_size x 49 x 512
        batch_size, channels, height, width = image_feature_map.shape
        image_tokens = image_feature_map.view(batch_size, channels, height * width)
        image_tokens = image_tokens.permute(0, 2, 1)

        # Projection.
        text_tokens = self.text_projection(text_tokens)
        image_tokens = self.image_projection(image_tokens)

        # Cross-attention fusion.
        # Text tokens đóng vai trò Query.
        # Image tokens đóng vai trò Key và Value.
        attended_text_tokens, attention_weights = self.cross_attention(
            query=text_tokens,
            key=image_tokens,
            value=image_tokens,
        )

        # Residual connection + LayerNorm giúp training ổn định hơn.
        attended_text_tokens = self.attention_norm(text_tokens + attended_text_tokens)

        # Lấy token đầu tiên làm đại diện.
        text_cls = text_tokens[:, 0, :]
        attended_cls = attended_text_tokens[:, 0, :]

        # Lấy trung bình các image tokens làm đặc trưng ảnh tổng quát.
        image_global = image_tokens.mean(dim=1)

        # Fusion cuối cùng sau cross-attention.
        fused_feature = torch.cat([text_cls, attended_cls, image_global], dim=1)

        # Classifier.
        logits = self.classifier(fused_feature)
        return logits


model = SimpleMultimodalModel(
    text_model_name="prajjwal1/bert-tiny",
    num_classes=3,
)

model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

print(model)


# %% [markdown]
# ## 18. Loss và optimizer

# %%
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
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

for epoch in range(3):
    print("=" * 60)
    print(f"Epoch {epoch + 1}/3")

    train_loss, train_acc, train_f1 = train_one_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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
model.load_state_dict(torch.load("best_simple_memotion_model.pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
    model=model,
    dataloader=val_loader,
    criterion=criterion,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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
def predict_one_sample(text, image_path, max_len, image_size, device):
    model.eval()

    encoded_text = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        image = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))

    image = val_image_transform(image).unsqueeze(0)

    input_ids = encoded_text["input_ids"].to(device)
    attention_mask = encoded_text["attention_mask"].to(device)
    image = image.to(device)

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

predict_one_sample(
    text=sample["text"],
    image_path=sample["image_path"],
    max_len=96,
    image_size=224,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)


# %% [markdown]
# ## 24. Nếu muốn chỉnh model
#
# ### Dùng BERT chuẩn thay BERT nano/tiny
#
# Sửa ? ph?n tham s? ??u notebook:
#
# ```python
# text_model_name="bert-base-uncased"
# ```
#
# ### Dùng DistilBERT thay BERT nano/tiny
#
# Sửa ? ph?n tham s? ??u notebook:
#
# ```python
# text_model_name="distilbert-base-uncased"
# ```
#
# ### Dùng ResNet50 thay ResNet18
#
# Nếu muốn dùng ResNet50, phần image encoder cần sửa cẩn thận hơn vì số channel
# output của ResNet50 là 2048 thay vì 512.
#
# Trong class `SimpleMultimodalModel`, đổi:
#
# ```python
# resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# image_feature_size = 512
# ```
#
# thành:
#
# ```python
# resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# image_feature_size = 2048
# ```
#
# ### Nếu bị CUDA out of memory
#
# Giảm:
#
# ```python
# batch_size=8
# ```
#
# hoặc:
#
# ```python
# batch_size=4
# ```
#
# ### Nếu DataLoader bị lỗi
#
# Sửa:
#
# ```python
# num_workers=0
# ```
#
# ### Nếu muốn train lâu hơn
#
# Tăng:
#
# ```python
# range(5)
# ```
#
# hoặc:
#
# ```python
# range(8)
# ```

# %% [markdown]
# ## 25. Tóm tắt
#
# Code này chỉ dùng **Memotion Dataset 7k**.
#
# Kiến trúc:
#
# ```text
# Text  -> BERT nano/tiny -> text tokens
# Image -> ResNet18       -> image tokens
# text tokens attend image tokens -> cross-attention fusion
# fused feature -> classifier -> negative/neutral/positive
# ```
#
# Đây là phiên bản đơn giản hơn bản 2 dataset, phù hợp để học, giải thích và demo trên Kaggle.

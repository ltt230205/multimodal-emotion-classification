# %% [markdown]
# # Multimodal Sentiment/Emotion Classification: Text + Image Fusion
#
# Pipeline cho Kaggle:
#
# - Dataset 1: `williamscott701/memotion-dataset-7k`
# - Dataset 2: `multimodal-sentiment-analysis-cuet-nlp`
# - Text branch: BERT/DistilBERT encoder
# - Image branch: ResNet encoder
# - Fusion: concatenate đặc trưng text + image
# - Classifier head: MLP phân loại `negative`, `neutral`, `positive`
#
# Ghi chú: hai bộ dữ liệu Kaggle này thường được dùng cho sentiment trên meme.
# Nếu bạn muốn gọi là emotion classification, có thể trình bày đây là bài toán
# nhận diện cảm xúc/thái độ biểu đạt qua meme ở mức 3 lớp sentiment.

# %% [markdown]
# ## 1. Cài đặt và import

# %%
import os
import re
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import torchvision.transforms as T
from torchvision import models

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True


# %% [markdown]
# ## 2. Config

# %%
class CFG:
    seed = 42
    max_len = 96
    image_size = 224
    batch_size = 16
    epochs = 4
    lr = 2e-5
    weight_decay = 1e-4
    num_workers = 2

    # Dùng DistilBERT để nhẹ hơn BERT-base trên Kaggle GPU.
    # Nếu muốn đúng "BERT" hơn, đổi thành: "bert-base-uncased"
    text_model_name = "distilbert-base-uncased"

    # Đặt đúng tên folder sau khi Add Data trên Kaggle nếu cần.
    memotion_root = Path("/kaggle/input/memotion-dataset-7k")
    cuet_root = Path("/kaggle/input/multimodal-sentiment-analysis-cuet-nlp")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG.seed)
print("Device:", CFG.device)


# %% [markdown]
# ## 3. Hàm tiện ích đọc metadata và ảnh
#
# Vì mỗi phiên bản dataset Kaggle có thể đặt tên file/cột hơi khác nhau, phần này
# cố gắng tự dò cột text, cột image và cột label.

# %%
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_files(root: Path, suffixes=None):
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file()]
    if suffixes is None:
        return files
    suffixes = {s.lower() for s in suffixes}
    return [p for p in files if p.suffix.lower() in suffixes]


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported table format: {path}")


def find_best_table(root: Path) -> Path:
    tables = list_files(root, suffixes={".csv", ".xlsx", ".xls", ".json"})
    if not tables:
        raise FileNotFoundError(f"Không tìm thấy file metadata trong {root}")

    scored = []
    for p in tables:
        name = p.name.lower()
        score = 0
        if "train" in name:
            score += 4
        if "label" in name or "annotation" in name:
            score += 3
        if "readme" in name:
            score -= 4
        score += min(p.stat().st_size / 1_000_000, 5)
        scored.append((score, p))

    return sorted(scored, reverse=True)[0][1]


def guess_column(columns, candidates):
    normalized = {c: re.sub(r"[^a-z0-9]+", "", str(c).lower()) for c in columns}
    for cand in candidates:
        cand_norm = re.sub(r"[^a-z0-9]+", "", cand.lower())
        for original, norm in normalized.items():
            if cand_norm == norm or cand_norm in norm:
                return original
    return None


def build_image_index(root: Path):
    image_files = list_files(root, suffixes=IMAGE_EXTS)
    index = {}
    for p in image_files:
        index[p.name.lower()] = p
        index[p.stem.lower()] = p
    return index


def resolve_image_path(value, image_index):
    if pd.isna(value):
        return None

    raw = str(value).strip()
    if not raw:
        return None

    raw_path = Path(raw)
    keys = [
        raw.lower(),
        raw_path.name.lower(),
        raw_path.stem.lower(),
    ]

    for key in keys:
        if key in image_index:
            return str(image_index[key])

    # Một số metadata lưu image_id không có đuôi file.
    for ext in IMAGE_EXTS:
        key = f"{raw.lower()}{ext}"
        if key in image_index:
            return str(image_index[key])

    return None


def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False


def normalize_label(value):
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()

    mapping = {
        "0": "positive",
        "1": "negative",
        "2": "neutral",
        "pos": "positive",
        "positive": "positive",
        "very positive": "positive",
        "neg": "negative",
        "negative": "negative",
        "very negative": "negative",
        "neu": "neutral",
        "neutral": "neutral",
    }
    return mapping.get(text)


def load_multimodal_dataset(root: Path, dataset_name: str) -> pd.DataFrame:
    table_path = find_best_table(root)
    df = read_table(table_path)
    df.columns = [str(c).strip() for c in df.columns]

    text_col = guess_column(
        df.columns,
        candidates=[
            "text",
            "ocr",
            "ocr_text",
            "extracted_text",
            "sentence",
            "caption",
            "tweet",
            "content",
        ],
    )
    image_col = guess_column(
        df.columns,
        candidates=[
            "image",
            "image_name",
            "image_path",
            "file_name",
            "filename",
            "img",
            "image_id",
        ],
    )
    label_col = guess_column(
        df.columns,
        candidates=[
            "overall_sentiment",
            "sentiment",
            "label",
            "class",
            "target",
        ],
    )

    if text_col is None or image_col is None or label_col is None:
        raise ValueError(
            f"Không tự dò được đủ cột trong {table_path}.\n"
            f"Columns hiện có: {list(df.columns)}\n"
            "Hãy sửa thủ công text_col, image_col, label_col trong hàm load_multimodal_dataset."
        )

    image_index = build_image_index(root)
    out = pd.DataFrame(
        {
            "text": df[text_col].fillna("").astype(str),
            "image_path": df[image_col].apply(lambda x: resolve_image_path(x, image_index)),
            "label_name": df[label_col].apply(normalize_label),
            "source": dataset_name,
        }
    )

    before = len(out)
    out = out.dropna(subset=["image_path", "label_name"]).reset_index(drop=True)

    before_image_check = len(out)
    out = out[out["image_path"].apply(is_valid_image)].reset_index(drop=True)

    print(
        f"{dataset_name}: metadata={table_path.name}, "
        f"giữ {len(out)}/{before} mẫu hợp lệ, "
        f"bỏ {before_image_check - len(out)} ảnh lỗi"
    )
    print(out["label_name"].value_counts())
    return out


# %% [markdown]
# ## 4. Load và gộp 2 bộ dataset Kaggle
#
# Trên Kaggle Notebook:
# 1. Bấm **Add Data**
# 2. Thêm `Memotion Dataset 7k`
# 3. Thêm `Multimodal Sentiment Analysis`
# 4. Nếu folder trong `/kaggle/input` khác tên config, sửa `CFG.memotion_root` và `CFG.cuet_root`.

# %%
datasets = []

if CFG.memotion_root.exists():
    datasets.append(load_multimodal_dataset(CFG.memotion_root, "memotion7k"))
else:
    print("Không thấy Memotion root:", CFG.memotion_root)

if CFG.cuet_root.exists():
    datasets.append(load_multimodal_dataset(CFG.cuet_root, "cuet_msa"))
else:
    print("Không thấy CUET root:", CFG.cuet_root)

if not datasets:
    raise RuntimeError(
        "Chưa có dataset trong /kaggle/input. Hãy Add Data trên Kaggle hoặc sửa đường dẫn trong CFG."
    )

data = pd.concat(datasets, ignore_index=True)
data = data.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)

label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}
data["label"] = data["label_name"].map(label2id)

print("Tổng số mẫu:", len(data))
print(data.groupby(["source", "label_name"]).size())
data.head()


# %% [markdown]
# ## 5. Train/validation split

# %%
train_df, val_df = train_test_split(
    data,
    test_size=0.2,
    random_state=CFG.seed,
    stratify=data["label"],
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)


# %% [markdown]
# ## 6. Dataset và DataLoader
#
# Text được tokenize bằng tokenizer của BERT/DistilBERT. Image được resize,
# normalize theo ImageNet để đưa vào ResNet pretrained.

# %%
tokenizer = AutoTokenizer.from_pretrained(CFG.text_model_name)

train_tfms = T.Compose(
    [
        T.Resize((CFG.image_size, CFG.image_size)),
        T.RandomHorizontalFlip(p=0.3),
        T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_tfms = T.Compose(
    [
        T.Resize((CFG.image_size, CFG.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class MemeMultimodalDataset(Dataset):
    def __init__(self, df, tokenizer, transforms=None, max_len=96):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        encoded = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (CFG.image_size, CFG.image_size), color=(0, 0, 0))
        if self.transforms:
            image = self.transforms(image)

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "image": image,
            "label": torch.tensor(row["label"], dtype=torch.long),
        }


train_ds = MemeMultimodalDataset(train_df, tokenizer, train_tfms, CFG.max_len)
val_ds = MemeMultimodalDataset(val_df, tokenizer, val_tfms, CFG.max_len)

train_loader = DataLoader(
    train_ds,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=CFG.num_workers,
    pin_memory=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True,
)


# %% [markdown]
# ## 7. Model: BERT branch + ResNet branch + Fusion classifier

# %%
class MultimodalFusionClassifier(nn.Module):
    def __init__(self, text_model_name, num_classes=3, dropout=0.25, freeze_backbones=False):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_hidden = self.text_encoder.config.hidden_size

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        image_hidden = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.image_encoder = resnet

        if freeze_backbones:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, image):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # DistilBERT/BERT đều có token đầu tiên tương đương CLS representation.
        text_feat = text_outputs.last_hidden_state[:, 0, :]
        image_feat = self.image_encoder(image)

        text_feat = self.text_proj(text_feat)
        image_feat = self.image_proj(image_feat)

        fused = torch.cat([text_feat, image_feat], dim=1)
        logits = self.classifier(fused)
        return logits


model = MultimodalFusionClassifier(
    text_model_name=CFG.text_model_name,
    num_classes=len(label2id),
    dropout=0.25,
    freeze_backbones=False,
).to(CFG.device)

print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


# %% [markdown]
# ## 8. Train và evaluate

# %%
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CFG.lr,
    weight_decay=CFG.weight_decay,
)

total_steps = len(train_loader) * CFG.epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)


def move_batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_one_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    losses = []
    preds, targets = [], []

    for batch in loader:
        batch = move_batch_to_device(batch, CFG.device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            image=batch["image"],
        )
        loss = criterion(logits, batch["label"])
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
        targets.extend(batch["label"].detach().cpu().numpy())

    return {
        "loss": float(np.mean(losses)),
        "acc": accuracy_score(targets, preds),
        "f1_macro": f1_score(targets, preds, average="macro"),
    }


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    losses = []
    preds, targets = [], []

    for batch in loader:
        batch = move_batch_to_device(batch, CFG.device)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            image=batch["image"],
        )
        loss = criterion(logits, batch["label"])

        losses.append(loss.item())
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
        targets.extend(batch["label"].detach().cpu().numpy())

    return {
        "loss": float(np.mean(losses)),
        "acc": accuracy_score(targets, preds),
        "f1_macro": f1_score(targets, preds, average="macro"),
        "preds": preds,
        "targets": targets,
    }


best_f1 = -1
history = []

for epoch in range(1, CFG.epochs + 1):
    train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, criterion)
    val_metrics = evaluate(model, val_loader, criterion)

    row = {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_acc": train_metrics["acc"],
        "train_f1": train_metrics["f1_macro"],
        "val_loss": val_metrics["loss"],
        "val_acc": val_metrics["acc"],
        "val_f1": val_metrics["f1_macro"],
    }
    history.append(row)
    print(row)

    if val_metrics["f1_macro"] > best_f1:
        best_f1 = val_metrics["f1_macro"]
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "label2id": label2id,
                "id2label": id2label,
                "cfg": {
                    "text_model_name": CFG.text_model_name,
                    "max_len": CFG.max_len,
                    "image_size": CFG.image_size,
                },
            },
            "best_multimodal_fusion_model.pt",
        )
        print("Saved best model!")

pd.DataFrame(history)


# %% [markdown]
# ## 9. Báo cáo kết quả

# %%
checkpoint = torch.load("best_multimodal_fusion_model.pt", map_location=CFG.device)
model.load_state_dict(checkpoint["model_state_dict"])

val_metrics = evaluate(model, val_loader, criterion)
target_names = [id2label[i] for i in range(len(id2label))]

print("Validation Accuracy:", val_metrics["acc"])
print("Validation Macro-F1:", val_metrics["f1_macro"])
print()
print(classification_report(val_metrics["targets"], val_metrics["preds"], target_names=target_names))
print("Confusion matrix:")
print(confusion_matrix(val_metrics["targets"], val_metrics["preds"]))


# %% [markdown]
# ## 10. Inference thử trên một mẫu

# %%
@torch.no_grad()
def predict_one(text, image_path):
    model.eval()

    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=CFG.max_len,
        return_tensors="pt",
    )
    image = Image.open(image_path).convert("RGB")
    image = val_tfms(image).unsqueeze(0)

    input_ids = encoded["input_ids"].to(CFG.device)
    attention_mask = encoded["attention_mask"].to(CFG.device)
    image = image.to(CFG.device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask, image=image)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    return {
        "label": id2label[int(np.argmax(probs))],
        "probabilities": {id2label[i]: float(probs[i]) for i in range(len(probs))},
    }


sample = val_df.iloc[0]
print("Text:", sample["text"])
print("True label:", sample["label_name"])
print("Image:", sample["image_path"])
predict_one(sample["text"], sample["image_path"])


# %% [markdown]
# ## 11. Gợi ý trình bày trong báo cáo
#
# Công thức mô hình:
#
# 1. Text encoder: `h_text = BERT(text)[CLS]`
# 2. Image encoder: `h_img = ResNet50(image)`
# 3. Projection: `z_text = MLP_text(h_text)`, `z_img = MLP_img(h_img)`
# 4. Fusion: `z = concat(z_text, z_img)`
# 5. Classification: `y_hat = MLP_classifier(z)`
#
# Điểm đúng yêu cầu đề:
#
# - Text và image được xử lý bằng 2 branch riêng.
# - BERT/DistilBERT chỉ trích xuất đặc trưng ngôn ngữ.
# - ResNet50 pretrained chỉ trích xuất đặc trưng ảnh.
# - Fusion sau khi có feature của từng modality.
# - Classifier head học từ biểu diễn đã fusion để phân loại.

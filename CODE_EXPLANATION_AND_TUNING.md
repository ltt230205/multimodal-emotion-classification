# Giải thích từng đoạn code và hướng dẫn tinh chỉnh

File này giải thích chi tiết code trong:

```text
kaggle_multimodal_emotion_fusion.py
```

Mục tiêu là giúp bạn hiểu luồng xử lý của bài DL multimodal:

```text
Text  -> BERT/DistilBERT -> text feature  \
                                           fusion -> classifier -> sentiment label
Image -> ResNet50        -> image feature /
```

## 1. Import thư viện

Đoạn đầu code import các thư viện cần thiết:

```python
import os
import re
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
```

Ý nghĩa:

- `os`, `Path`: xử lý đường dẫn file/folder.
- `re`: xử lý chuỗi bằng regex, dùng khi đoán tên cột.
- `random`, `numpy`, `torch`: cố định seed để kết quả ổn định hơn.
- `pandas`: đọc metadata `.csv`, `.xlsx`, `.json`.
- `PIL.Image`: đọc ảnh.
- `ImageFile.LOAD_TRUNCATED_IMAGES = True`: giúp PIL cố đọc một số ảnh bị lỗi/truncated trong dataset Kaggle.

Phần import PyTorch:

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
```

Ý nghĩa:

- `Dataset`: định nghĩa cách lấy một sample text + image + label.
- `DataLoader`: gom nhiều sample thành batch.
- `nn.Module`: viết model deep learning.

Phần import model:

```python
import torchvision.transforms as T
from torchvision import models

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
```

Ý nghĩa:

- `torchvision.transforms`: tiền xử lý ảnh.
- `models.resnet50`: model ResNet pretrained.
- `AutoTokenizer`: tokenizer cho BERT/DistilBERT.
- `AutoModel`: backbone text encoder.
- `get_linear_schedule_with_warmup`: scheduler giảm learning rate dần trong quá trình train.

## 2. Config

Code gom các tham số quan trọng vào class `CFG`:

```python
class CFG:
    seed = 42
    max_len = 96
    image_size = 224
    batch_size = 16
    epochs = 4
    lr = 2e-5
    weight_decay = 1e-4
    num_workers = 2
    text_model_name = "distilbert-base-uncased"
```

Các tham số quan trọng:

| Tham số | Ý nghĩa | Khi nào cần sửa |
|---|---|---|
| `seed` | cố định random | thường để nguyên |
| `max_len` | độ dài token tối đa của text | text dài thì tăng lên 128 |
| `image_size` | kích thước ảnh input cho ResNet | ResNet thường dùng 224 |
| `batch_size` | số mẫu trong một batch | lỗi CUDA memory thì giảm |
| `epochs` | số vòng train | muốn train kỹ hơn thì tăng |
| `lr` | learning rate | model không học thì chỉnh |
| `num_workers` | số worker load data | lỗi DataLoader thì đặt 0 |
| `text_model_name` | tên model Hugging Face | đổi BERT/DistilBERT tại đây |

Đường dẫn dataset:

```python
memotion_root = Path("/kaggle/input/memotion-dataset-7k")
cuet_root = Path("/kaggle/input/multimodal-sentiment-analysis-cuet-nlp")
```

Trên Kaggle của bạn, nếu dataset nằm trong `/kaggle/input/datasets/...`, hãy sửa thành dạng:

```python
memotion_root = Path("/kaggle/input/datasets/williamscott701/memotion-dataset-7k")
cuet_root = Path("/kaggle/input/datasets/hosen42/multimodal-sentiment-analysis-cuet-nlp")
```

Muốn kiểm tra tên folder thật:

```python
import os
print(os.listdir("/kaggle/input"))
print(os.listdir("/kaggle/input/datasets"))
```

## 3. Cố định seed

Hàm:

```python
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Mục đích là giảm sự thay đổi kết quả giữa các lần chạy.

Lưu ý: trong deep learning, kết quả vẫn có thể dao động nhẹ vì GPU và DataLoader.

## 4. Các hàm đọc file dataset

### `list_files`

```python
def list_files(root: Path, suffixes=None):
```

Hàm này duyệt toàn bộ folder dataset và lấy danh sách file.

Nếu truyền `suffixes`, nó chỉ lấy file có đuôi mong muốn. Ví dụ:

```python
list_files(root, suffixes={".csv", ".xlsx"})
```

sẽ chỉ lấy file bảng metadata.

### `read_table`

```python
def read_table(path: Path) -> pd.DataFrame:
```

Hàm này đọc metadata:

- `.csv` bằng `pd.read_csv`
- `.xlsx`, `.xls` bằng `pd.read_excel`
- `.json` bằng `pd.read_json`

Output là một `DataFrame`.

### `find_best_table`

```python
def find_best_table(root: Path) -> Path:
```

Dataset Kaggle có thể có nhiều file metadata. Hàm này tự chọn file có vẻ quan trọng nhất.

Nó ưu tiên file có tên chứa:

- `train`
- `label`
- `annotation`

Nếu hàm chọn sai metadata, bạn có thể sửa thủ công:

```python
table_path = root / "ten_file_metadata.csv"
```

trong hàm `load_multimodal_dataset`.

### `guess_column`

```python
def guess_column(columns, candidates):
```

Hàm này tự đoán cột nào là:

- cột text
- cột image
- cột label

Ví dụ nếu metadata có cột `ocr_text`, hàm sẽ nhận ra đây là cột text.

Nếu code báo lỗi không tìm được cột, bạn xem danh sách columns rồi sửa tay:

```python
text_col = "ten_cot_text"
image_col = "ten_cot_image"
label_col = "ten_cot_label"
```

## 5. Tìm đường dẫn ảnh

### `build_image_index`

```python
def build_image_index(root: Path):
```

Hàm này quét toàn bộ ảnh trong dataset và tạo dictionary để tìm ảnh nhanh.

Nó lưu cả:

- tên file đầy đủ, ví dụ `abc.png`
- tên không có đuôi, ví dụ `abc`

Vì một số metadata chỉ lưu `image_id` chứ không lưu đầy đủ `abc.png`.

### `resolve_image_path`

```python
def resolve_image_path(value, image_index):
```

Hàm này lấy giá trị trong cột image của metadata, rồi tìm file ảnh thật trong dataset.

Ví dụ metadata ghi:

```text
image_123
```

Hàm sẽ thử tìm:

```text
image_123
image_123.jpg
image_123.png
...
```

Nếu không tìm thấy, sample đó sẽ bị bỏ.

## 6. Chuẩn hóa label

Hàm:

```python
def normalize_label(value):
```

Mục tiêu là đưa nhiều kiểu nhãn khác nhau về 3 lớp:

```text
negative
neutral
positive
```

Ví dụ:

| Nhãn gốc | Nhãn sau chuẩn hóa |
|---|---|
| `positive` | `positive` |
| `very positive` | `positive` |
| `negative` | `negative` |
| `very negative` | `negative` |
| `neutral` | `neutral` |
| `0` | `positive` |
| `1` | `negative` |
| `2` | `neutral` |

Lưu ý quan trọng: nếu dataset của bạn mã hóa số khác, ví dụ:

```text
0 = negative
1 = neutral
2 = positive
```

thì phải sửa mapping trong hàm này. Đây là một trong những chỗ cần kiểm tra kỹ nhất.

## 7. Kiểm tra ảnh lỗi

Hàm:

```python
def is_valid_image(path):
```

Mục đích là bỏ những ảnh bị hỏng trước khi train.

Nó thử:

```python
img.verify()
img.convert("RGB")
```

Nếu lỗi, ảnh đó bị loại.

Điều này giúp tránh lỗi:

```text
OSError: image file is truncated
```

## 8. Load và gộp 2 dataset

Đoạn:

```python
datasets = []

if CFG.memotion_root.exists():
    datasets.append(load_multimodal_dataset(CFG.memotion_root, "memotion7k"))

if CFG.cuet_root.exists():
    datasets.append(load_multimodal_dataset(CFG.cuet_root, "cuet_msa"))
```

Ý nghĩa:

- Nếu tìm thấy folder Memotion, load dataset Memotion.
- Nếu tìm thấy folder CUET, load dataset CUET.
- Sau đó gộp hai dataset lại.

Đoạn gộp:

```python
data = pd.concat(datasets, ignore_index=True)
data = data.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)
```

`sample(frac=1)` dùng để shuffle toàn bộ dữ liệu.

Gán label số:

```python
label2id = {"negative": 0, "neutral": 1, "positive": 2}
data["label"] = data["label_name"].map(label2id)
```

Model không học trực tiếp string `positive`, mà học số `0, 1, 2`.

## 9. Train/validation split

```python
train_df, val_df = train_test_split(
    data,
    test_size=0.2,
    random_state=CFG.seed,
    stratify=data["label"],
)
```

Ý nghĩa:

- 80% dữ liệu dùng để train.
- 20% dữ liệu dùng để validation.
- `stratify=data["label"]` giúp tỉ lệ nhãn ở train/validation gần giống nhau.

Bạn có thể chỉnh:

```python
test_size=0.1
```

nếu muốn train nhiều hơn và validation ít hơn.

## 10. Tokenizer và image transforms

Tokenizer:

```python
tokenizer = AutoTokenizer.from_pretrained(CFG.text_model_name)
```

Nó biến text thành:

- `input_ids`
- `attention_mask`

Image transform train:

```python
train_tfms = T.Compose([
    T.Resize((CFG.image_size, CFG.image_size)),
    T.RandomHorizontalFlip(p=0.3),
    T.ColorJitter(...),
    T.ToTensor(),
    T.Normalize(...)
])
```

Ý nghĩa:

- `Resize`: đưa ảnh về 224x224.
- `RandomHorizontalFlip`: augmentation lật ảnh.
- `ColorJitter`: augmentation màu sắc.
- `ToTensor`: đổi ảnh PIL sang tensor.
- `Normalize`: chuẩn hóa theo ImageNet vì ResNet pretrained trên ImageNet.

Validation transform không dùng augmentation:

```python
val_tfms = T.Compose([
    T.Resize((CFG.image_size, CFG.image_size)),
    T.ToTensor(),
    T.Normalize(...)
])
```

## 11. Class `MemeMultimodalDataset`

Đây là class rất quan trọng.

```python
class MemeMultimodalDataset(Dataset):
```

Nó định nghĩa cách lấy một sample.

### `__init__`

```python
def __init__(self, df, tokenizer, transforms=None, max_len=96):
```

Lưu:

- dataframe
- tokenizer
- image transforms
- max length cho text

### `__len__`

```python
def __len__(self):
    return len(self.df)
```

Trả về số sample trong dataset.

### `__getitem__`

```python
def __getitem__(self, idx):
```

Hàm này lấy một dòng trong dataframe, sau đó xử lý text và image riêng.

Text:

```python
encoded = self.tokenizer(
    row["text"],
    padding="max_length",
    truncation=True,
    max_length=self.max_len,
    return_tensors="pt",
)
```

Image:

```python
try:
    image = Image.open(row["image_path"]).convert("RGB")
except Exception:
    image = Image.new("RGB", (CFG.image_size, CFG.image_size), color=(0, 0, 0))
```

Nếu ảnh lỗi, code thay bằng ảnh đen để train không bị dừng.

Output:

```python
return {
    "input_ids": encoded["input_ids"].squeeze(0),
    "attention_mask": encoded["attention_mask"].squeeze(0),
    "image": image,
    "label": torch.tensor(row["label"], dtype=torch.long),
}
```

Một sample có đủ input cho cả 2 branch.

## 12. DataLoader

```python
train_loader = DataLoader(
    train_ds,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=CFG.num_workers,
    pin_memory=True,
)
```

Ý nghĩa:

- `batch_size`: số mẫu mỗi batch.
- `shuffle=True`: đảo dữ liệu khi train.
- `num_workers`: số tiến trình load data.
- `pin_memory=True`: giúp chuyển batch lên GPU nhanh hơn.

Nếu bị lỗi DataLoader khó hiểu, đặt:

```python
num_workers = 0
```

trong `CFG`.

## 13. Model `MultimodalFusionClassifier`

Đây là phần chính của bài.

```python
class MultimodalFusionClassifier(nn.Module):
```

### Text encoder

```python
self.text_encoder = AutoModel.from_pretrained(text_model_name)
text_hidden = self.text_encoder.config.hidden_size
```

Nếu dùng DistilBERT, `text_hidden = 768`.

Text encoder nhận:

```python
input_ids
attention_mask
```

và trả về embedding.

### Image encoder

```python
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
image_hidden = resnet.fc.in_features
resnet.fc = nn.Identity()
self.image_encoder = resnet
```

ResNet50 gốc có lớp cuối để phân loại ImageNet 1000 lớp. Ta bỏ lớp đó để lấy feature.

`image_hidden` thường là `2048`.

### Freeze backbone

```python
if freeze_backbones:
    for p in self.text_encoder.parameters():
        p.requires_grad = False
    for p in self.image_encoder.parameters():
        p.requires_grad = False
```

Nếu `freeze_backbones=True`, BERT và ResNet không được cập nhật trọng số.

Khi nào nên bật?

- GPU yếu.
- Train quá chậm.
- Dataset nhỏ.
- Muốn chỉ train fusion classifier cho nhanh.

Nhược điểm:

- Model có thể kém hơn fine-tune toàn bộ.

### Projection layers

```python
self.text_proj = nn.Sequential(
    nn.Linear(text_hidden, 256),
    nn.ReLU(),
    nn.Dropout(dropout),
)
```

Mục đích: đưa feature text về 256 chiều.

Image cũng tương tự:

```python
self.image_proj = nn.Sequential(
    nn.Linear(image_hidden, 256),
    nn.ReLU(),
    nn.Dropout(dropout),
)
```

Sau projection:

```text
text feature:  256 chiều
image feature: 256 chiều
```

### Classifier

```python
self.classifier = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(256, num_classes),
)
```

Vì fusion là nối 256 + 256 nên input classifier là 512 chiều.

Output là 3 logits ứng với:

```text
negative, neutral, positive
```

## 14. Forward pass

```python
def forward(self, input_ids, attention_mask, image):
```

Text branch:

```python
text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
text_feat = text_outputs.last_hidden_state[:, 0, :]
```

Image branch:

```python
image_feat = self.image_encoder(image)
```

Projection:

```python
text_feat = self.text_proj(text_feat)
image_feat = self.image_proj(image_feat)
```

Fusion:

```python
fused = torch.cat([text_feat, image_feat], dim=1)
```

Classifier:

```python
logits = self.classifier(fused)
```

`logits` chưa phải xác suất. Khi inference mới dùng:

```python
torch.softmax(logits, dim=1)
```

## 15. Loss, optimizer, scheduler

Loss:

```python
criterion = nn.CrossEntropyLoss()
```

Phù hợp với bài phân loại nhiều lớp.

Optimizer:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CFG.lr,
    weight_decay=CFG.weight_decay,
)
```

AdamW thường được dùng khi fine-tune BERT.

Scheduler:

```python
scheduler = get_linear_schedule_with_warmup(...)
```

Ý nghĩa:

- Ban đầu tăng learning rate từ từ trong warmup.
- Sau đó giảm tuyến tính.

Điều này giúp fine-tune ổn định hơn.

## 16. Train một epoch

Hàm:

```python
def train_one_epoch(model, loader, optimizer, scheduler, criterion):
```

Quy trình:

1. Bật train mode:

```python
model.train()
```

2. Lặp qua batch:

```python
for batch in loader:
```

3. Đưa batch lên GPU:

```python
batch = move_batch_to_device(batch, CFG.device)
```

4. Forward:

```python
logits = model(...)
```

5. Tính loss:

```python
loss = criterion(logits, batch["label"])
```

6. Backpropagation:

```python
loss.backward()
```

7. Gradient clipping:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Giúp tránh gradient quá lớn.

8. Cập nhật trọng số:

```python
optimizer.step()
scheduler.step()
```

9. Tính metric:

```python
accuracy_score(...)
f1_score(..., average="macro")
```

## 17. Evaluate

Hàm:

```python
@torch.no_grad()
def evaluate(model, loader, criterion):
```

Khác train ở chỗ:

- `model.eval()`
- không backprop
- không update trọng số
- chỉ tính loss và metric

`@torch.no_grad()` giúp tiết kiệm VRAM.

## 18. Training loop

```python
for epoch in range(1, CFG.epochs + 1):
    train_metrics = train_one_epoch(...)
    val_metrics = evaluate(...)
```

Mỗi epoch:

- train trên train set
- đánh giá trên validation set
- lưu metric vào `history`
- nếu `val_f1` tốt nhất thì save model

Save checkpoint:

```python
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "label2id": label2id,
        "id2label": id2label,
        "cfg": {...},
    },
    "best_multimodal_fusion_model.pt",
)
```

Metric chọn model tốt nhất là `Macro-F1`, không phải accuracy.

Vì sao dùng Macro-F1?

Dataset sentiment thường lệch nhãn. Accuracy có thể cao nếu model đoán nhiều vào lớp đông mẫu. Macro-F1 đánh giá công bằng hơn giữa các lớp.

## 19. Classification report và confusion matrix

Classification report:

```python
print(classification_report(...))
```

Nó cho biết:

- `precision`: model dự đoán lớp đó đúng bao nhiêu phần trong các lần dự đoán lớp đó.
- `recall`: model tìm được bao nhiêu phần trong toàn bộ mẫu thật của lớp đó.
- `f1-score`: trung bình điều hòa giữa precision và recall.

Confusion matrix:

```python
print(confusion_matrix(...))
```

Ma trận này giúp xem model hay nhầm lớp nào với lớp nào.

Ví dụ nếu hàng `negative` bị dự đoán nhiều thành `neutral`, nghĩa là model chưa phân biệt tốt negative và neutral.

## 20. Inference

Hàm:

```python
def predict_one(text, image_path):
```

Nhận:

- một đoạn text
- một đường dẫn ảnh

Trả về:

```python
{
    "label": "positive",
    "probabilities": {
        "negative": ...,
        "neutral": ...,
        "positive": ...
    }
}
```

Đây là phần demo cho báo cáo hoặc presentation.

## 21. Cách tinh chỉnh để chạy ổn hơn

### Trường hợp 1: CUDA out of memory

Giảm batch size:

```python
batch_size = 8
```

Nếu vẫn lỗi:

```python
batch_size = 4
```

Bật freeze backbone:

```python
model = MultimodalFusionClassifier(
    text_model_name=CFG.text_model_name,
    num_classes=len(label2id),
    dropout=0.25,
    freeze_backbones=True,
).to(CFG.device)
```

### Trường hợp 2: DataLoader lỗi ảnh

Đặt:

```python
num_workers = 0
```

và đảm bảo có:

```python
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

Trong `__getitem__`, nên có:

```python
try:
    image = Image.open(row["image_path"]).convert("RGB")
except Exception:
    image = Image.new("RGB", (CFG.image_size, CFG.image_size), color=(0, 0, 0))
```

### Trường hợp 3: Train quá chậm

Các cách giảm thời gian:

```python
epochs = 2
batch_size = 8
freeze_backbones = True
```

Hoặc dùng ResNet18 thay ResNet50:

```python
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
```

Nếu đổi ResNet18, phần:

```python
image_hidden = resnet.fc.in_features
resnet.fc = nn.Identity()
```

vẫn giữ nguyên được.

### Trường hợp 4: Accuracy cao nhưng Macro-F1 thấp

Nguyên nhân thường là lệch nhãn.

Kiểm tra phân bố nhãn:

```python
print(data["label_name"].value_counts())
```

Cách cải thiện: dùng class weights.

Thêm trước khi tạo criterion:

```python
from sklearn.utils.class_weight import compute_class_weight

classes = np.array([0, 1, 2])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["label"].values,
)
weights = torch.tensor(weights, dtype=torch.float).to(CFG.device)
criterion = nn.CrossEntropyLoss(weight=weights)
```

### Trường hợp 5: Model overfit

Dấu hiệu:

```text
train_f1 tăng cao
val_f1 không tăng hoặc giảm
```

Cách xử lý:

- tăng dropout:

```python
dropout=0.35
```

- giảm learning rate:

```python
lr = 1e-5
```

- tăng augmentation ảnh nhẹ.

### Trường hợp 6: Model underfit

Dấu hiệu:

```text
train_f1 thấp
val_f1 thấp
```

Cách xử lý:

- tăng epoch:

```python
epochs = 6
```

- tắt freeze backbone:

```python
freeze_backbones=False
```

- tăng `max_len`:

```python
max_len = 128
```

## 22. Nên chỉnh gì trước?

Nếu bạn chỉ có ít thời gian, thứ tự tinh chỉnh nên là:

1. Chạy baseline với `epochs=4`, `batch_size=16`.
2. Nếu lỗi VRAM, giảm `batch_size=8`.
3. Nếu train quá chậm, dùng `freeze_backbones=True`.
4. Nếu Macro-F1 thấp, kiểm tra phân bố nhãn.
5. Nếu lệch nhãn, dùng `class weights`.
6. Nếu kết quả vẫn thấp, tăng `epochs` hoặc đổi text model sang `bert-base-uncased`.

## 23. Cách viết trong báo cáo

Bạn có thể mô tả ngắn gọn như sau:

```text
Bài toán sử dụng hai nguồn dữ liệu đa phương thức gồm văn bản và hình ảnh.
Mỗi mẫu được biểu diễn bởi hai nhánh độc lập. Nhánh văn bản sử dụng
BERT/DistilBERT để trích xuất đặc trưng ngữ nghĩa từ token [CLS]. Nhánh hình
ảnh sử dụng ResNet50 pretrained trên ImageNet, bỏ lớp fully-connected cuối để
lấy vector đặc trưng ảnh. Hai vector đặc trưng được chiếu về cùng kích thước,
sau đó nối lại bằng concatenation. Vector fusion được đưa qua một MLP classifier
để dự đoán một trong ba lớp sentiment: negative, neutral và positive.
```

## 24. Checklist trước khi nộp bài

- Code chạy hết từ đầu đến cuối.
- Có dùng 2 dataset Kaggle.
- Có text branch riêng.
- Có image branch riêng.
- Có fusion sau khi trích xuất feature.
- Có classifier head.
- Có metric `accuracy` và `macro-F1`.
- Có `classification_report`.
- Có `confusion_matrix`.
- Có giải thích kiến trúc trong markdown.
- Có xử lý lỗi ảnh hỏng.

## 25. Kết luận

Code này là một baseline multimodal rõ ràng, dễ giải thích và đúng trọng tâm:

```text
x_text  -> BERT encoder   -> z_text
x_image -> ResNet encoder -> z_image
concat(z_text, z_image)   -> classifier -> y
```

Nếu muốn nâng điểm, phần nên cải thiện nhất là đánh giá kết quả: phân tích Macro-F1 theo từng lớp, chỉ ra model nhầm lớp nào nhiều nhất, rồi đề xuất class weights hoặc attention fusion như hướng phát triển.

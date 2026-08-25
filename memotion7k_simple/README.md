# Memotion Dataset 7k - Simple Multimodal Classification

Thư mục này chứa một bài code Kaggle đơn giản cho bài toán phân loại cảm xúc/sentiment đa phương thức, chỉ sử dụng **1 dataset**:

```text
Memotion Dataset 7k
```

File notebook chính để chạy trên Kaggle:

```text
kaggle_memotion7k_simple_multimodal.ipynb
```

File code script cùng nội dung, dùng để đọc/sửa bằng VS Code:

```text
kaggle_memotion7k_simple_multimodal.py
```

Bạn nên dùng file `.ipynb` khi chạy trên Kaggle vì đây là định dạng Notebook chuẩn. File `.py` chỉ là bản script có cùng nội dung để dễ quản lý code trong Git.

- upload file `.ipynb` lên Kaggle rồi chạy từng cell
- mở file `.py` bằng VS Code nếu muốn đọc hoặc chỉnh code dạng script

## 1. Mục tiêu bài toán

Mỗi mẫu dữ liệu gồm:

```text
text + image + label
```

Model cần dự đoán sentiment:

```text
negative / neutral / positive
```

Kiến trúc tổng quát:

```text
Text  -> BERT nano/tiny -> text tokens   \
                                          cross-attention -> classifier -> sentiment
Image -> ResNet18       -> image tokens  /
```

Nói đơn giản:

- Text được xử lý bằng model ngôn ngữ.
- Image được xử lý bằng model ảnh.
- Text tokens dùng cross-attention để chú ý vào các vùng ảnh quan trọng.
- Classifier học từ vector đã fusion để dự đoán nhãn.

## 2. Dataset cần Add trên Kaggle

Dataset dùng:

```text
Memotion Dataset 7k
```

Link Kaggle:

```text
https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k
```

## 3. Cách chạy trên Kaggle

### Bước 1: Tạo Kaggle Notebook

Bạn có hai cách:

```text
Cách 1: Upload file kaggle_memotion7k_simple_multimodal.ipynb lên Kaggle
Cách 2: Tạo Notebook mới rồi copy từng cell từ file ipynb
```

Cách nên dùng là upload trực tiếp file `.ipynb`.

Nên bật GPU:

```text
Notebook settings -> Accelerator -> GPU
```

### Bước 2: Add Input

Ở panel bên phải:

```text
Notebook -> Input -> + Add Input
```

Tìm:

```text
Memotion Dataset 7k
```

Chọn dataset của `williamscott701`, rồi bấm Add.

### Bước 3: Kiểm tra đường dẫn dataset

Chạy cell này trên Kaggle:

```python
import os
print(os.listdir("/kaggle/input"))
```

Nếu output là:

```python
['memotion-dataset-7k']
```

thì code dùng được ngay.

Nếu output có dạng:

```python
['datasets']
```

thì chạy tiếp:

```python
print(os.listdir("/kaggle/input/datasets"))
```

Nếu thấy:

```python
['williamscott701']
```

chạy tiếp:

```python
print(os.listdir("/kaggle/input/datasets/williamscott701"))
```

Sau đó sửa `DATA_ROOT` trong code cho đúng.

Ví dụ:

```python
DATA_ROOT = Path("/kaggle/input/datasets/williamscott701/memotion-dataset-7k")
```

## 4. Cấu trúc code

Code được chia thành nhiều phần rõ ràng:

```text
1. Import thư viện
2. Config
3. Cố định seed
4. Kiểm tra dataset
5. Tìm metadata
6. Đọc metadata
7. Chọn cột image/text/label
8. Tìm đường dẫn ảnh
9. Chuẩn hóa label
10. Tạo dataframe sạch
11. Bỏ ảnh lỗi
12. Mã hóa label
13. Chia train/validation
14. Tokenizer và transform ảnh
15. Dataset class
16. DataLoader
17. Model multimodal
18. Loss và optimizer
19. Train một epoch
20. Evaluate
21. Training loop
22. Kết quả cuối cùng
23. Inference thử
24. Gợi ý chỉnh model
```

## 5. Giải thích các phần quan trọng

### 5.1. Config

Phần này chứa các tham số chính. Bản notebook mới không dùng `class CFG`, mà khai báo biến trực tiếp để dễ nhìn và dễ sửa:

```python
DATA_ROOT = Path("/kaggle/input/memotion-dataset-7k")
TEXT_MODEL = "prajjwal1/bert-tiny"
MAX_LEN = 96
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Bạn thường chỉnh các tham số này:

| Tham số | Ý nghĩa | Khi nào chỉnh |
|---|---|---|
| `DATA_ROOT` | đường dẫn dataset Kaggle | khi Kaggle đặt folder khác |
| `TEXT_MODEL` | model xử lý text | muốn dùng BERT chuẩn hoặc DistilBERT |
| `BATCH_SIZE` | số mẫu mỗi batch | lỗi CUDA memory thì giảm |
| `EPOCHS` | số vòng train | muốn train lâu hơn |
| `LEARNING_RATE` | learning rate | model học quá chậm/quá bất ổn |
| `NUM_WORKERS` | số worker load data | lỗi DataLoader thì đặt 0 |

### 5.2. Đọc metadata

Dataset có file metadata chứa tên ảnh, text và label. Code tìm các file:

```python
.csv, .xlsx, .xls
```

rồi chọn file có vẻ phù hợp nhất.

Nếu code chọn sai metadata, bạn sửa:

```python
metadata_path = Path("duong_dan_file_metadata")
```

### 5.3. Chọn cột image, text và label

Code tự đoán:

```python
image_col
text_col
label_col
```

Nếu tự đoán sai, sửa thủ công:

```python
image_col = "ten_cot_anh"
text_col = "ten_cot_text"
label_col = "ten_cot_label"
```

Bạn xem danh sách cột bằng output:

```python
print(list(df.columns))
```

### 5.4. Chuẩn hóa label

Memotion có thể có nhãn:

```text
very positive
positive
neutral
negative
very negative
```

Code gom về 3 lớp:

```text
very positive -> positive
positive      -> positive
neutral       -> neutral
negative      -> negative
very negative -> negative
```

Lý do: bài toán đơn giản hơn và classifier chỉ cần dự đoán 3 lớp.

### 5.5. Dataset class

Class:

```python
class MemotionDataset(Dataset):
```

mỗi lần lấy một sample sẽ trả về:

```python
{
    "input_ids": ...,
    "attention_mask": ...,
    "image": ...,
    "label": ...
}
```

Ý nghĩa:

- `input_ids`: token id của text
- `attention_mask`: mask cho BERT biết token nào là thật, token nào là padding
- `image`: tensor ảnh đã resize và normalize
- `label`: nhãn số 0, 1, 2

### 5.6. Text branch

Trong model:

```python
self.text_encoder = AutoModel.from_pretrained(text_model_name)
```

Mặc định:

```python
prajjwal1/bert-tiny
```

Đây là một BERT rất nhỏ. Trong bài báo cáo, bạn có thể gọi là **BERT tiny/nano-like**. Nó nhẹ hơn `bert-base-uncased`, phù hợp để chạy nhanh trên Kaggle.

Text branch biến một câu thành chuỗi token đặc trưng.

Trong `forward`:

```python
text_output = self.text_encoder(
    input_ids=input_ids,
    attention_mask=attention_mask,
)
text_tokens = text_output.last_hidden_state
```

`text_tokens` có dạng:

```text
batch_size x max_len x hidden_size
```

Nghĩa là mỗi token trong câu có một vector riêng.

### 5.7. Image branch

Trong model:

```python
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
self.image_encoder = nn.Sequential(*list(resnet.children())[:-2])
```

ResNet18 pretrained dùng để trích xuất đặc trưng ảnh.

Thay vì lấy một vector ảnh duy nhất, code bỏ phần cuối của ResNet để giữ lại feature map:

```text
batch_size x 512 x 7 x 7
```

Sau đó feature map được đổi thành chuỗi image tokens:

```text
batch_size x 49 x 512
```

Vì `7 x 7 = 49`, ta có 49 vùng ảnh. Mỗi vùng ảnh là một image token.

### 5.8. Cross-attention fusion

Code này dùng kỹ thuật:

```text
Cross-Attention Fusion
```

Không phải concat fusion đơn thuần.

Trước tiên, text tokens và image tokens được đưa về cùng kích thước 256:

```python
text_tokens = self.text_projection(text_tokens)
image_tokens = self.image_projection(image_tokens)
```

Sau đó dùng cross-attention:

```python
attended_text_tokens, attention_weights = self.cross_attention(
    query=text_tokens,
    key=image_tokens,
    value=image_tokens,
)
```

Ý nghĩa:

```text
Query = text tokens
Key   = image tokens
Value = image tokens
```

Nói dễ hiểu: mỗi token trong text sẽ học cách chú ý tới các vùng ảnh liên quan.

Ví dụ nếu text có từ mang cảm xúc mạnh, cross-attention có thể học xem vùng ảnh nào hỗ trợ cho cảm xúc đó.

Sau cross-attention, code lấy 3 loại đặc trưng:

```text
text_cls      : đặc trưng text gốc
attended_cls  : đặc trưng text sau khi attend vào image
image_global  : đặc trưng ảnh tổng quát
```

Rồi nối 3 vector này lại:

```python
fused_feature = torch.cat([text_cls, attended_cls, image_global], dim=1)
```

Đây là bước fusion cuối cùng sau cross-attention.

### 5.9. Classifier

Classifier nhận vector fusion và dự đoán 3 lớp:

```python
self.classifier = nn.Sequential(
    nn.Linear(256 * 3, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, num_classes),
)
```

Output là 3 logits:

```text
negative / neutral / positive
```

## 6. Kết quả sau khi chạy

Sau khi train xong, code in:

```text
Best validation accuracy
Best validation macro-F1
classification_report
confusion_matrix
```

Nên đánh giá chính bằng:

```text
macro-F1
```

vì dataset sentiment có thể bị lệch lớp.

## 7. Các lỗi thường gặp

### Không thấy dataset

Chạy:

```python
import os
print(os.listdir("/kaggle/input"))
```

rồi sửa:

```python
DATA_ROOT = Path("duong_dan_dung")
```

### Lỗi CUDA out of memory

Giảm batch size:

```python
BATCH_SIZE = 8
```

hoặc:

```python
BATCH_SIZE = 4
```

### Lỗi DataLoader hoặc ảnh hỏng

Đổi:

```python
NUM_WORKERS = 0
```

Code đã có xử lý ảnh lỗi:

```python
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

và fallback ảnh đen nếu ảnh không đọc được.

### Train quá chậm

Giảm:

```python
EPOCHS = 2
```

hoặc giữ ResNet18 thay vì đổi sang ResNet50.

## 8. Cách tinh chỉnh

### Dùng BERT chuẩn

Sửa:

```python
TEXT_MODEL = "bert-base-uncased"
```

`prajjwal1/bert-tiny` nhẹ hơn, `bert-base-uncased` có thể mạnh hơn nhưng tốn GPU hơn.

### Dùng DistilBERT

Sửa:

```python
TEXT_MODEL = "distilbert-base-uncased"
```

### Dùng ResNet50

Trong model, đổi:

```python
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
image_feature_size = 512
```

thành:

```python
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
image_feature_size = 2048
```

### Train lâu hơn

Sửa:

```python
EPOCHS = 5
```

hoặc:

```python
EPOCHS = 8
```

### Tăng độ dài text

Nếu text dài, sửa:

```python
MAX_LEN = 128
```

## 9. Đoạn mô tả cho báo cáo

Bạn có thể dùng đoạn này:

```text
Trong bài này, em sử dụng Memotion Dataset 7k cho bài toán phân loại sentiment
đa phương thức. Mỗi mẫu gồm ảnh meme, nội dung text/OCR và nhãn sentiment. Mô
hình được xây dựng với hai nhánh xử lý độc lập. Nhánh văn bản sử dụng một mô
hình BERT rất nhỏ (`prajjwal1/bert-tiny`) để trích xuất chuỗi đặc trưng token
từ text. Nhánh hình ảnh sử dụng ResNet18 pretrained trên ImageNet và lấy feature
map trước các lớp cuối để tạo các image tokens. Hai modality được kết hợp bằng
cross-attention fusion, trong đó text tokens đóng vai trò query, còn image tokens
đóng vai trò key và value. Vector sau fusion được đưa qua MLP classifier để phân
loại thành ba lớp negative, neutral và positive.
```

## 10. Tóm tắt ngắn

Thư mục này là bản đơn giản, dễ hiểu:

```text
Dataset: Memotion Dataset 7k
Text model: BERT tiny/nano-like (`prajjwal1/bert-tiny`)
Image model: ResNet18
Fusion: cross-attention
Classifier: MLP
Output: negative / neutral / positive
```

Đây là bản phù hợp để học, demo trên Kaggle và giải thích trong báo cáo.

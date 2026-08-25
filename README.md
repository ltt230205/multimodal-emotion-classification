# Multimodal Emotion/Sentiment Classification với BERT + ResNet

Project này là một bài code kiểu Kaggle cho bài toán phân loại cảm xúc/sentiment đa phương thức từ **text + image**. Ý tưởng chính là không trộn dữ liệu ngay từ đầu, mà xử lý từng loại input bằng một nhánh riêng:

- Text được đưa qua BERT/DistilBERT để lấy đặc trưng ngôn ngữ.
- Image được đưa qua ResNet50 pretrained để lấy đặc trưng hình ảnh.
- Hai vector đặc trưng sau đó được nối lại với nhau.
- Một classifier head nhỏ học trên vector fusion để dự đoán nhãn.

File chính:

```text
kaggle_multimodal_emotion_fusion.py
```

File này được viết theo kiểu notebook script, có các cell `# %%` nên có thể mở tốt trong VS Code, Jupyter hoặc copy từng phần sang Kaggle Notebook.

## Dataset sử dụng

Code được thiết kế để dùng 2 bộ dataset trên Kaggle:

1. **Memotion Dataset 7k**
   - Kaggle: `williamscott701/memotion-dataset-7k`
   - Gồm meme image, text/OCR và nhãn sentiment.

2. **Multimodal Sentiment Analysis CUET-NLP**
   - Kaggle competition: `multimodal-sentiment-analysis-cuet-nlp`
   - Gồm meme image, text và nhãn sentiment.

Hai dataset này đều phù hợp với bài toán multimodal vì mỗi mẫu có cả ảnh và chữ. Code chuẩn hóa nhãn về 3 lớp:

```text
negative -> 0
neutral  -> 1
positive -> 2
```

## Kiến trúc mô hình

Mô hình gồm 3 phần lớn.

### 1. Text branch

Text được tokenize bằng tokenizer của Hugging Face:

```python
AutoTokenizer.from_pretrained("distilbert-base-uncased")
```

Sau đó đưa vào text encoder:

```python
AutoModel.from_pretrained("distilbert-base-uncased")
```

Vector được lấy từ token đầu tiên:

```python
text_feat = text_outputs.last_hidden_state[:, 0, :]
```

Với BERT/DistilBERT, token đầu tiên thường được dùng như biểu diễn tổng quát của câu.

### 2. Image branch

Image được resize về `224x224`, normalize theo ImageNet, rồi đưa vào ResNet50 pretrained:

```python
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet.fc = nn.Identity()
```

Dòng `resnet.fc = nn.Identity()` bỏ lớp phân loại gốc của ResNet. Khi đó ResNet chỉ còn vai trò trích xuất đặc trưng ảnh.

### 3. Fusion + classifier

Đặc trưng text và image được chiếu về cùng kích thước 256:

```python
text_feat = self.text_proj(text_feat)
image_feat = self.image_proj(image_feat)
```

Sau đó nối lại:

```python
fused = torch.cat([text_feat, image_feat], dim=1)
```

Classifier nhận vector fusion 512 chiều và dự đoán 3 lớp:

```python
logits = self.classifier(fused)
```

Tóm tắt công thức:

```text
h_text = BERT(text)[CLS]
h_img  = ResNet50(image)
z_text = MLP_text(h_text)
z_img  = MLP_img(h_img)
z      = concat(z_text, z_img)
y_hat  = MLP_classifier(z)
```

## Cách chạy trên Kaggle

### Bước 1: Tạo Notebook

Tạo một Kaggle Notebook mới và bật GPU:

```text
Notebook settings -> Accelerator -> GPU
```

### Bước 2: Add Data

Trong Kaggle Notebook, bấm **Add Data** rồi thêm:

```text
Memotion Dataset 7k
Multimodal Sentiment Analysis CUET-NLP
```

Sau khi add, dataset thường nằm trong:

```text
/kaggle/input/memotion-dataset-7k
/kaggle/input/multimodal-sentiment-analysis-cuet-nlp
```

Nếu Kaggle đặt tên folder khác, sửa trong class `CFG`:

```python
memotion_root = Path("/kaggle/input/memotion-dataset-7k")
cuet_root = Path("/kaggle/input/multimodal-sentiment-analysis-cuet-nlp")
```

### Bước 3: Copy hoặc import code

Cách đơn giản nhất là copy nội dung file:

```text
kaggle_multimodal_emotion_fusion.py
```

sang Kaggle Notebook và chạy từ trên xuống dưới.

## Cấu trúc code

### Config

Các tham số chính nằm trong class `CFG`:

```python
class CFG:
    seed = 42
    max_len = 96
    image_size = 224
    batch_size = 16
    epochs = 4
    lr = 2e-5
    text_model_name = "distilbert-base-uncased"
```

Một số chỉnh sửa thường dùng:

- Nếu thiếu GPU memory, giảm `batch_size`.
- Nếu muốn train lâu hơn, tăng `epochs`.
- Nếu muốn dùng BERT chuẩn, đổi:

```python
text_model_name = "bert-base-uncased"
```

### Hàm load dataset

Hàm:

```python
load_multimodal_dataset(root, dataset_name)
```

có nhiệm vụ:

- Tìm file metadata như `.csv`, `.xlsx`, `.json`.
- Tự đoán cột text.
- Tự đoán cột image.
- Tự đoán cột label.
- Tìm đường dẫn ảnh thật trong folder dataset.
- Chuẩn hóa label về `negative`, `neutral`, `positive`.

Nếu dataset trên Kaggle đổi tên cột và code không tự đoán được, lỗi sẽ in ra danh sách columns. Khi đó sửa thủ công trong hàm `load_multimodal_dataset`.

### Dataset class

Class:

```python
MemeMultimodalDataset
```

trả về một sample gồm:

```python
{
    "input_ids": ...,
    "attention_mask": ...,
    "image": ...,
    "label": ...
}
```

Đây là phần đảm bảo mỗi batch có đủ input cho cả text branch và image branch.

### Model class

Class:

```python
MultimodalFusionClassifier
```

là phần quan trọng nhất của project. Nó định nghĩa:

- `self.text_encoder`: BERT/DistilBERT.
- `self.image_encoder`: ResNet50.
- `self.text_proj`: chiếu feature text.
- `self.image_proj`: chiếu feature image.
- `self.classifier`: phân loại sau fusion.

### Training

Code dùng:

```python
CrossEntropyLoss
AdamW
Linear warmup scheduler
gradient clipping
```

Metric chính:

```text
accuracy
macro-F1
classification report
confusion matrix
```

Macro-F1 hữu ích hơn accuracy nếu dataset bị lệch nhãn.

## Output sau khi train

Model tốt nhất được lưu thành:

```text
best_multimodal_fusion_model.pt
```

File checkpoint chứa:

- `model_state_dict`
- `label2id`
- `id2label`
- một phần config cơ bản

## Inference

Cuối file có hàm:

```python
predict_one(text, image_path)
```

Hàm này nhận một đoạn text và đường dẫn ảnh, sau đó trả về:

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

## Vì sao dùng DistilBERT thay vì BERT-base?

Trong code mặc định là:

```python
distilbert-base-uncased
```

DistilBERT nhẹ hơn, chạy nhanh hơn và ít tốn VRAM hơn trên Kaggle GPU. Về mặt báo cáo, có thể viết là sử dụng BERT-family encoder cho text. Nếu đề yêu cầu đúng BERT-base, chỉ cần đổi config:

```python
text_model_name = "bert-base-uncased"
```

## Các lỗi thường gặp

### Không thấy dataset trong `/kaggle/input`

Nguyên nhân thường là chưa Add Data hoặc Kaggle đặt tên folder khác.

Cách xử lý:

```python
print(os.listdir("/kaggle/input"))
```

rồi sửa lại đường dẫn trong `CFG`.

### Không tự dò được cột text/image/label

Code sẽ báo lỗi kèm danh sách columns. Khi đó mở file metadata, xem tên cột thật, rồi sửa đoạn:

```python
text_col = ...
image_col = ...
label_col = ...
```

trong hàm `load_multimodal_dataset`.

### CUDA out of memory

Giảm batch size:

```python
batch_size = 8
```

hoặc freeze backbone:

```python
freeze_backbones=True
```

khi khởi tạo model.

### Train chậm

Có thể:

- Giảm `epochs`.
- Dùng `distilbert-base-uncased`.
- Freeze BERT và ResNet trong vài epoch đầu.
- Giảm kích thước dữ liệu để demo.

## Hướng phát triển thêm

Một số cách cải thiện nếu muốn nâng cấp bài:

- Dùng attention fusion thay vì concat.
- Fine-tune từng branch riêng trước rồi mới fusion.
- Dùng class weights nếu nhãn lệch nhiều.
- Thêm OCR nếu dataset chỉ có ảnh nhưng chưa có text.
- Thử EfficientNet hoặc ViT cho image branch.
- Thử RoBERTa hoặc DeBERTa cho text branch.

## Tóm tắt ngắn gọn

Project này thực hiện đúng pipeline multimodal cơ bản:

```text
Text  -> BERT      -> text feature  \
                                      concat -> classifier -> label
Image -> ResNet50  -> image feature /
```

Đây là kiến trúc dễ giải thích trong báo cáo, dễ chạy trên Kaggle, và thể hiện rõ yêu cầu “mỗi modality xử lý riêng rồi mới fusion”.

# Twitter Multimodal Sentiment Analysis

Thư mục này là bài Kaggle riêng cho dataset:

```text
https://www.kaggle.com/datasets/dunyajasim/twitter-dataset-for-sentiment-analysis
```

File chạy trên Kaggle:

```text
kaggle_twitter_multimodal_sentiment.ipynb
```

## Dataset dùng những gì?

Dataset gồm 2 phần:

```text
1. File Excel .xlsx
2. Folder ảnh
```

File Excel có 3 cột chính:

```text
File Name | Caption | LABEL
```

Ý nghĩa:

- `File Name`: tên file text, ví dụ `1.txt`, `10.txt`, `100.txt`.
- `Caption`: nội dung tweet/caption dùng làm input text.
- `LABEL`: nhãn sentiment, ví dụ `negative`, `neutral`, `positive`.

Folder ảnh có cấu trúc gần giống:

```text
Images/
  Images/
    Negative/
      1.jpg
      10.jpg
    Positive/
      100.jpg
    Neutral/
      1000.jpg
```

Notebook sẽ lấy stem của `File Name`:

```text
1.txt -> 1
```

rồi tìm ảnh có stem tương ứng:

```text
1.jpg
```

## Ý tưởng model

Notebook dùng cả text và image:

```text
Caption -> BERT tiny -> text feature
Image   -> ResNet18  -> image feature
text feature + image feature -> classifier -> LABEL
```

Text branch:

```text
prajjwal1/bert-tiny
```

Image branch:

```text
ResNet18 pretrained ImageNet
```

Fusion đang dùng:

```text
Concat fusion
```

Tức là nối vector đặc trưng text và vector đặc trưng ảnh lại:

```python
fused_feature = torch.cat([text_feature, image_feature], dim=1)
```

sau đó đưa qua classifier.

## Cách Add Input trên Kaggle

1. Mở notebook Kaggle.
2. Nhìn cột bên phải, mục **Input**.
3. Bấm **Add Input**.
4. Tìm:

```text
twitter-dataset-for-sentiment-analysis
```

5. Chọn dataset của `dunyajasim`.

Với Kaggle của bạn, path đang là:

```text
/kaggle/input/datasets/dunyajasim/twitter-dataset-for-sentiment-analysis
```

Notebook đã trỏ sẵn vào path này.

## Cách chạy

Chạy notebook từ trên xuống dưới:

1. Import thư viện.
2. Kiểm tra dataset path.
3. Đọc file `.xlsx`.
4. Dùng trực tiếp 3 cột `File Name`, `Caption`, `LABEL`.
5. Tìm ảnh tương ứng.
6. Làm sạch text và label.
7. Lọc ảnh lỗi.
8. Chia train/validation.
9. Tạo Dataset và DataLoader.
10. Train model BERT tiny + ResNet18 + fusion.
11. Xem accuracy, macro F1, classification report.
12. Xem confusion matrix.

## Chỉnh cho chạy nhanh hơn

Giảm số epoch:

```python
for epoch in range(3):
```

đổi thành:

```python
for epoch in range(1):
```

Giảm batch size nếu bị hết GPU:

```python
batch_size=16
```

đổi thành:

```python
batch_size=8
```

## Output

Model tốt nhất được lưu ở:

```text
/kaggle/working/twitter_multimodal_best_model.pt
```

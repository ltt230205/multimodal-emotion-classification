# Twitter Sentiment Analysis bằng BERT Tiny

Thư mục này là một bài Kaggle riêng cho dataset:

```text
https://www.kaggle.com/datasets/dunyajasim/twitter-dataset-for-sentiment-analysis
```

File chính để chạy trên Kaggle:

```text
kaggle_twitter_sentiment_bert.ipynb
```

## Ý tưởng bài toán

Bài này làm **phân loại sentiment từ text Twitter**.

Pipeline:

```text
Tweet text
-> làm sạch text đơn giản
-> tokenizer của BERT
-> BERT tiny
-> classification head
-> sentiment label
```

Model dùng:

```text
prajjwal1/bert-tiny
```

Đây là một bản BERT nhỏ, nhẹ hơn `bert-base-uncased`, phù hợp để chạy demo/fine-tune nhanh trên Kaggle.

## Cách Add Input trên Kaggle

1. Mở notebook Kaggle.
2. Nhìn cột bên phải, mục **Input**.
3. Bấm **Add Input**.
4. Tìm:

```text
twitter-dataset-for-sentiment-analysis
```

5. Chọn dataset của `dunyajasim`.
6. Sau khi add xong, dataset thường nằm ở:

```text
/kaggle/input/twitter-dataset-for-sentiment-analysis
```

## Cách chạy

Chạy lần lượt từ trên xuống dưới:

1. Import thư viện.
2. Kiểm tra dataset.
3. Đọc CSV.
4. Chọn cột text và label.
5. Làm sạch dữ liệu.
6. Chia train/validation.
7. Fine-tune BERT.
8. Xem classification report và confusion matrix.
9. Dự đoán thử một tweet.

## Nếu notebook chọn sai cột

Notebook có cell:

```python
text_column = find_column(...)
label_column = find_column(...)
```

Nếu kết quả in ra sai, sửa trực tiếp:

```python
text_column = "tên_cột_text"
label_column = "tên_cột_label"
```

Sau đó chạy lại từ cell đó trở xuống.

## Chỉnh cho chạy nhanh hơn

Giảm số epoch:

```python
for epoch in range(3):
```

đổi thành:

```python
for epoch in range(1):
```

Giảm số mẫu:

```python
if len(data) > 30000:
    data = data.sample(n=30000, random_state=42).reset_index(drop=True)
```

đổi `30000` thành `5000` hoặc `10000`.

## Chỉnh cho model mạnh hơn

Đổi:

```python
prajjwal1/bert-tiny
```

thành:

```python
distilbert-base-uncased
```

hoặc:

```python
bert-base-uncased
```

Lưu ý: model càng lớn thì chạy càng lâu và tốn GPU hơn.

## Output

Model tốt nhất được lưu ở:

```text
/kaggle/working/twitter_sentiment_bert_model
```

Trên Kaggle, bạn có thể tải folder này trong phần Output sau khi notebook chạy xong.

## Ghi chú

Bài này là bản **text-only** vì dataset Twitter sentiment này chủ yếu phục vụ phân loại sentiment từ văn bản. Nếu dataset bạn tải về có thêm ảnh, có thể mở rộng thành bài multimodal bằng cách thêm nhánh ResNet và fusion giống bài Memotion.

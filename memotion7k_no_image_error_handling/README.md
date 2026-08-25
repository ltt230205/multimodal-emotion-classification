# Memotion 7k - Không xử lý ảnh lỗi

Thư mục này chứa bản notebook train bình thường trên Memotion Dataset 7k, nhưng **không có bước xử lý ảnh lỗi**.

File chạy trên Kaggle:

```text
kaggle_memotion7k_no_image_error_handling.ipynb
```

Khác với bản `memotion7k_simple`:

- Không dùng `ImageFile.LOAD_TRUNCATED_IMAGES = True`.
- Không có hàm `is_valid_image(...)`.
- Không lọc ảnh hỏng trước khi train.
- Không dùng `try/except` khi đọc ảnh.
- Không thay ảnh lỗi bằng ảnh đen.

Ảnh được đọc trực tiếp:

```python
image = Image.open(image_path).convert("RGB")
```

Mục đích của bản này là chạy pipeline bình thường nhất có thể để bạn so sánh với bản có xử lý ảnh lỗi.

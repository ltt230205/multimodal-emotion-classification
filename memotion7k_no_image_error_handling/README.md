# Memotion 7k - Bản đã lọc ảnh lỗi để train được

Thư mục này chứa bản notebook train trên Memotion Dataset 7k và đã thêm bước **lọc ảnh lỗi trước khi train**.

File chạy trên Kaggle:

```text
kaggle_memotion7k_no_image_error_handling.ipynb
```

Notebook này xử lý lỗi:

```text
OSError: image file is truncated
```

Bằng cách:

- Đọc thẳng file metadata `labels.csv`.
- Tìm đường dẫn ảnh thật từ cột `image_name`.
- Bỏ mẫu thiếu ảnh hoặc thiếu label.
- Kiểm tra từng ảnh bằng `is_valid_image(...)`.
- Loại các ảnh hỏng khỏi dataframe trước khi tạo `train_loader`.

Sau khi lọc, Dataset vẫn đọc ảnh bình thường:

```python
image = Image.open(image_path).convert("RGB")
```

Nghĩa là code không thay ảnh lỗi bằng ảnh đen, mà bỏ hẳn các mẫu ảnh lỗi để quá trình train không bị dừng giữa chừng.

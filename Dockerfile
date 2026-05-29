# 1. Sử dụng Python 3.11 mỏng nhẹ (Slim)
FROM python:3.11-slim

# 2. Cấu hình biến môi trường để Python không tạo file .pyc và log ra terminal ngay lập tức
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# 3. Tạo thư mục làm việc
WORKDIR /app

# 4. Cài đặt công cụ hệ thống (Gộp lại và dọn dẹp để giảm dung lượng)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Tận dụng Cache cho Pip (Kỹ thuật Két sắt bạn vừa học)
COPY requirements.txt .
# 5. Cài đặt thư viện Python
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt
# 6. Copy code vào sau cùng để không làm hỏng Cache của bước trên
# Lưu ý: Nhớ dùng .dockerignore để loại bỏ venv, __pycache__
COPY . .

# 7. Copy file Key (Nếu bạn chọn cách nhét thẳng vào Image)
# Nếu đã có trong bước 'COPY . .' thì không cần dòng này, 
# trừ khi bạn để vertex-key.json trong .dockerignore
COPY vertex-key.json .

# 8. Khởi chạy với định dạng JSON (Khuyên dùng để Docker quản lý tiến trình tốt hơn)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
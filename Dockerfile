# 1. Dùng bản Python 3.10 mỏng nhẹ
FROM python:3.11-slim

# 2. Tạo thư mục làm việc
WORKDIR /app

# 3. Cài đặt các công cụ hệ thống và thư viện lõi cho FAISS / AI
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy file cấu hình thư viện
COPY requirements.txt .

# 5. Cài đặt thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy toàn bộ code
COPY . .

# 7. Khai báo Port
ENV PORT=8080

# 8. Lệnh khởi chạy ứng dụng
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
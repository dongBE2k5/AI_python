from locust import HttpUser, task, between
import json

class ChatBotUser(HttpUser):
    wait_time = between(1, 5)  # Thời gian chờ giữa các request từ 1 đến 5 giây
    host = "http://localhost:8080"  # Giả định server chạy trên localhost:8080

    @task
    def chat_with_bot(self):
        # Danh sách các prompt mẫu để test chatbot
        prompts = [
            "Chất lượng dạy hoc của trường như thế nào?",
            "Tôi cần thông tin về trường Cao đẳng Công nghệ Thủ Đức.",
            "Làm thế nào để đăng ký học?",
            "Các ngành học chính của trường là gì?",
            "Bạn có thể giải thích về chương trình đào tạo không?",
            "Thông tin liên hệ của trường?",
            "Lịch học như thế nào?",
            "Yêu cầu tuyển sinh là gì?",
            "Có hỗ trợ tài chính không?",
            "Bạn có thể giới thiệu về giảng viên không?"
        ]

        # Chọn một prompt ngẫu nhiên
        import random
        prompt = random.choice(prompts)

        # Chuẩn bị dữ liệu JSON
        data = {
            "session_id": "test_session",
            "prompt": prompt
        }

        # Gửi POST request đến endpoint /api/chat
        with self.client.post("/api/chat", json=data, catch_response=True) as response:
            if response.status_code == 200:
                # Kiểm tra xem response có chứa "reply" không
                try:
                    json_response = response.json()
                    if "reply" in json_response:
                        response.success()
                    else:
                        response.failure("Response không chứa 'reply'")
                except json.JSONDecodeError:
                    response.failure("Response không phải JSON hợp lệ")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
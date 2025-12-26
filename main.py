import os
import faiss
import requests
import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

STORAGE_DIR = "storage"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
YOUR_SITE_URL = os.getenv("YOUR_SITE_URL")
YOUR_SITE_NAME = os.getenv("YOUR_SITE_NAME")

# --- ĐỒNG BỘ EMBEDDING VỚI build_index.py ---
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L12-v2"
)

# --- HÀM MỞ RỘNG CÂU HỎI (QUERY EXPANSION) ---
def expand_queries(original_query):
    """Sử dụng LLM để tạo ra 6 biến thể của câu hỏi nhằm tối ưu hóa việc tìm kiếm."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": f"{YOUR_SITE_URL}",
        "X-Title": f"{YOUR_SITE_NAME}",
    }
    
    prompt_expansion = f"""Bạn là một chuyên gia tra cứu tài liệu tại Trường Cao đẳng Công nghệ Thủ Đức (TDC). 
    Từ câu hỏi gốc của người dùng, hãy tạo ra 6 câu hỏi biến thể có cùng ý nghĩa nhưng đầy đủ từ khóa hơn (ví dụ thêm năm học 2025-2026, tên trường) để hỗ trợ tìm kiếm chính xác.
    
    Câu hỏi gốc: "{original_query}"
    
    Yêu cầu trả về: CHỈ trả về danh sách các câu hỏi, mỗi câu một dòng, không đánh số, không thêm văn bản dẫn nhập."""

    payload = {
        "model": "deepseek/deepseek-r1-0528:free", # Dùng bản free để tiết kiệm token mở rộng
        "messages": [{"role": "user", "content": prompt_expansion}],
        "max_tokens": 500,
        "temperature": 0.8
    }
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(payload))
        res_json = response.json()
        content = res_json['choices'][0]['message']['content'].strip()
        lines = content.split('\n')
        expanded_list = [original_query] + [line.strip() for line in lines if line.strip()]
        return expanded_list[:6] # Lấy tối đa 6 câu (gốc + 5 phụ)
    except Exception as e:
        print(f"⚠️ Lỗi mở rộng câu hỏi: {e}")
        return [original_query]

@asynccontextmanager
async def lifespan(app: FastAPI):
    faiss_path = os.path.join(STORAGE_DIR, "faiss.index")
    
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"Không thấy {faiss_path}. Chạy 'python build_index.py' trước!")

    # 1. Load FAISS & Index
    faiss_index = faiss.read_index(faiss_path)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context)

    # 2. Lưu retriever vào app.state (Mỗi câu truy xuất 2 đoạn để bao phủ tốt hơn)
    app.state.retriever = index.as_retriever(similarity_top_k=2)
    
    print("✅ Hệ thống RAG (Multi-Query) đã sẵn sàng!")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str

@app.post("/api/chat")
async def chat(req: ChatRequest):
    # Bước 1: Sinh các biến thể câu hỏi
    queries = expand_queries(req.prompt)
    print(f"🔍 Đang truy xuất với các câu hỏi: {queries}")

    # Bước 2: Truy xuất tài liệu cho tất cả các câu hỏi
    all_nodes = []
    for q in queries:
        nodes = app.state.retriever.retrieve(q)
        all_nodes.extend(nodes)

    # Bước 3: Loại bỏ nội dung trùng lặp (Deduplication)
    unique_contents = {}
    for node in all_nodes:
        # Dùng content làm key để tránh trùng lặp thông tin
        unique_contents[node.get_content()[:200]] = node.get_content()
    
    context_text = "\n---\n".join(unique_contents.values())

    # Bước 4: Xây dựng Prompt cuối cùng
    system_prompt = f"""   Bạn là một trợ lý tư vấn chuyên nghiệp, tận tâm.
    Nhiệm vụ DUY NHẤT của bạn là giải đáp thắc mắc dựa trên thông tin trong [NGỮ CẢNH TÀI LIỆU] được cung cấp dưới đây.
---
  [NGỮ CẢNH TÀI LIỆU START]
{context_text}
    [NGỮ CẢNH TÀI LIỆU END]
---

  QUY TẮC TRẢ LỜI NGHIÊM NGẶT (BẮT BUỘC TUÂN THỦ):
    1. **PHẠM VI:** CHỈ được sử dụng thông tin có trong [NGỮ CẢNH TÀI LIỆU]. 
       - TUYỆT ĐỐI KHÔNG sử dụng kiến thức bên ngoài (kiến thức huấn luyện trước đó) để thêm thắt thông tin không có trong văn bản.
       - Nếu thông tin người dùng hỏi KHÔNG có trong tài liệu, hãy trả lời thẳng thắn và lịch sự: "Xin lỗi, trong tài liệu tôi được cung cấp hiện không có thông tin về vấn đề này."

    2. **TỔNG HỢP THÔNG TIN:**
       - Đối với câu hỏi về số lượng hoặc danh sách (ví dụ: "có bao nhiêu ngành", "gồm những gì"), hãy trả lời tổng quát và tự nhiên (ví dụ: "Dạ, theo tài liệu thì trường có khoảng X ngành, tiêu biểu là A, B, C..."). 
       - Tránh liệt kê danh sách dài dòng trừ khi được yêu cầu cụ thể.

    3. **PHONG CÁCH:** 
       - Giọng văn tự nhiên, thân thiện, giống người tư vấn thật sự.
       - Dùng từ ngữ lịch sự (Dạ, Thưa, Bạn...).

    Hãy nhớ: Sự chính xác theo tài liệu là ưu tiên hàng đầu."""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": f"{YOUR_SITE_URL}",
        "X-Title": f"{YOUR_SITE_NAME}",
    }
    print("🤖 Đang gửi yêu cầu đến OpenRouter LLM...", system_prompt)
    payload = {
        "model": "google/gemini-2.5-flash-lite", 
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{system_prompt}\n\nCâu hỏi của người dùng: {req.prompt}"
                    }
                ]
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.2 # Giảm xuống để đảm bảo tính chính xác cho RAG
    }

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload)
    )

    result = response.json()
    
    try:
        reply = result['choices'][0]['message']['content']
    except (KeyError, IndexError):
        reply = "Có lỗi khi kết nối với AI: " + str(result)

    return {"reply": reply}

@app.get("/")
def health():
    return {"status": "ok"}
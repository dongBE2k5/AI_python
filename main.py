import os
import sys
import asyncio
import faiss
import json
import shutil
import subprocess
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Đổi từ OpenAI sang AsyncOpenAI để không làm treo server
from openai import AsyncOpenAI 

# --- IMPORT GOOGLE GENAI (UNIFIED SDK) ---
from google import genai
from google.genai import types 

# --- LlamaIndex Core ---
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load biến môi trường từ file .env
load_dotenv()

# --- CẤU HÌNH THƯ MỤC & FILE ---
STORAGE_DIR = "storage"
DATA_DIR = "data"
HISTORY_FILE = "chat_history.json"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

# --- KHỞI TẠO CÁC CLIENT AI ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# 1. Khởi tạo Client OpenAI (cho OpenRouter nếu bạn cần dùng)
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# 2. Khởi tạo Client Gemini (Sử dụng Vertex AI qua Unified SDK)
project_id = os.getenv("PROJECT_ID")
location = os.getenv("LOCATION") # Thường là "us-central1"

genai_client = genai.Client(
    vertexai=True, 
    project=project_id, 
    location=location
)

# --- CẤU HÌNH EMBEDDING MODEL (Local) ---
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    embed_batch_size=8
)

def load_all_history():
    """Đọc toàn bộ dữ liệu của tất cả mọi người từ file JSON"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Nếu đọc lên mà thấy là List (cấu trúc cũ), thì reset thành Dict
                if isinstance(data, list):
                    return {}
                return data
        except: 
            return {}
    return {}

# --- HÀM XỬ LÝ LỊCH SỬ (JSON) ---
def load_history(session_id: str):
    """Trích xuất lịch sử của đúng người dùng đang chat"""
    all_history = load_all_history()
    # Lấy data của session_id, nếu chưa có thì trả về mảng rỗng []
    return all_history.get(session_id, [])


def save_history(session_id: str, messages: list):
    """Cập nhật lại đoạn chat của người dùng vào file chung"""
    all_history = load_all_history()
    
    # Ghi đè lịch sử mới của người này vào tổng thể
    all_history[session_id] = messages
    
    # Lưu toàn bộ cục tổng thể đó xuống lại file
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_history, f, ensure_ascii=False, indent=4)

# --- HÀM LOAD DỮ LIỆU KNOWLEDGE BASE ---
def load_knowledge_base(app_instance: FastAPI):
    print("🔄 Đang nạp dữ liệu từ Storage vào RAM...")
    faiss_path = os.path.join(STORAGE_DIR, "faiss.index")
    
    if not os.path.exists(faiss_path):
        print("⚠️ Chưa có dữ liệu index. Hãy gọi API /api/admin/rebuild-index")
        return False

    try:
        faiss_index = faiss.read_index(faiss_path)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
        
        app_instance.state.retriever = index.as_retriever(similarity_top_k=6)
        print("✅ Đã nạp dữ liệu thành công!")
        return True
    except Exception as e:
        print(f"❌ Lỗi nạp dữ liệu: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_knowledge_base(app)
    yield

# --- KHỞI TẠO FASTAPI ---
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    prompt: str

# --- TÍNH NĂNG: QUERY EXPANSION (ĐÃ THÊM TRẢ VỀ TOKEN) ---
async def expand_queries(original_query):
    prompt_expansion = f"""Bạn là chuyên gia tra cứu. 
    Viết lại câu hỏi gốc thành 3 biến thể tìm kiếm (xử lý từ đồng nghĩa Tiếng Trung/Hoa, Anh/TOEIC).
    Câu hỏi gốc: "{original_query}"
    CHỈ trả về danh sách câu hỏi, mỗi câu một dòng và bằng tiếng việt."""

    try:
        response = await genai_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_expansion,
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.3
            )
        )
        
        content = response.text.strip()
        lines = content.split('\n')
        
        # Lấy số token đã dùng cho Query Expansion
        usage = response.usage_metadata
        expansion_tokens = usage.total_token_count if usage else 0
        
        print(f"📝 Query Expansion (Dùng {expansion_tokens} tokens): \n{content}")
        return [original_query] + [line.strip() for line in lines if line.strip()], expansion_tokens
        
    except Exception as e:
        print(f"❌ Lỗi gọi Gemini (Query Expansion): {e}")
        return [original_query], 0

# --- API: CHAT CHÍNH (/api/chat) ---
@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not hasattr(app.state, 'retriever'):
        raise HTTPException(status_code=400, detail="Hệ thống chưa có dữ liệu. Vui lòng Rebuild Index.")

    # 1. Tìm kiếm dữ liệu (RAG) - Lấy thêm thông tin token từ expansion
    queries, expansion_tokens = await expand_queries(req.prompt)
    print(f"🔍 Đang tìm kiếm với các queries: {queries}")

    all_nodes = []
    for q in queries:
        nodes = app.state.retriever.retrieve(q)
        all_nodes.extend(nodes)

    unique_contents = {}
    for node in all_nodes:
        unique_contents[node.get_content()[:200]] = node.get_content()
    
    context_text = "\n---\n".join(unique_contents.values())[:12000]

    # 2. Xử lý Lịch sử Chat
    history = load_history(req.session_id)
    # 3. Chuẩn bị Format cho Gemini
    system_text = f"Bạn là trợ lý tư vấn sinh viên TDC (Trường Cao đẳng Công nghệ Thủ Đức). Dựa vào nội dung này để trả lời: {context_text}. Chỉ trả lời câu hỏi dựa trên tài liệu được cung cấp."
    
    gemini_messages = []
    for msg in history[-10:]:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})
    
    gemini_messages.append({"role": "user", "parts": [{"text": req.prompt}]})
    history.append({"role": "user", "content": req.prompt})
    
    # 4. Gọi AI Gemini qua Vertex AI
    try:
        print("🤖 Đang gọi Vertex AI (Unified SDK)...")
        response = await genai_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=gemini_messages,
            config=types.GenerateContentConfig(
                system_instruction=system_text,
                temperature=0.8,
            )
        )
        
        answer = response.text 
        
        # --- ĐẾM TOKEN CHÍNH TẠI ĐÂY ---
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage else 0
        completion_tokens = usage.candidates_token_count if usage else 0
        chat_tokens = usage.total_token_count if usage else 0
        
        # Tổng token của cả Request (Mở rộng câu hỏi + Chat chính)
        total_tokens = expansion_tokens + chat_tokens
        
        # 5. Lưu Lịch sử 
        history.append({"role": "assistant", "content": answer})
        save_history(req.session_id, history)
        print("Số token đã dùng - Expansion:", expansion_tokens, "| Chat:", chat_tokens, "| Tổng:", total_tokens)
        print("contens gửi",gemini_messages)
        # Trả về câu trả lời kèm theo thống kê token
        return {
            "reply": answer,
            "usage": {
                "expansion_tokens": expansion_tokens,
                "chat_prompt_tokens": prompt_tokens,
                "chat_completion_tokens": completion_tokens,
                "chat_total_tokens": chat_tokens,
                "total_request_tokens": total_tokens
            }
        }

    except Exception as e:
        print(f"❌ Lỗi AI: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi AI: {str(e)}")

# --- API: ADMIN QUẢN LÝ FILE & INDEX ---
@app.get("/api/admin/files")
async def list_files():
    return {"files": os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []}

@app.get("/api/filesdata")
async def list_filesdata():
    return {"files": os.listdir(STORAGE_DIR) if os.path.exists(STORAGE_DIR) else []}

@app.delete("/api/admin/files/{filename}")
async def delete_file(filename: str):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        return {"message": "Đã xóa"}
    raise HTTPException(status_code=404, detail="Không tìm thấy file")

@app.post("/api/admin/upload")
async def upload_file(file: UploadFile = File(...)):
    path = os.path.join(DATA_DIR, file.filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "Upload thành công"}

@app.post("/api/admin/rebuild-index")
async def rebuild_index():
    print("🚀 Đang chạy Rebuild Index (Async)...")
    try:
        loop = asyncio.get_running_loop()
        def run_script():
            my_env = os.environ.copy()
            my_env["PYTHONIOENCODING"] = "utf-8"
            return subprocess.run(
                [sys.executable, "build_index.py"], 
                capture_output=True, text=True, encoding='utf-8', env=my_env
            )

        process = await loop.run_in_executor(None, run_script)
        
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Lỗi Script: {process.stderr}")
        
        if load_knowledge_base(app):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            return {"message": "Cập nhật thành công! Đã reset lịch sử chat.", "output": process.stdout}
        else:
            raise HTTPException(status_code=500, detail="Nạp dữ liệu vào RAM thất bại")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- KHỞI CHẠY SERVER ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
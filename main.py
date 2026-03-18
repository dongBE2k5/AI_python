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

from google import genai
from google.genai import types 

# --- LlamaIndex Core ---
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# --- CẤU HÌNH ---
STORAGE_DIR = "storage"
DATA_DIR = "data"
HISTORY_FILE = "chat_history.json"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
key_gemini = os.getenv("GOOGLE_API_KEY")

# --- KHỞI TẠO CLIENT OPENAI (Async) ---
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Khởi tạo Client Gemini
genai_client = genai.Client(api_key=key_gemini)

# --- 1. EMBEDDING MODEL (Local) ---
# LƯU Ý: Model này cần >2GB RAM để chạy. 
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    embed_batch_size=8
)

# --- XỬ LÝ LỊCH SỬ (JSON) ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: 
            return []
    return []

def save_history(messages):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)

# --- HÀM LOAD DỮ LIỆU ---
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

# --- QUERY EXPANSION (Đã chuyển thành Async) ---
async def expand_queries(original_query):
    prompt_expansion = f"""Bạn là chuyên gia tra cứu. 
    Viết lại câu hỏi gốc thành 3 biến thể tìm kiếm (xử lý từ đồng nghĩa Tiếng Trung/Hoa, Anh/TOEIC).
    Câu hỏi gốc: "{original_query}"
    CHỈ trả về danh sách câu hỏi, mỗi câu một dòng."""

    try:
        # Gọi API Gemini bất đồng bộ (Lưu ý dùng client.aio)
        response = await genai_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_expansion,
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.3 # Khuyên dùng temperature thấp để nó chỉ trả về đúng danh sách, không nói lảm nhảm
            )
        )
        
        # Lấy text từ response của Gemini chuẩn xác
        content = response.text.strip()
        lines = content.split('\n')
        print(content)
        # Gộp câu hỏi gốc và các biến thể lại
        return [original_query] + [line.strip() for line in lines if line.strip()]
        
    except Exception as e:
        print(f"❌ Lỗi gọi Gemini (Query Expansion): {e}")
        return [original_query]

# --- API CHAT CHÍNH ---
@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not hasattr(app.state, 'retriever'):
        raise HTTPException(status_code=400, detail="Hệ thống chưa có dữ liệu. Vui lòng Rebuild Index.")

    # 1. Tìm kiếm dữ liệu (RAG) - Thêm await
    queries = await expand_queries(req.prompt)
    print(f"🔍 Queries: {queries}")

    all_nodes = []
    for q in queries:
        nodes = app.state.retriever.retrieve(q)
        all_nodes.extend(nodes)

    unique_contents = {}
    for node in all_nodes:
        unique_contents[node.get_content()[:200]] = node.get_content()
    
    context_text = "\n---\n".join(unique_contents.values())[:12000]

    # 2. Xử lý Lịch sử (Load History)
    history = load_history()

    # 3. Chuẩn bị Format cho Gemini
    system_text = f"Bạn là trợ lý tư vấn sinh viên TDC. Dựa vào nội dung này để trả lời: {context_text}. Chỉ trả lời câu hỏi dựa trên tài liệu được cung cấp."
    
    gemini_messages = []
    for msg in history[-10:]:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_messages.append({"role": role, "parts": [{"text": msg["content"]}]})
    
    gemini_messages.append({"role": "user", "parts": [{"text": req.prompt}]})

    history.append({"role": "user", "content": req.prompt})
    
    # 4. Gọi AI Gemini
    try:
        print("🤖 Đang gọi Gemini...")
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=gemini_messages,
            config=types.GenerateContentConfig(
                system_instruction=system_text,
                temperature=0.8,
            ),

        )
        
        answer = response.text 
        
        # 5. Lưu Lịch sử 
        history.append({"role": "assistant", "content": answer})
        save_history(history)

        return {"reply": answer}

    except Exception as e:
        print(f"❌ Lỗi AI: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi AI: {str(e)}")

# --- API ADMIN ---
@app.get("/api/admin/files")
async def list_files():
    return {"files": os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []}

@app.get("/api/filesdata")
async def list_files():
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
    print("🚀 Rebuild Index (Async)...")
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
            raise HTTPException(status_code=500, detail="Nạp RAM thất bại")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
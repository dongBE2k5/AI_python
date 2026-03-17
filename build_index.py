import os
import sys
import io
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import faiss
import torch
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- 1. CẤU HÌNH ĐƯỜNG DẪN ---
DATA_DIR = "data"
STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

# --- 2. CẤU HÌNH EMBEDDING MODEL (LOCAL) ---
# Đã đổi sang MiniLM-L12-v2 (384 chiều) - Nhẹ và hiệu quả cho tiếng Việt/Anh
print("⏳ Đang tải embedding model...")
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cuda" if torch.cuda.is_available() else "cpu", # Dùng GPU nếu có
    normalize=True  # 🔥 quan trọng
)
Settings.embed_model = embed_model

# --- 3. CẤU HÌNH SEMANTIC CHUNKING ---
# Thay vì cắt theo độ dài, chúng ta cắt khi ý nghĩa thay đổi
print("🧠 Đang khởi tạo Semantic Splitter...")
splitter = SemanticSplitterNodeParser(
    buffer_size=2,
    breakpoint_percentile_threshold=90,
    embed_model=embed_model
)

# --- 4. ĐỌC TÀI LIỆU ---
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"⚠️ Thư mục '{DATA_DIR}' trống. Hãy bỏ file PDF/Docx vào đó rồi chạy lại.")
    exit()

print(f"📄 Đang đọc tài liệu từ {DATA_DIR}...")
documents = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()

# --- 5. CHUYỂN ĐỔI SANG NODES (CẮT THEO NGỮ NGHĨA) ---
print("✂️ Đang phân tích và cắt nhỏ tài liệu theo ngữ nghĩa (có thể mất ít phút)...")
nodes = splitter.get_nodes_from_documents(documents)
print(f"✅ Đã tạo {len(nodes)} chunks (nodes) chất lượng.")

# --- 6. CẤU HÌNH KHO VECTOR FAISS ---
# BẮT BUỘC ĐỔI THÀNH 384 ĐỂ KHỚP VỚI MODEL MINILM
dimension = 384 
faiss_index = faiss.IndexHNSWFlat(dimension, 32)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- 7. XÂY DỰNG CHỈ MỤC (INDEXING) ---
print("🚀 Đang xây dựng Vector Index...")
index = VectorStoreIndex(
    nodes, 
    storage_context=storage_context, 
    show_progress=True
)

# --- 8. LƯU TRỮ VĨNH VIỄN ---
print("💾 Đang lưu trữ dữ liệu xuống ổ đĩa...")
# Lưu metadata (các file json)
index.storage_context.persist(persist_dir=STORAGE_DIR)

# Lưu file nhị phân FAISS (Quan trọng để main.py đọc được)
faiss.write_index(faiss_index, os.path.join(STORAGE_DIR, "faiss.index"))

print("\n" + "="*30)
print("✅ THÀNH CÔNG: FAISS Index đã sẵn sàng!")
print(f"📍 Vị trí lưu: {STORAGE_DIR}")
print("="*30)
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import boto3
import os
import mimetypes
import time
import random
import tempfile
import base64
from dotenv import load_dotenv
from typing import List, Optional

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Load biến môi trường
load_dotenv()

print("="*50)
print("Kiểm tra cấu hình:")
print("OPENAI_API_KEY:", "✅ Có" if os.getenv("OPENAI_API_KEY") else "❌ KHÔNG")
print("AWS keys:", "✅ Có" if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY") else "❌ KHÔNG")
print("AWS_BUCKET_NAME:", os.getenv("AWS_BUCKET_NAME") or "❌ KHÔNG")
print("="*50)

# ==============================
# HÀM MÃ HÓA/GIẢI MÃ UTF-8 CHO S3 METADATA
# ==============================
def encode_metadata(value):
    """Mã hóa chuỗi UTF-8 thành ASCII để lưu vào S3 metadata"""
    if value is None:
        return ""
    return base64.b64encode(value.encode('utf-8')).decode('ascii')

def decode_metadata(value):
    """Giải mã metadata từ S3"""
    if not value:
        return ""
    try:
        return base64.b64decode(value).decode('utf-8')
    except:
        return value

# ==============================
# Lifespan event handler
# ==============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load và index PDFs từ S3
    print("🚀 Đang khởi động và load tài liệu...")
    load_and_index_pdfs_from_s3()
    yield
    # Shutdown: cleanup nếu cần
    print("👋 Đang tắt server...")

# ==============================
# FastAPI app
# ==============================
app = FastAPI(
    title="Kho Tài Liệu Sinh Viên API",
    description="Hệ thống tài liệu + hỏi đáp AI cho sinh viên",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==============================
# AWS S3 Config
# ==============================
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION_NAME = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
PREFIX = 'pdfs/'

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=REGION_NAME
)

# ==============================
# Hàm hỗ trợ
# ==============================
def generate_random_file_name(extension):
    millis = int(time.time() * 1000)
    suffix = random.randint(100, 999)
    return f"{millis}{suffix}.{extension}"

def upload_to_s3(file_path: str, metadata: dict = None):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File không tồn tại: {file_path}")

        content_type, _ = mimetypes.guess_type(file_path)
        if content_type is None:
            content_type = 'application/octet-stream'

        extension = os.path.splitext(file_path)[1][1:].lower()
        random_name = generate_random_file_name(extension)
        s3_key = f"{PREFIX}{random_name}"

        print(f"Upload: {file_path} → {s3_key}")
        
        extra_args = {
            "ContentType": content_type,
            "ContentDisposition": "inline"
        }
        
        if metadata:
            extra_args["Metadata"] = metadata

        s3_client.upload_file(
            Filename=file_path,
            Bucket=BUCKET_NAME,
            Key=s3_key,
            ExtraArgs=extra_args
        )

        file_url = f"https://{BUCKET_NAME}.s3.{REGION_NAME}.amazonaws.com/{s3_key}"
        print("URL:", file_url)
        return file_url, s3_key

    except Exception as e:
        print("Lỗi upload S3:", str(e))
        return None, None

# ==============================
# LangChain & Vector Store
# ==============================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

vector_store = None
all_documents_metadata = {}

# Prompt template
prompt_template = """Bạn là trợ lý học tập thông minh dành cho sinh viên Việt Nam tên là Amy.

Dưới đây là các đoạn văn bản được trích xuất từ tài liệu (có thể đến từ nhiều phần khác nhau). 
Hãy tổng hợp thông tin từ tất cả các đoạn để trả lời câu hỏi một cách đầy đủ và chính xác nhất.

**Quan trọng:**
- Nếu thông tin nằm rải rác ở nhiều đoạn, hãy kết hợp chúng lại
- Nếu có nhiều nguồn khác nhau, hãy tổng hợp
- Nếu không có thông tin, hãy nói "Xin lỗi, tôi chưa có thông tin về câu hỏi này trong tài liệu hiện có"
- Trả lời bằng tiếng Việt, thân thiện, dễ hiểu

**Các đoạn văn bản tham khảo:**
{context}

**Câu hỏi:** {question}

**Trả lời (tổng hợp từ các tài liệu):**"""

prompt = ChatPromptTemplate.from_template(prompt_template)

def get_file_metadata_from_s3(key: str):
    try:
        response = s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
        metadata = response.get("Metadata", {})
        decoded_metadata = {}
        for k, v in metadata.items():
            decoded_metadata[k] = decode_metadata(v)
        return decoded_metadata
    except:
        return {}

def load_and_index_pdfs_from_s3():
    """Load và index PDF từ S3 với chunking thông minh"""
    global vector_store, all_documents_metadata
    docs = []
    all_documents_metadata = {}

    try:
        print("🔍 Đang tìm file PDF trong S3...")
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
        
        if "Contents" not in response:
            print("❌ Không tìm thấy file nào trong pdfs/")
            return

        pdf_files = [obj for obj in response["Contents"] if obj["Key"].endswith(".pdf")]
        print(f"📄 Tìm thấy {len(pdf_files)} file PDF")
        
        for idx, obj in enumerate(pdf_files):
            key = obj["Key"]
            file_name = key.split('/')[-1]
            print(f"  [{idx+1}/{len(pdf_files)}] Đang xử lý: {file_name}")

            file_metadata = get_file_metadata_from_s3(key)
            major = file_metadata.get("major", "Chưa phân loại")
            original_name = file_metadata.get("original_name", file_name)
            
            print(f"    📋 Metadata: ngành={major}, tên gốc={original_name}")
            
            all_documents_metadata[key] = {
                "major": major,
                "original_name": original_name,
                "size": obj["Size"],
                "last_modified": obj["LastModified"]
            }

            tmp_path = None
            try:
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp_path = tmp_file.name
                tmp_file.close()
                
                s3_client.download_file(BUCKET_NAME, key, tmp_path)
                
                loader = PyPDFLoader(tmp_path)
                pdf_docs = loader.load()
                print(f"    📖 Đọc được {len(pdf_docs)} trang")
                
                for page_num, doc in enumerate(pdf_docs):
                    doc.metadata["source"] = key
                    doc.metadata["file_name"] = file_name
                    doc.metadata["original_name"] = original_name
                    doc.metadata["major"] = major
                    doc.metadata["page"] = page_num + 1
                    doc.metadata["total_pages"] = len(pdf_docs)
                
                docs.extend(pdf_docs)
                print(f"    ✅ Đã xử lý xong {original_name}")
                
            except Exception as e:
                print(f"    ❌ Lỗi xử lý {file_name}: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

        if not docs:
            print("❌ Không có PDF nào để index")
            return

        print(f"📝 Đang chia nhỏ văn bản với chunking thông minh...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        
        splits = text_splitter.split_documents(docs)
        
        for i, split in enumerate(splits):
            split.metadata["chunk_id"] = i
            split.metadata["total_chunks"] = len(splits)
        
        print(f"🔧 Đang tạo vector store với {len(splits)} chunks...")
        vector_store = FAISS.from_documents(splits, embeddings)
        
        # ========== KHỞI TẠO CORE CONFIG VÀ MODELS ==========
        try:
            from core.config import initialize_config
            from core.models import set_documents
            
            initialize_config(llm, vector_store, embeddings, docs)
            set_documents(docs)
            print("✅ Core config và models đã được khởi tạo")
        except ImportError as e:
            print(f"⚠️ Không thể import core modules: {e}")
        except Exception as e:
            print(f"⚠️ Lỗi khởi tạo core: {e}")
        
        # Lưu documents gốc cho BM25
        global all_documents_raw
        all_documents_raw = docs
        
        # Thống kê
        docs_by_major = {}
        for doc in docs:
            major = doc.metadata.get("major", "Chưa phân loại")
            docs_by_major[major] = docs_by_major.get(major, 0) + 1
        
        print(f"✅ Đã index xong!")
        print(f"   - Tổng số chunks: {len(splits)}")
        print(f"   - Tổng số trang: {len(docs)}")
        print(f"   - Phân bố theo ngành:")
        for major, count in docs_by_major.items():
            print(f"       • {major}: {count} trang")

    except Exception as e:
        print(f"❌ Lỗi load/index từ S3: {str(e)}")
        import traceback
        traceback.print_exc()

all_documents_raw = []

# ==============================
# Models
# ==============================
class Question(BaseModel):
    question: str

# ==============================
# Endpoints
# ==============================
@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Không tìm thấy file index.html</h1>")

@app.get("/documents", response_class=HTMLResponse)
async def documents_page():
    try:
        with open("documents.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Không tìm thấy file documents.html</h1>", status_code=404)

@app.get("/test-s3")
def test_s3():
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
        keys = [obj['Key'] for obj in response.get('Contents', [])]
        return {"status": "OK", "files": keys[:10]}
    except Exception as e:
        return {"status": "ERROR", "detail": str(e)}

@app.get("/status")
def get_status():
    if vector_store is None:
        return {"status": "no_documents", "message": "Chưa có tài liệu"}
    else:
        return {
            "status": "ready",
            "message": "Đã có tài liệu",
            "vector_store_type": str(type(vector_store))
        }

@app.get("/api/documents")
def get_all_documents():
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
        documents = []
        
        if "Contents" in response:
            for obj in response["Contents"]:
                if obj["Key"].endswith(".pdf"):
                    file_name = obj["Key"].split('/')[-1]
                    
                    try:
                        head = s3_client.head_object(Bucket=BUCKET_NAME, Key=obj["Key"])
                        metadata = head.get("Metadata", {})
                        major = decode_metadata(metadata.get("major", ""))
                        original_name = decode_metadata(metadata.get("original_name", ""))
                        
                        if not major:
                            major = "Chưa phân loại"
                        if not original_name:
                            original_name = file_name
                    except:
                        major = "Chưa phân loại"
                        original_name = file_name
                    
                    documents.append({
                        "name": file_name,
                        "original_name": original_name,
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "size_mb": round(obj["Size"] / (1024 * 1024), 2),
                        "last_modified": obj["LastModified"].isoformat(),
                        "last_modified_display": obj["LastModified"].strftime("%d/%m/%Y %H:%M:%S"),
                        "major": major,
                        "url": f"https://{BUCKET_NAME}.s3.{REGION_NAME}.amazonaws.com/{obj['Key']}"
                    })
        
        documents.sort(key=lambda x: x["last_modified"], reverse=True)
        majors = list(set([doc["major"] for doc in documents]))
        
        return {
            "success": True,
            "documents": documents,
            "total": len(documents),
            "total_size_mb": round(sum(d["size"] for d in documents) / (1024 * 1024), 2),
            "majors": majors
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/download/{file_name}")
def download_document(file_name: str):
    try:
        s3_key = f"{PREFIX}{file_name}"
        try:
            s3_client.head_object(Bucket=BUCKET_NAME, Key=s3_key)
        except:
            raise HTTPException(404, detail="Không tìm thấy file")
        
        from fastapi.responses import StreamingResponse
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        
        return StreamingResponse(
            response['Body'].iter_chunks(),
            media_type='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{file_name}"',
                'Content-Type': 'application/pdf'
            }
        )
    except Exception as e:
        raise HTTPException(500, detail=f"Lỗi tải file: {str(e)}")

@app.get("/api/view/{file_name}")
def view_document(file_name: str):
    try:
        s3_key = f"{PREFIX}{file_name}"
        try:
            s3_client.head_object(Bucket=BUCKET_NAME, Key=s3_key)
        except:
            raise HTTPException(404, detail="Không tìm thấy file")
        
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': s3_key},
            ExpiresIn=900
        )
        return {"success": True, "url": url, "file_name": file_name}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/api/documents/{file_name}")
def delete_document_api(file_name: str):
    try:
        s3_key = f"{PREFIX}{file_name}"
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
        load_and_index_pdfs_from_s3()
        return {"success": True, "message": f"Đã xóa {file_name}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/delete-all-documents")
def delete_all_documents():
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
        if "Contents" in response:
            for obj in response["Contents"]:
                if obj["Key"].endswith(".pdf"):
                    s3_client.delete_object(Bucket=BUCKET_NAME, Key=obj["Key"])
        load_and_index_pdfs_from_s3()
        return {"message": "Đã xóa tất cả tài liệu", "success": True}
    except Exception as e:
        raise HTTPException(500, detail=f"Lỗi: {str(e)}")

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    major: str = Form(...)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, detail="Chỉ chấp nhận file .pdf")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        extension = os.path.splitext(file.filename)[1]
        random_name = f"{int(time.time()*1000)}{random.randint(100,999)}{extension}"
        s3_key = f"{PREFIX}{random_name}"
        
        encoded_major = encode_metadata(major)
        encoded_original_name = encode_metadata(file.filename)
        
        print(f"📤 Upload file: {file.filename}")
        print(f"   - Ngành (gốc): {major}")
        print(f"   - Ngành (mã hóa): {encoded_major}")
        
        s3_client.upload_file(
            Filename=tmp_path,
            Bucket=BUCKET_NAME,
            Key=s3_key,
            ExtraArgs={
                "Metadata": {
                    "major": encoded_major,
                    "original_name": encoded_original_name
                },
                "ContentType": "application/pdf"
            }
        )
        
        load_and_index_pdfs_from_s3()
        
        return {
            "message": "Upload thành công",
            "s3_key": s3_key,
            "major": major,
            "original_filename": file.filename
        }

    except Exception as e:
        raise HTTPException(500, detail=f"Lỗi: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

@app.post("/reindex")
def reindex():
    global vector_store
    print("🔄 Bắt đầu re-index...")
    load_and_index_pdfs_from_s3()
    return {"message": "Re-index thành công"}

@app.post("/ask")
def ask_rag(question: Question = Body(...)):
    print(f"📝 Nhận câu hỏi: {question.question}")
    
    if vector_store is None:
        raise HTTPException(503, detail="Chưa có tài liệu nào. Vui lòng upload PDF trước.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 7})
    docs = retriever.invoke(question.question)
    
    print(f"   🔍 Tìm thấy {len(docs)} đoạn văn bản liên quan")
    
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    chain = (
        {"context": lambda x: context, "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    answer = chain.invoke({"question": question.question})
    
    sources = list(set([
        doc.metadata.get("original_name", doc.metadata.get("file_name", "unknown"))
        for doc in docs
    ]))
    
    return {
        "question": question.question,
        "answer": answer,
        "sources": sources,
        "details": {
            "num_chunks": len(docs),
            "majors": list(set([doc.metadata.get("major", "Khác") for doc in docs]))
        }
    }

@app.get("/stats")
def get_stats():
    if vector_store is None:
        return {"status": "no_data"}
    
    major_stats = {}
    if all_documents_metadata:
        for key, meta in all_documents_metadata.items():
            major = meta.get("major", "Khác")
            major_stats[major] = major_stats.get(major, 0) + 1
    
    return {
        "total_documents": len(all_documents_metadata),
        "major_stats": major_stats,
        "vector_store_ready": vector_store is not None
    }

@app.post("/suggest-documents")
def suggest_documents(question: Question = Body(...)):
    print(f"📝 Nhận yêu cầu gợi ý: {question.question}")
    
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
        documents_info = []
        
        if "Contents" in response:
            for obj in response["Contents"]:
                if obj["Key"].endswith(".pdf"):
                    file_name = obj["Key"].split('/')[-1]
                    try:
                        head = s3_client.head_object(Bucket=BUCKET_NAME, Key=obj["Key"])
                        metadata = head.get("Metadata", {})
                        major = decode_metadata(metadata.get("major", ""))
                        original_name = decode_metadata(metadata.get("original_name", ""))
                        
                        if not major:
                            major = "Chưa phân loại"
                        if not original_name:
                            original_name = file_name
                    except:
                        major = "Chưa phân loại"
                        original_name = file_name
                    
                    documents_info.append({
                        "name": file_name,
                        "original_name": original_name,
                        "major": major,
                        "size_mb": round(obj["Size"] / (1024 * 1024), 2),
                        "last_modified": obj["LastModified"].strftime("%d/%m/%Y")
                    })
        
        if not documents_info:
            return {
                "answer": "📭 Chưa có tài liệu nào. Hãy upload tài liệu nhé!",
                "suggestions": []
            }
        
        suggested_docs = []
        if vector_store:
            try:
                retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(question.question)
                
                seen = set()
                for doc in docs:
                    file_name = doc.metadata.get("file_name", "")
                    original_name = doc.metadata.get("original_name", file_name)
                    major = doc.metadata.get("major", "Chưa phân loại")
                    
                    if original_name not in seen:
                        seen.add(original_name)
                        size_mb = 0
                        for d in documents_info:
                            if d["name"] == file_name:
                                size_mb = d["size_mb"]
                                break
                        
                        suggested_docs.append({
                            "name": file_name,
                            "original_name": original_name,
                            "major": major,
                            "size_mb": size_mb
                        })
                suggested_docs = suggested_docs[:5]
            except Exception as e:
                print(f"Lỗi vector search: {e}")
        
        if not suggested_docs:
            suggested_docs = sorted(documents_info, key=lambda x: x["last_modified"], reverse=True)[:5]
        
        doc_list_str = "\n".join([
            f"📄 {doc['original_name']} (Ngành: {doc['major']}, {doc['size_mb']} MB)" 
            for doc in suggested_docs
        ])
        
        suggestion_prompt = f"""Bạn là trợ lý học tập Amy. Dựa vào danh sách tài liệu dưới đây, hãy gợi ý tài liệu phù hợp với câu hỏi.

Danh sách tài liệu gợi ý:
{doc_list_str}

Câu hỏi: "{question.question}"

Hãy trả lời ngắn gọn, thân thiện, giới thiệu từng tài liệu và giải thích tại sao phù hợp."""

        response_ai = llm.invoke(suggestion_prompt)
        
        return {
            "answer": response_ai.content,
            "suggestions": suggested_docs
        }
        
    except Exception as e:
        print(f"Lỗi gợi ý: {e}")
        return {
            "answer": "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại sau.",
            "suggestions": []
        }

@app.post("/summarize/{file_name}")
def summarize_document(file_name: str):
    try:
        s3_key = f"{PREFIX}{file_name}"
        
        try:
            s3_client.head_object(Bucket=BUCKET_NAME, Key=s3_key)
        except:
            raise HTTPException(404, detail="Không tìm thấy file")
        
        tmp_path = None
        try:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp_path = tmp_file.name
            tmp_file.close()
            
            s3_client.download_file(BUCKET_NAME, s3_key, tmp_path)
            
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            full_text = "\n".join([p.page_content for p in pages[:10]])
            
            summary_prompt = f"""Tóm tắt nội dung chính của tài liệu sau thành 5-7 ý ngắn gọn:

{full_text[:3000]}

Tóm tắt:"""
            
            response = llm.invoke(summary_prompt)
            
            return {
                "success": True,
                "file_name": file_name,
                "summary": response.content,
                "total_pages": len(pages),
                "preview_pages": min(10, len(pages))
            }
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 Khởi động server tại: http://127.0.0.1:8000")
    print("📄 Truy cập giao diện: http://127.0.0.1:8000")
    print("📚 Truy cập quản lý tài liệu: http://127.0.0.1:8000/documents")
    print("📊 Xem thống kê: http://127.0.0.1:8000/stats")
    print("="*50 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
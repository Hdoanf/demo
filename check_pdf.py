import boto3
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv(r"D:\web tài liệu sinh viên\.env")

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
PREFIX = 'pdfs/'

# Liệt kê file
response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
print("Files in S3:")
for obj in response.get('Contents', []):
    print(f"  - {obj['Key']}")

# Thử đọc file mới nhất
for obj in response.get('Contents', []):
    key = obj['Key']
    if key.endswith('.pdf'):
        print(f"\nĐang thử đọc file: {key}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            s3_client.download_file(BUCKET_NAME, key, tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            print(f"  ✅ Đọc được {len(pages)} trang")
            print(f"  Nội dung trang đầu: {pages[0].page_content[:200]}...")
        except Exception as e:
            print(f"  ❌ Lỗi đọc PDF: {e}")
        finally:
            os.unlink(tmp_path)
        break  # chỉ test file đầu tiên
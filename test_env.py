from dotenv import load_dotenv
import os

load_dotenv(r"D:\Web tài liệu sinh viên\.env")

print("OPENAI_API_KEY:", "✅ Có" if os.getenv("OPENAI_API_KEY") else "❌ Không")
print("AWS_ACCESS_KEY_ID:", "✅ Có" if os.getenv("AWS_ACCESS_KEY_ID") else "❌ Không")
print("AWS_SECRET_ACCESS_KEY:", "✅ Có" if os.getenv("AWS_SECRET_ACCESS_KEY") else "❌ Không")
print("AWS_BUCKET_NAME:", os.getenv("AWS_BUCKET_NAME") or "❌ Không")
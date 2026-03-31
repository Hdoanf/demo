from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv(dotenv_path=r"D:\Web tài liệu sinh viên\backend\.env")
print("OPENAI_API_KEY từ env:", os.getenv("OPENAI_API_KEY")[:10] + "..." if os.getenv("OPENAI_API_KEY") else "KHÔNG CÓ KEY")

try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke("Xin chào, bạn là ai?")
    print("Kết quả từ OpenAI:", response.content)
except Exception as e:
    print("LỖI KHI GỌI OPENAI:", str(e))
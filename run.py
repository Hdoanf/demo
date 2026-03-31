import uvicorn

if __name__ == "__main__":
    print("🚀 Đang khởi động server...")
    print("📱 Truy cập: http://localhost:8000")
    print("="*50)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
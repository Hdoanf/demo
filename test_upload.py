import boto3
import mimetypes
import os
from dotenv import load_dotenv
import time
import random

# Load env

load_dotenv(dotenv_path=r"D:\Web tài liệu sinh viên\backend\.env")
# Lấy config từ .env
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION_NAME = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

PREFIX = 'pdfs/'


def generate_random_file_name(extension):
    millis = int(time.time() * 1000)
    suffix = random.randint(100, 999)
    return f"{millis}{suffix}.{extension}"


def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=REGION_NAME
    )


def upload_to_s3(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"File không tồn tại: {file_path}")
            return None

        # Detect content type
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type is None:
            content_type = 'application/octet-stream'

        # Extension
        extension = os.path.splitext(file_path)[1][1:].lower()
        print(f"Extension: {extension}")

        # Generate key
        random_name = generate_random_file_name(extension)
        s3_key = f"{PREFIX}{random_name}"

        s3_client = get_s3_client()

        print(f"Đang upload: {file_path} → {s3_key}")

        s3_client.upload_file(
            Filename=file_path,
            Bucket=BUCKET_NAME,
            Key=s3_key,
            ExtraArgs={
                "ContentType": content_type,
                "ContentDisposition": "inline"
            }
        )

        file_url = f"https://{BUCKET_NAME}.s3.{REGION_NAME}.amazonaws.com/{s3_key}"

        print("Upload thành công!")
        print(file_url)

        return file_url

    except Exception as e:
        print("Lỗi upload:", str(e))
        return None


def delete_from_s3(file_url_or_key):
    s3_client = get_s3_client()

    if file_url_or_key.startswith("http"):
        s3_key = file_url_or_key.split(f"{BUCKET_NAME}.s3.{REGION_NAME}.amazonaws.com/")[-1]
    else:
        s3_key = file_url_or_key

    try:
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
        print("Đã xóa:", s3_key)
        return True
    except Exception as e:
        print("Lỗi xóa:", str(e))
        return False


# ================= TEST =================
if __name__ == "__main__":
    # Debug (quan trọng)
    print("KEY:", AWS_ACCESS_KEY_ID)
    print("REGION:", REGION_NAME)
    print("BUCKET:", BUCKET_NAME)

    file_path = r"D:\Web tài liệu sinh viên\backend\AI sẽ làm cho Hello Job.pdf"

    url = upload_to_s3(file_path)

    if url:
        print("URL:", url)
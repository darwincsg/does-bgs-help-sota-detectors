import boto3
from pathlib import Path

def upload_folder_to_s3(local_folder: Path, bucket_name: str, s3_prefix: str):
    s3 = boto3.client('s3')
    for file_path in local_folder.glob("*"):
        if file_path.is_file():
            s3_key = f"{s3_prefix}/{file_path.name}"
            print(f"Subiendo {file_path} a s3://{bucket_name}/{s3_key} ...")
            s3.upload_file(str(file_path), bucket_name, s3_key)
    print("All files uploaded.")


def main():
    local_folder = Path("./zips")
    bucket_name = "darvideosbucket"  # Bucket name
    s3_prefix = "text_annotations/RCNN"

    upload_folder_to_s3(local_folder, bucket_name, s3_prefix)

if __name__ == "__main__":
    main()

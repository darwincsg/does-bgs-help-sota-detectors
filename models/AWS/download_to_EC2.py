import boto3
import os

bucket_name = "darvideosbucket"  # Bucket name
s3_prefix = "text_annotations/RCNN"
dest_folder = '/home/Raw/'  

os.makedirs(dest_folder, exist_ok=True)

s3 = boto3.client('s3')

paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

for page in pages:
    for obj in page.get('Contents', []):
        key = obj['Key']
        filename = os.path.basename(key)
        if not filename:  
            continue

        local_path = os.path.join(dest_folder, filename)
        print(f"Downloading {key}...")
        s3.download_file(bucket_name, key, local_path)

print("\n Download completed.")

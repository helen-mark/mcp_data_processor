#!/usr/bin/env python3
# /home/user/download_from_storage.py

import boto3
from botocore.client import Config
import os
from datetime import datetime

def download_files():
    BUCKET_NAME = "mcp-data"
    LOCAL_PATH = "data"
    ENDPOINT_URL = "https://storage.yandexcloud.net"

    session = boto3.session.Session()
    s3 = session.client(
        service_name='s3',
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4')
    )
    
    os.makedirs(LOCAL_PATH, exist_ok=True)
    
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_NAME)
    
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            local_file = os.path.join(LOCAL_PATH, key)
            
            s3.download_file(BUCKET_NAME, key, local_file)
            print(f"Downloaded: {key}")
    

if __name__ == "__main__":
    download_files()

import json
import os
import pandas as pd
import yaml

from get_files_from_storage import download_files
from json2csv import json_to_csv
from process_csv_data import launch_llm_processing 
from send_between_VM import transfer_all

with open('config.yml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
raw_data_path = config.get('folders').get('raw_data')
csv_mail_path = config.get('folders').get('csv_mail')
csv_calls_path = config.get('folders').get('csv_calls')
mail_filename = config.get('folders').get('mail_filename')
calls_filename = config.get('folders').get('calls_filename')

def merge_files():
    print(raw_data_path)
    for filename in os.listdir(raw_data_path):
        if 'mail' in filename:
            mail = json_to_csv(os.path.join(raw_data_path, filename))
            mail.rename(columns={'body': 'text'}, inplace=True)
        elif 'call' in filename:
            calls = json_to_csv(os.path.join(raw_data_path, filename))

    mail_tagged = pd.read_csv(os.path.join(csv_mail_path, mail_filename))

    cols_to_add = set(mail_tagged.columns) - set(mail.columns)
    mail = mail[~mail['text'].isin(mail_tagged['text'])].copy()
    
    for col in cols_to_add:
        mail[col] = None
        calls[col] = None

    mail = pd.concat([mail, mail_tagged], ignore_index=True)

    calls_tagged = pd.read_csv(os.path.join(csv_calls_path, calls_filename))
    calls = calls[~calls['text'].isin(calls_tagged['text'])].copy()
    calls = pd.concat([calls, calls_tagged], ignore_index=True)

    mail.to_csv(os.path.join(csv_mail_path, mail_filename))
    calls.to_csv(os.path.join(csv_calls_path, calls_filename))
    

print('Downloading...')
download_files()
print('Merging...')
merge_files()
print('Start tagger...')
launch_llm_processing()
print('Sending files...')
transfer_all()
print('Done')


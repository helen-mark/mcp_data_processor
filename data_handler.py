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
    new_mail_ids = set()
    new_call_ids = set()

    for filename in os.listdir(raw_data_path):
        if 'mail' in filename:
            mail = json_to_csv(os.path.join(raw_data_path, filename))
            mail.rename(columns={'body': 'text'}, inplace=True)
            new_mail_ids.update(mail['message_id'].tolist())
        elif 'call' in filename:
            calls = json_to_csv(os.path.join(raw_data_path, filename))
            new_call_ids.update(calls['call_id'].tolist() if 'call_id' in calls.columns else calls.index.tolist())

    mail_tagged = pd.read_csv(os.path.join(csv_mail_path, mail_filename))
    mail_tagged = mail_tagged[[c for c in mail_tagged.columns if 'Unnam' not in c]]

    cols_to_add = set(mail_tagged.columns) - set(mail.columns)
    print(cols_to_add)

    print('MAIL TAGGED LEN', len(mail_tagged))
    read_status_map = dict(zip(mail['message_id'], mail['is_read']))

    mail_tagged['is_read'] = mail_tagged.apply(
        lambda x: read_status_map.get(x['message_id'], x['is_read']),
        axis=1
    )

    new_mail_mask = ~mail['message_id'].isin(mail_tagged['message_id'])
    new_mail_ids = set(mail[new_mail_mask]['message_id'].tolist())

    mail = mail[~mail['text'].isin(mail_tagged['text'])].copy()
    for col in cols_to_add:
        mail[col] = None
        calls[col] = None

    mail = pd.concat([mail, mail_tagged], ignore_index=True)
    mail = mail.drop_duplicates(subset=['text'], keep='first')
    print("CONCAT LEN", len(mail))

    calls_tagged = pd.read_csv(os.path.join(csv_calls_path, calls_filename))

    if 'call_id' in calls.columns and 'call_id' in calls_tagged.columns:
        new_call_mask = ~calls['call_id'].isin(calls_tagged['call_id'])
        new_call_ids = set(calls[new_call_mask]['call_id'].tolist())
    else:
        new_call_mask = ~calls['text'].isin(calls_tagged['text'])
        new_call_ids = set(calls[new_call_mask].index.tolist())

    calls = calls[~calls['text'].isin(calls_tagged['text'])].copy()
    calls = pd.concat([calls, calls_tagged], ignore_index=True)
    calls['date_str'] = calls['date'].astype(str)

    # Сохраняем ID новых записей во временный файл
    new_records_info = {
        'new_mail_ids': list(new_mail_ids),
        'new_call_ids': list(new_call_ids)
    }
    with open('new_records_info.json', 'w', encoding='utf-8') as f:
        json.dump(new_records_info, f, ensure_ascii=False)

    mail.to_csv(os.path.join(csv_mail_path, mail_filename), index=False)
    calls.to_csv(os.path.join(csv_calls_path, calls_filename), index=False)


print('Downloading...')
download_files()
print('Merging...')
merge_files()
print('Start tagger...')
launch_llm_processing()
print('Sending files...')
transfer_all()
print('Done')
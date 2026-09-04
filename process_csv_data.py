import ast
import json
import os
import pandas as pd
import re
import yaml

from csv_tagger import CsvProcessor, DataType


def process_tags_and_summary(config, data_type, new_ids=None):
    output_csv_path = os.path.join(config["folders"]["csv_" + data_type],
                                   config['folders'][data_type + '_filename'])
    predicted_path = os.path.join(config["folders"]["ai_rct"],
                                  config["folders"]["predictions_filename"])
    INN_path = os.path.join(config["folders"]["ai_rct"],
                            config["folders"]["INN_filename"])

    predicted_list = pd.read_excel(predicted_path)
    predicted_list = predicted_list[predicted_list["Risk"] >= 1]["INN"]

    INNs = pd.read_excel(INN_path)
    predicted_INNs = INNs[INNs["ИНН"].isin(predicted_list)]

    def match_rct(sender: str) -> bool:
        def extract_address(text: str) -> str:
            return re.search(r'<([^>]+)>', text).group(1) if re.search(r'<([^>]+)>', text) else ""

        if extract_address(sender) in predicted_INNs["EMAIL"].str.lower().values:
            print("Found sender! ", sender)
            return True
        for n, row in predicted_INNs.iterrows():
            if str(row['Контрагент']).lower() in sender.lower():
                print("Found contragent! ", sender)
                return True
        return False

    def add_ai_rct_tag(row):
        tags_list = ast.literal_eval(row['tags'])

        if 'mail' in tags_list and 'ai rct' not in tags_list:
            sender = row['from']
            if match_rct(sender):
                tags_list.append('ai rct')
        return str(tags_list)

    def add_outrage_tag(row):
        tags_list = ast.literal_eval(row['tags'])
        summary = row['summary']
        outrage_words = ['негодует', 'негодование', 'возмущ', 'ужас']
        if any(word in summary for word in outrage_words):
            if 'клиент возмущен' not in tags_list:
                tags_list.append('клиент возмущен')
        return str(tags_list)

    df = pd.read_csv(output_csv_path).dropna(subset=['tags', 'summary'])

    if new_ids and len(new_ids) > 0:
        if data_type == 'mail' and 'message_id' in df.columns:
            mask = df['message_id'].isin(new_ids)
        elif data_type == 'calls' and 'call_id' in df.columns:
            mask = df['call_id'].isin(new_ids)
        else:
            mask = pd.Series([True] * len(df), index=df.index)

        print(f"Обрабатываем {mask.sum()} новых записей из {len(df)}")

        df.loc[mask, 'tags'] = df.loc[mask].apply(add_ai_rct_tag, axis=1)
        df.loc[mask, 'tags'] = df.loc[mask].apply(add_outrage_tag, axis=1)
    else:
        df['tags'] = df.apply(add_ai_rct_tag, axis=1)
        df['tags'] = df.apply(add_outrage_tag, axis=1)

    df.to_csv(output_csv_path, index=False)
    return


def launch_llm_processing():
    print("CSV PROCESSOR - LAUNCH")
    print("=" * 50)
    with open('config.yml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    new_records_info = {'new_mail_ids': [], 'new_call_ids': []}
    try:
        with open('new_records_info.json', 'r', encoding='utf-8') as f:
            new_records_info = json.load(f)
    except FileNotFoundError:
        print("Файл new_records_info.json не найден, обрабатываем все записи")

    for data_type in [DataType.MAIL, DataType.CALLS]:
        print("Processing " + data_type.value + " file...")
        processor = CsvProcessor(
            model=config["llm_model"],
            output_csv_path=os.path.join(config["folders"]["csv_" + data_type.value],
                                         config["folders"][data_type.value + "_filename"]),
            batch_size=50,
            data_type=data_type,
            config_path='config.yml'
        )

        print("\nStarting CSV tagging process...")
        processor.process(add_tags=False)

        # Передаем ID новых записей для оптимизации
        if data_type == DataType.MAIL:
            new_ids = new_records_info.get('new_mail_ids', [])
        else:
            new_ids = new_records_info.get('new_call_ids', [])

        process_tags_and_summary(config, data_type.value, new_ids)

    print(f"\n Processing finished!")
    print(f" Results saved")

    try:
        os.remove('new_records_info.json')
    except:
        pass


if __name__ == "__main__":
    launch_llm_processing()
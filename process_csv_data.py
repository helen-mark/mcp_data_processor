import ast
import os
import pandas as pd
import yaml

from csv_tagger import CsvProcessor


def process_tags_and_summary(config):
    output_csv_path=os.path.join(config["folders"]["csv_mail"], config['folders']['mail_filename'])

    def add_missing_tag(tags_str):
        if tags_str is None:
            return None
        tags_list = ast.literal_eval(tags_str)
        if 'mail' not in tags_list and 'call' not in tags_list:
            tags_list.append('mail')
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
    df['tags'] = df['tags'].apply(add_missing_tag)
    df['tags'] = df.apply(add_outrage_tag, axis=1)
    df.to_csv(output_csv_path)
    return

def launch_llm_processing():
    print("CSV PROCESSOR - LAUNCH")
    print("=" * 50)
    with open('config.yml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    processor = CsvProcessor(
        model=config["llm_model"],
        output_csv_path=os.path.join(config["folders"]["csv_mail"], config['folders']['mail_filename']),
        batch_size=50,
        mail=True,
        config_path='config.yml'
    )

    print("\nStarting CSV tagging process...")
    processor.process()
    process_tags_and_summary(config)

    print(f"\n Processing finished!")
    print(f" Results saved")


if __name__ == "__main__":
    launch_llm_processing()

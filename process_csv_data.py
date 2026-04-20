import ast
import os
import pandas as pd
import re
import yaml

from csv_tagger import CsvProcessor


def process_tags_and_summary(config):
    output_csv_path=os.path.join(config["folders"]["csv_mail"], config['folders']['mail_filename'])
    predicted_path = os.path.join(config["folders"]["ai_rct"], config["folders"]["predictions_filename"])
    INN_path = os.path.join(config["folders"]["ai_rct"], config["folders"]["INN_filename"])
    
    predicted_list = pd.read_excel(predicted_path)
    predicted_list = predicted_list[predicted_list["Risk"]==1]["INN"]
    
    INNs = pd.read_excel(INN_path)
    predicted_INNs = INNs[INNs["ИНН"].isin(predicted_list)]
    print("predicted_INNS:", predicted_INNs)
    
    def match_rct(sender: str) -> bool:
        def extract_address(text: str) -> str:
            #print(re.search(r'<([^>]+)>', text).group(1))
            return re.search(r'<([^>]+)>', text).group(1) if re.search(r'<([^>]+)>', text) else ""
        #print(f"Emails list: {predicted_INNs['EMAIL']}")

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
    df['tags'] = df.apply(add_ai_rct_tag, axis=1)
    df['tags'] = df.apply(add_outrage_tag, axis=1)
    print(df.head(5))
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

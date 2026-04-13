import pandas as pd
import json
import ast


def json_to_csv(json_file, csv_file=None):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for record in data:
        for key, value in record.items():
            if isinstance(value, str):
                if value.startswith("'") and value.endswith("'"):
                    record[key] = value[1:-1]
                elif value.startswith('"') and value.endswith('"'):
                    record[key] = value[1:-1]

    df = pd.DataFrame(data)
    if csv_file:
        df.to_csv(
            csv_file,
            index=False,
            encoding='utf-8-sig',
            quotechar='"',
            quoting=1
        )

    print(f"Converted {len(df)} rows")

    return df



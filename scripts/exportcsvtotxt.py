import os
import pandas as pd

def export_csv_to_txt():
    csv_file_path = '/home/theo/work/graphrag/sap/input/123.csv'
    output_dir = '/home/theo/work/graphrag/sap/input'

    os.makedirs(output_dir, exist_ok=True)

    # Read CSV with semicolon delimiter
    df = pd.read_csv(csv_file_path, sep=";", on_bad_lines='skip')

    records_per_file = 10
    total_records = len(df)

    for start in range(0, total_records, records_per_file):
        end = start + records_per_file
        chunk = df.iloc[start:end]

        txt_filename = f'records_{start + 1}_to_{min(end, total_records)}.txt'
        txt_file_path = os.path.join(output_dir, txt_filename)

        with open(txt_file_path, 'w', encoding='utf-8') as txtfile:
            for i, (index, row) in enumerate(chunk.iterrows(), start=start + 1):
                txtfile.write(f'--- Record {i} ---\n')
                for col, val in row.items():
                    txtfile.write(f'{col}: {val}\n')
                txtfile.write('\n')

        print(f'Created: {txt_filename}')

    print(f'Export completed. Files saved to: {output_dir}')

if __name__ == '__main__':
    export_csv_to_txt()

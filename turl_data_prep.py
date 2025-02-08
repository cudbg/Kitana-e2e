import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import bisect
from collections import defaultdict

def main(data_dir = "data/turl/"):
    # get all the csvs in the directory
    csv_dir = os.path.join(data_dir, "csv")
    csvs = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    # iterate through all the csvs, if the column has "Country" as a header, then keep it and copy to os.path.join(data_dir, seller)
    for csv in tqdm(csvs):
        csv_path = os.path.join(csv_dir, csv)
        if os.stat(csv_path).st_size == 0:
            print(f"Skipping empty file: {csv}")
            continue

        try:
            df = pd.read_csv(
                csv_path,
                engine="python",
                on_bad_lines='skip'
            )
        except pd.errors.EmptyDataError:
            print(f"Skipping file with no columns: {csv}")
            continue

        if "Country" in df.columns:
            df.to_csv(os.path.join(data_dir, "seller", csv), index=False)

def construct_table_json_mapping(
    table_dir="data/turl/seller/",
    json_dir="data/WebTable/webtables-evaluation-data/json/WikipediaGS_json/entities_instance/",
    output_file="data/WebTable/table_join_mapping.csv",
):
    # -------------------------------
    # 1. 构建 CSV 映射（hash map）
    # -------------------------------
    print("Building CSV hash map...")
    csv_files = [f for f in os.listdir(table_dir) if f.endswith('.csv')]
    csv_map = defaultdict(list)
    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        key = csv_file.split("#")[0]
        csv_map[key].append(csv_file)
    print(f"CSV hash map built with {len(csv_map)} keys.")

    # -------------------------------
    # 2. 打开输出 CSV 文件（流式写入）
    # -------------------------------
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_f = open(output_file, "w", newline="", encoding="utf-8-sig")
    writer = None  # 后续第一次写入时创建 DictWriter

    # -------------------------------
    # 3. 处理 JSON 文件并做 join
    # -------------------------------
    for filename in tqdm(os.listdir(json_dir), desc="Processing JSON files"):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(json_dir, filename)
        if os.stat(filepath).st_size == 0:
            print(f"Skipping empty json file: {filename}")
            continue

        # 尝试一次性加载 JSON 文件，否则逐行加载
        with open(filepath, "r", encoding="utf-8-sig") as jf:
            try:
                obj = json.load(jf)
                records = obj if isinstance(obj, list) else [obj]
            except Exception as e:
                print(f"Error decoding {filename} with full load: {e}. Fallback to line-by-line parsing.")
                jf.seek(0)
                records = []
                for line in jf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except Exception as e2:
                        print(f"Error decoding line in {filename}: {e2}")
                        continue

        # 处理每个 JSON 记录
        for record in records:
            # 根据 title 生成 join_key（逻辑与原来一致）
            title = record.get("title", "")
            join_key = ""
            if isinstance(title, str):
                join_key = title.split(" - ")[0].strip().replace(" ", "_")
            
            # 使用 hash 映射寻找所有以 join_key 开头的 CSV 文件
            matched_csvs = []
            for key, csv_list in csv_map.items():
                if key.startswith(join_key):
                    matched_csvs.extend(csv_list)
            
            if matched_csvs:
                # 对每个匹配的 csv_file 写出一条记录
                for csv_file in matched_csvs:
                    record_out = record.copy()
                    record_out["csv_file"] = csv_file
                    # 第一次写入时创建 DictWriter（假设各记录结构一致）
                    if writer is None:
                        header = list(record_out.keys())
                        writer = csv.DictWriter(out_f, fieldnames=header)
                        writer.writeheader()
                    writer.writerow(record_out)
            else:
                record_out = record.copy()
                record_out["csv_file"] = None
                if writer is None:
                    header = list(record_out.keys())
                    writer = csv.DictWriter(out_f, fieldnames=header)
                    writer.writeheader()
                writer.writerow(record_out)
    
    out_f.close()
    print(f"Finished writing joined mapping to {output_file}")


if __name__ == "__main__":
    construct_table_json_mapping()
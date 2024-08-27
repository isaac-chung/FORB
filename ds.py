from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
import json
import pandas as pd
import glob
import os

output_path = "saved_images"

for p in glob.glob("metadata/*.ndjson"):
    metadata_file_name = os.path.basename(p)
    split_pos = metadata_file_name.rfind('_')
    object_type = metadata_file_name[:split_pos]
    image_type = metadata_file_name[split_pos + 1:].split('.')[0]

    folder = os.path.join(output_path, object_type, image_type)
    # folder= "saved_images/animated_cards/database"
    if image_type == "gt":
        ds_type = "qrels"
    else:
        ds_type = "query" if image_type == "query" else "corpus"
    pq_fname = f"{ds_type}-{object_type}-00000-of-00001.parquet"
    if os.path.exists(pq_fname):
        continue

    json_data = []
    with open(p) as f:
        for i,line in enumerate(f):
            line_data = json.loads(line)

            if image_type == "gt":
                query_img_path = os.path.join(output_path, line_data['query_image'])
                gt_img_path = os.path.join(output_path, line_data['groundtruth_images'][0])
                if (not os.path.exists(query_img_path)) or (not os.path.exists(gt_img_path)) :
                    continue
                qid = line_data['query_id']
                cid = line_data['groundtruth_ids'][0]
                d = line_data['difficulty']

                line_data = {}
                line_data['query-id'] = f"query-{object_type}-{qid}"
                line_data['corpus-id'] = cid
                line_data['score'] = 1
                line_data['difficulty'] = d
            else:
                img_path = os.path.join(output_path, line_data['image'])
                if not os.path.exists(img_path):
                    continue
                line_data['id'] = f"{ds_type}-{object_type}-{i+1}"
                line_data['modality'] = "image"
            json_data.append(line_data)

    df = pd.DataFrame(json_data)
    ### load dataset
    # queries_ = {"id": [], "modality": [], "image": []}
    # corpus_ = {"id": [], "modality": [], "image": []}
    # relevant_docs_ = {"query-id": [], "corpus-id": [], "score": [], "difficulty": []}

    if image_type == "gt":
        try:
            ds = Dataset.from_pandas(df, split="train")
            ds.to_parquet(pq_fname)
        except Exception as e:
            print(pq_fname, e)
    else:
        try:
            ds = load_dataset("imagefolder", data_dir=folder, split="train")
            ds = ds.add_column("modality", df['modality'].to_list())
            ds = ds.add_column("id", df['id'].to_list())
            ds = ds.add_column("image_url", df['image_url'].to_list())
            ds = ds.add_column("path", df['image'].to_list())

            ds.to_parquet(pq_fname)
        except Exception as e:
            print(pq_fname, e)



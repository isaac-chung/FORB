from datasets import Dataset, DatasetDict, load_dataset

import glob

corpus_files = []
queries_files = []
qrel_files = []

for fname in glob.glob("*parquet"):
    if "corpus" in fname:
        corpus_files.append(fname)
    elif "qrels" in fname:
        qrel_files.append(fname)
    else:
        queries_files.append(fname)


corpus = load_dataset('parquet', data_files={'test': corpus_files})
print(corpus)
corpus['test'].to_parquet("hf_files/corpus-00000-of-00001.parquet")

queries = load_dataset('parquet', data_files={'test': queries_files})
print(queries)

qrel = load_dataset('parquet', data_files={'test': qrel_files})
print(qrel)
qrel['test'].to_parquet("hf_files/qrels-00000-of-00001.parquet")

## trim queries based on qrel
qdf = queries['test'].to_pandas()
valid_qid = qrel['test']['query-id']

qdf = qdf[qdf['id'].isin(valid_qid)]
queries = Dataset.from_pandas(qdf, split="test").remove_columns("__index_level_0__")
print(queries)
queries.to_parquet("hf_files/queries-00000-of-00001.parquet")

"""Turn OpenWebText parquet shards into train/val text files plus a character vocab.

The video extracts .xz archives with WinRAR; the Hugging Face copy of OpenWebText
comes as parquet, so we read those directly instead.
"""
import glob
import os

import pyarrow.parquet as pq

raw_files = sorted(glob.glob(os.path.join('data', 'raw', '*.parquet')))
output_train = os.path.join('data', 'output_train.txt')
output_val = os.path.join('data', 'output_val.txt')
vocab_file = 'vocab.txt'

vocab = set()
docs_written = {'train': 0, 'val': 0}

with open(output_train, 'w', encoding='utf-8') as train_out, \
        open(output_val, 'w', encoding='utf-8') as val_out:
    for path in raw_files:
        print('reading', path)
        table = pq.read_table(path, columns=['text'])
        for i, doc in enumerate(table.column('text').to_pylist()):
            doc = doc.replace('\r', '')
            # every 10th document goes to validation (90/10 split)
            split, out = ('val', val_out) if i % 10 == 0 else ('train', train_out)
            out.write(doc + '\n\n')
            vocab.update(doc)
            docs_written[split] += 1

vocab.add('\n')
with open(vocab_file, 'w', encoding='utf-8') as f:
    f.write(''.join(sorted(vocab)))

print(docs_written)
print(len(vocab), 'unique characters')

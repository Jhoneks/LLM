"""Download OpenWebText parquet shards and append them to train/val text files.

The video extracts .xz archives with WinRAR; the Hugging Face copy of OpenWebText
comes as parquet, so we read those directly instead. Shards are handled one at a
time (download, extract, delete) so the parquet files never pile up on disk.

Usage: python data-extract.py [first_shard] [last_shard]   (default 0 79)
"""
import os
import sys
import time
import urllib.request

import pyarrow.parquet as pq

SHARD_URL = ('https://huggingface.co/datasets/Skylion007/openwebtext/resolve/'
             'refs%2Fconvert%2Fparquet/plain_text/train/{:04d}.parquet')
NUM_SHARDS = 80

raw_dir = os.path.join('data', 'raw')
output_train = os.path.join('data', 'output_train.txt')
output_val = os.path.join('data', 'output_val.txt')
done_file = os.path.join('data', 'extracted.txt')
vocab_file = 'vocab.txt'

first = int(sys.argv[1]) if len(sys.argv) > 1 else 0
last = int(sys.argv[2]) if len(sys.argv) > 2 else NUM_SHARDS - 1

os.makedirs(raw_dir, exist_ok=True)

done = set()
if os.path.exists(done_file):
    with open(done_file) as f:
        done = {int(line) for line in f if line.strip()}

# Only build the vocab on the first run. A trained model depends on it, so later
# runs leave it alone; characters outside it are skipped by training.py.
build_vocab = not os.path.exists(vocab_file)
vocab = set()


def download(shard, path):
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(SHARD_URL.format(shard), path)
            return
        except Exception as e:
            print(f'  download failed ({e}), retrying...', flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f'could not download shard {shard}')


for shard in range(first, last + 1):
    if shard in done:
        continue
    path = os.path.join(raw_dir, f'{shard:04d}.parquet')
    print(f'shard {shard}: downloading', flush=True)
    download(shard, path)

    print(f'shard {shard}: extracting', flush=True)
    table = pq.read_table(path, columns=['text'])
    with open(output_train, 'a', encoding='utf-8') as train_out, \
            open(output_val, 'a', encoding='utf-8') as val_out:
        for i, doc in enumerate(table.column('text').to_pylist()):
            doc = doc.replace('\r', '')
            # every 10th document goes to validation (90/10 split)
            out = val_out if i % 10 == 0 else train_out
            out.write(doc + '\n\n')
            if build_vocab:
                vocab.update(doc)
    del table
    os.remove(path)

    with open(done_file, 'a') as f:
        f.write(f'{shard}\n')

if build_vocab:
    vocab.add('\n')
    with open(vocab_file, 'w', encoding='utf-8') as f:
        f.write(''.join(sorted(vocab)))
    print(len(vocab), 'unique characters')

print('done')

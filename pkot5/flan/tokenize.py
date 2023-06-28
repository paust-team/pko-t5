import json
import pickle
from pathlib import Path

import fire
from tqdm import tqdm
from transformers import T5TokenizerFast
import aimrocks


def main(output_file: str, data_dir: str, pretrained_model_name_or_path: str = 'paust/pko-t5-base'):
    input_texts, output_texts = [], []
    for jsonl_file in Path(data_dir).iterdir():
        if not jsonl_file.name.endswith('.jsonl'):
            continue

        with open(jsonl_file, 'rt') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            record = json.loads(line.strip())
            input_texts.append(record['input_text'])
            output_texts.append(record['output_text'])
    print(f"### Succeeded reading data length={len(input_texts)}")
    assert len(input_texts) == len(output_texts)

    db = aimrocks.DB(output_file, aimrocks.Options(create_if_missing=True, paranoid_checks=False), read_only=False)
    tokenizer = T5TokenizerFast.from_pretrained(pretrained_model_name_or_path)
    max_length = 0
    for i in tqdm(range(0, len(input_texts), 4096)):
        input_ids = tokenizer(input_texts[i:i + 4096], add_special_tokens=True).input_ids
        target_ids = tokenizer(output_texts[i:i + 4096], add_special_tokens=True).input_ids

        batch = aimrocks.WriteBatch()
        for j in range(len(input_ids)):
            max_length = max(len(input_ids[j]), max_length)
            batch.put(f"{i+j}".encode('utf-8'), pickle.dumps({'input_ids': input_ids[j], 'target_ids': target_ids[j]}))
        db.write(batch)
    db.put(b'metadata', pickle.dumps({'max_length': max_length, 'total': len(input_texts)}))
    db.close()

    print(f"### max_length={max_length}")


def check(db_file):
    db = aimrocks.DB(db_file, aimrocks.Options(), read_only=True)
    meta = pickle.loads(db.get(b'metadata'))
    target_max_length = 0
    for i in tqdm(range(meta['total'])):
        record = pickle.loads(db.get(f"{i}".encode('utf-8')))
        target_max_length = max(len(record['target_ids']), target_max_length)
    print(f"input max length: {meta['max_length']}")
    print(f"target max length: {target_max_length}")
    print(f"total: {meta['total']}")


if __name__ == '__main__':
    fire.Fire({
        'run': main,
        'check': check,
    })

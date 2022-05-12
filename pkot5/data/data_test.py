import numpy as np
from transformers import T5TokenizerFast

from .data import DataCollatorForT5MLM


def test_data_collator():
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    input_ids = [222] * 566

    collator = DataCollatorForT5MLM(tokenizer, prefix="fill: ")
    batch = collator([{
        'input_ids': np.array(input_ids)
    }] * 32)

    assert batch is not None
    print(batch)

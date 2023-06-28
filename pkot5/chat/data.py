import json
import logging
import pickle

import fire
from torch.utils.data import Dataset
from transformers import T5TokenizerFast, PreTrainedTokenizer
from typing import *


logger = logging.getLogger(__name__)


def prepare_evolve_instruct(data_path, **kwargs):
    file_path = f"{data_path}/evol_instruct.json"
    with open(file_path, "rt") as f:
        dataset = json.load(f)

    input_tpl = "사용자가 한 말을 읽고 그에 질문에 답하거나 명령에 응답하는 비서입니다.\n\n사용자:\n{text}\n\n비서:\n"
    all_input_texts = [input_tpl.format(text=data['input']) for data in dataset]
    all_output_texts = [data['output'] for data in dataset]

    return all_input_texts, all_output_texts


def prepare_koalpaca_data(data_path: str):
    prompt_dict = {
        "prompt_input": (
            "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
            "Instruction(명령어):\n{instruction}\n\nInput(입력):\n{input}\n\nResponse(응답):\n"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task.\n"
            "아래는 작업을 설명하는 명령어입니다.\n\n"
            "Write a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
            "Instruction(명령어):\n{instruction}\n\nResponse(응답):"
        ),
    }

    logger.warning("Loading data...")
    with open(f"{data_path}/ko_alpaca_data.json", 'rt') as f:
        list_data_dict = json.load(f)

    logger.warning("Formatting inputs...")
    prompt_input, prompt_no_input = prompt_dict["prompt_input"], prompt_dict["prompt_no_input"]
    sources = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in list_data_dict
    ]
    targets = [f"{example['output']}" for example in list_data_dict]

    assert len(sources) == len(targets)
    return sources, targets


def prepare_koalpaca_v1_1(data_path: str):
    with open(f"{data_path}/KoAlpaca_v1.1a_textonly.json", 'rt') as f:
        records = [json.loads(line.strip()) for line in f]

    input_tpl = "다음 질문에 대해서 적절하게 비서가 답변해줍니다.\n\n질문:\n{question}\n\n답변:\n"
    inputs, targets = [], []
    for record in records:
        text = record['text'].strip()
        text = text.replace('<|endoftext|>', '')
        text, answer = text.split('### 답변:')
        _, question = text.split('### 질문:')
        question = question.strip()
        answer = answer.strip()
        inputs.append(input_tpl.format(question=question))
        targets.append(answer)
    return inputs, targets


def main(data_path: str):
    tokenizer = T5TokenizerFast.from_pretrained('paust/pko-t5-base')
    input_ids, target_ids = [], []
    for prepare_fn in [prepare_evolve_instruct, prepare_koalpaca_v1_1, prepare_koalpaca_data]:
        input_texts, target_texts = prepare_fn(data_path)
        encodings = tokenizer(input_texts + target_texts, add_special_tokens=True)
        input_ids += encodings.input_ids[:len(input_texts)]
        target_ids += encodings.input_ids[len(input_texts):]

    with open("./chat_t5_data.pkl", 'wb') as f:
        pickle.dump([{'input_ids': inp, 'target_ids': tgt} for inp, tgt in zip(input_ids, target_ids)], f)


if __name__ == '__main__':
    fire.Fire(main)

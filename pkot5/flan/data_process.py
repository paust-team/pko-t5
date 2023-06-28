import concurrent.futures
import json
import math
import os
import pickle
import random
from pathlib import Path
from typing import Iterable, Dict, Optional

import aimrocks
import fire
import grpc
import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, T5TokenizerFast

from . import pb
from .templates import TEMPLATES


def _tokenize(all_data, tokenizer, skip_input_max_length: bool = False, skip_output_max_length: bool = False, output_tokenizer_options = None):
    input_ids = tokenizer([t['input_text'] for t in all_data], add_special_tokens=True).input_ids
    for i in range(len(input_ids)):
        if skip_input_max_length:
            if len(input_ids[i]) > 1024:
                input_ids[i] = None
    input_ids = [t for t in input_ids if t is not None]
    max_len = max(len(ids) for ids in input_ids)
    num_long_ids = sum(1 if len(ids) > 1024 else 0 for ids in input_ids)
    assert max_len <= 1024, f"input max length: {max_len}, num long input: {num_long_ids}"

    if output_tokenizer_options is None:
        output_tokenizer_options = {}
    target_ids = tokenizer([t['output_text'] for t in all_data], add_special_tokens=True, **output_tokenizer_options).input_ids
    max_len = 0
    for i in range(len(target_ids)):
        max_len = max(max_len, len(target_ids[i]))
        if skip_output_max_length:
            if len(target_ids[i]) > 256:
                target_ids[i] = None
    target_ids = [t for t in target_ids if t is not None]
    max_len = max(len(t) for t in target_ids)
    num_long_ids = sum(1 if len(ids) > 256 else 0 for ids in target_ids)
    assert max_len <= 256, f"target max length: {max_len}, num long output: {num_long_ids}"
    return input_ids, target_ids


def nsmc(rng: random.Random, tokenizer: T5TokenizerFast, **kwargs):
    datasets = load_dataset("nsmc")
    label_to_text = {
        0: '부정',
        1: '긍정',
    }

    tmpl = TEMPLATES['nsmc']

    all_data = []
    options_ = "옵션:\n" + '\n'.join([f"- {t}" for t in label_to_text.values()])
    for data_split in ['train', 'test']:
        for data in datasets[data_split]:
            input_tmpl, output_tmpl = rng.choice(tmpl)
            sentence = data['document'].strip()
            answer = label_to_text[data['label']]

            input_text = input_tmpl.format(sentence=sentence, options_=options_, answer=answer)
            output_text = output_tmpl.format(sentence=sentence, options_=options_, answer=answer)

            all_data.append({
                'input_text': input_text,
                'output_text': output_text,
            })
    return _tokenize(all_data, tokenizer)


def _kor_hate_speech(rng, data_path):
    news_title_files = ['https://raw.githubusercontent.com/kocohub/korean-hate-speech/master/news_title/train.news_title.txt',
                        'https://raw.githubusercontent.com/kocohub/korean-hate-speech/master/news_title/dev.news_title.txt']
    news_titles = []
    for file in news_title_files:
        with requests.get(file) as resp:
            for line in resp.iter_lines(decode_unicode=True):
                news_titles.append(line)
    dataframes = [
        pd.read_csv("https://raw.githubusercontent.com/kocohub/korean-hate-speech/master/labeled/train.tsv", sep='\t'),
        pd.read_csv("https://raw.githubusercontent.com/kocohub/korean-hate-speech/master/labeled/dev.tsv", sep='\t')
    ]
    khs_df = pd.concat(dataframes)
    assert len(khs_df) == len(news_titles)
    khs_df['news_titles'] = news_titles

    prompt_template_1 = [
        "뉴스 제목에 대한 댓글이 성차별 또는 성적인 혐오발언을 포함하는지 판별하여 적절히 응답을 작성해주세요.\n\n- 뉴스 제목: \"{title}\"\n### 댓글: \"{comment}\"",
        "뉴스 제목: \"{title}\"\n댓글: \"{comment}\"\n\n위의 뉴스 제목과 댓글을 보고 성차별 또는 성적인 혐오발언을 포함하는지 판별하여 적절히 응답을 작성해주세요.",
        "뉴스 제목: \"{title}\"\n\n위의 뉴스 제목을 보고 아래 댓글이 성차별 또는 성적인 혐오발언을 포함하는지 판별하여 적절히 응답을 작성해주세요.\n\n\n댓글: \"{comment}\"",
    ]
    prompt_template_2 = [
        "아래 뉴스 제목을 보고 공격적이고 편견이 있는 댓글을 작성해주세요.\n\n뉴스 제목: \"{title}\"",
        "뉴스 제목: \"{title}\"\n\n위 뉴스 제목을 보고 공격적이고 편견이 있는 댓글을 작성해주세요.",
    ]
    prompt_template_3 = [
        "뉴스 제목을 보고 성차별 발언을 포함하는 공겨적인 댓글을 작성해주세요.\n\n뉴스 제목: \"{title}\"",
        "뉴스 제목: \"{title}\"\n\n위 뉴스 제목을 보고 성차별 발언을 포함하는 공겨적인 댓글을 작성해주세요.",
    ]
    prompt_template_4 = [
        "뉴스 제목을 보고 편견이 있으면서 비꼬는 댓글을 작성해주세요.\n\n뉴스 제목: \"{title}\"",
        "뉴스 제목: \"{title}\"\n\n위의 뉴스 제목을 보고 편견이 있으면서 비꼬는 댓글을 작성해주세요.",
    ]
    prompt_template_5 = [
        "뉴스 제목을 보고 성차별 발언을 포함하고 비꼬는 댓글을 작성해주세요.\n\n뉴스 제목: \"{title}\"",
        "뉴스 제목: \"{title}\"\n\n위 뉴스 제목을 보고 성차별 발언을 포함하고 비꼬는 댓글을 작성해주세요.",
    ]
    prompt_template_6 = [
        "뉴스 제목: \"{title}\"\n\n위 뉴스 제목을 보고 비꼬는 댓글을 작성해주세요.",
        "위 뉴스 제목을 보고 비꼬는 댓글을 작성해주세요.\n\n뉴스 제목: \"{title}\"",
    ]
    prompt_template_7 = [
        "뉴스 제목을 보고 공격적인 댓글을 작성해주세요.\n\n뉴스 제목: \"{title}\"",
        "뉴스 제목: \"{title}\"\n\n위 뉴스 제목을 보고 공격적인 댓글을 작성해주세요.",
    ]

    prompt_template_8 = [
        "뉴스 제목을 보고 혐오 발언이 없는 댓글을 작성해주세요.\n\n뉴스 제목: \"{title}\"",
        "뉴스 제목: \"{title}\"\n\n위 뉴스 제목을 보고 혐오 발언이 없는 댓글을 작성해주세요.",
    ]
    prompt_template_9 = [
        "뉴스 제목을 보고 성차별 발언을 포함하는 댓글을 작성해주세요.\n\n뉴스 제목: \"{title}\"",
        "뉴스 제목: \"{title}\"\n\n위 뉴스 제목을 보고 성차별 발언을 포함하는 댓글을 작성해주세요."
    ]
    prompt_template_10 = [
        "뉴스 제목: \"{title}\"\n\n위 뉴스 제목을 보고 편견이 있는 댓글을 작성해주세요."
    ]

    prompt_template_11 = [
        "뉴스 제목을 보고 아래의 댓글이 편견이 존재하는지 판단하여 적절히 응답을 작성해주세요.\n\n뉴스 제목: \"{title}\"\n댓글: \"{comment}\"",
        "뉴스 제목: \"{title}\"\n\n위 뉴스 제목을 보고 아래의 댓글이 편견이 존재하는지 판단하여 적절히 응답을 작성해주세요.\n\n댓글: \"{comment}\"",
        "뉴스 제목: \"{title}\"\n댓글: \"{comment}\"\n\n위 뉴스 제목과 댓글을 보고 편견이 존재하는지 판단하여 적절히 응답을 작성해주세요."
    ]
    prompt_template_12 = [
        "아래 뉴스 제목과 댓글을 보고 혐오적인지 판단하여 적절히 응답을 작성해주세요.\n\n뉴스 제목: \"{title}\"\n댓글: \"{comment}\""
        "뉴스 제목: \"{title}\"\n\n위 뉴스 제목을 보고 아래의 댓글이 혐오적인지 판단하여 적절히 응답을 작성해주세요.\n\n댓글: \"{comment}\""
        "뉴스 제목: \"{title}\"\n댓글: \"{comment}\"\n\n위 뉴스 제목과 댓글을 보고 혐오적인지 판단하여 적절히 응답을 작성해주세요."
    ]
    for i, row in khs_df.iterrows():
        cmt = str(row['comments']).strip()
        title = str(row['news_titles']).strip()
        contain_gender_bias = bool(row['contain_gender_bias'])
        if contain_gender_bias:
            label = "성적 혐오를 갖고 있습니다."
        else:
            label = "성적 혐오를 갖고 있지 않습니다."

        yield {
            'input_text': rng.choice(prompt_template_1).format(title=title, comment=cmt),
            'output_text': label,
        }

        bias = str(row['bias']).strip()
        hate = str(row['hate']).strip()
        if hate == 'offensive':
            if bias == 'others':
                input_text = rng.choice(prompt_template_2).format(title=title)
            elif bias == 'gender':
                input_text = rng.choice(prompt_template_3).format(title=title)
            elif bias == 'none':
                input_text = rng.choice(prompt_template_7).format(title=title)
            else:
                raise RuntimeError()
        elif hate == 'hate':
            if bias == 'others':
                input_text = rng.choice(prompt_template_4).format(title=title)
            elif bias == 'gender':
                input_text = rng.choice(prompt_template_5).format(title=title)
            elif bias == 'none':
                input_text = rng.choice(prompt_template_6).format(title=title)
            else:
                raise RuntimeError()
        elif hate == 'none':
            if bias == 'others':
                input_text = rng.choice(prompt_template_10).format(title=title)
            elif bias == 'gender':
                input_text = rng.choice(prompt_template_9).format(title=title)
            elif bias == 'none':
                input_text = rng.choice(prompt_template_8).format(title=title)
            else:
                raise RuntimeError()
        else:
            raise RuntimeError()
        yield {
            'input_text': input_text,
            'output_text': cmt,
        }

        if bias == 'none':
            label = "편견이 존재하지 않습니다."
        elif bias == 'gender':
            label = "성적 편견이 존재합니다."
        elif bias == 'others':
            label = "편견이 존재합니다."
        else:
            raise RuntimeError()

        yield {
            'input_text': rng.choice(prompt_template_11).format(title=title, comment=cmt),
            'output_text': label,
        }

        if hate == 'none':
            label = "혐오적이지 않습니다."
        elif hate == 'hate':
            label = '혐오적입니다.'
        elif hate == 'offensive':
            label = '공격적입니다.'
        else:
            raise RuntimeError()

        yield {
            'input_text': rng.choice(prompt_template_12).format(title=title, comment=cmt),
            'output_text': label,
        }


def kor_hate_speech(rng, data_path, tokenizer, **kwargs):
    all_data = [record for record in _kor_hate_speech(rng, data_path)]
    return _tokenize(all_data, tokenizer)


def kor_nli(rng, data_path, tokenizer, **kwargs):
    all_data = []
    file_path = f"{data_path}/kornli/"
    files = [os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files]
    options_ = "옵션:\n- 네\n- 아니오"
    with tqdm(files, total=len(files)) as pbar:
        for file_name in pbar:
            with open(file_name, 'r') as file_object:
                file_object.readline()
                for line in file_object:
                    line = line.strip()
                    line = [t.strip() for t in line.split('\t')]
                    sentence1, sentence2, answer = line[:3]

                    input_template, output_template = rng.choice(TEMPLATES['kor_nli'])

                    all_data.append({
                        'input_text': input_template.format(sentence1=sentence1, sentence2=sentence2, answer=answer, options_=options_),
                        'output_text': output_template.format(sentence1=sentence1, sentence2=sentence2, answer=answer, options_=options_),
                    })
    return _tokenize(all_data, tokenizer)


def kor_sts(rng, data_path, tokenizer, **kwargs):
    all_data = []
    file_path = f"{data_path}/korsts/"
    files = [os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files]

    pbar = tqdm(total=len(files))
    options_ = "옵션: 0에서 5사이의 실수"
    for file_name in files:
        with open(file_name, 'r') as file_object:
            file_object.readline()
            for line in file_object:
                line = line.strip()
                line = [str(t).strip() for t in line.split('\t')]
                sentence1 = line[5]
                sentence2 = line[6]
                answer_str = line[4]

                input_template, output_template = rng.choice(TEMPLATES['kor_sts'])
                all_data.append({
                    'input_text': input_template.format(sentence1=sentence1, sentence2=sentence2, options_=options_, answer_str=answer_str),
                    'output_text': output_template.format(sentence1=sentence1, sentence2=sentence2, options_=options_, answer_str=answer_str)
                })

        pbar.update(1)
    pbar.close()
    return _tokenize(all_data, tokenizer)


def question_pair(rng, data_path, tokenizer, **kwargs):
    all_data = []
    df = pd.read_csv(f"{data_path}/question_pair/kor_pair_train.csv", header=0)

    options_ = "옵션:\n- 네\n- 아니오"
    for i, row in df.iterrows():
        question1 = row['question1'].strip()
        question2 = row['question2'].strip()
        answer = "네" if row['is_duplicate'] == 1 else "아니오"

        input_template, output_template = rng.choice(TEMPLATES['question_pair'])
        all_data.append({
            'input_text': input_template.format(question1=question1, question2=question2, answer=answer, options_=options_),
            'output_text': output_template.format(question1=question1, question2=question2, answer=answer, options_=options_),
        })
    return _tokenize(all_data, tokenizer)


def klue_ynat(rng, tokenizer, **kwargs):
    all_data = []
    datasets = load_dataset("klue", "ynat")
    label2text = {
        0: "IT과학",
        1: "경제",
        2: "사회",
        3: "생활문화",
        4: "세계",
        5: "스포츠",
        6: "정치",
    }
    options_ = "옵션:\n" + '\n'.join([f"- {t}" for t in label2text.keys()])
    for row in datasets['train']:
        title = row['title'].strip()
        answer = label2text[int(row['label'])]

        input_template, output_template = rng.choice(TEMPLATES['klue_ynat'])
        all_data.append({
            'input_text': input_template.format(title=title, answer=answer, options_=options_),
            'output_text': output_template.format(title=title, answer=answer, options_=options_),
        })
    return _tokenize(all_data, tokenizer)


def korquad_v1(rng, tokenizer, **kwargs):
    all_data = []
    datasets = load_dataset('squad_kor_v1')
    all_context_ids = tokenizer([data['context'].strip() for data in datasets['train']], add_special_tokens=False).input_ids

    for data, context_ids in tqdm(zip(datasets['train'], all_context_ids), total=len(datasets['train'])):
        question = data['question'].strip()
        title = data['title'].strip()
        answer = data['answers']['text'][0].strip()

        for context in tokenizer.batch_decode([context_ids[i:i + 768] for i in range(0, len(context_ids), 256)], skip_special_tokens=True):
            if answer in context:
                input_tmpl, output_tmpl = rng.choice(TEMPLATES['korquad_v1'])

                all_data.append({
                    'input_text': input_tmpl.format(question=question, title=title, context=context, answer=answer),
                    'output_text': output_tmpl.format(question=question, title=title, context=context, answer=answer)
                })
    return _tokenize(all_data, tokenizer)


def klue_nli(rng, tokenizer, **kwargs):
    all_data = []
    datasets = load_dataset('klue', 'nli')

    options_ = "옵션:\n- 네\n- 아니오\n- 애매합니다."
    for data in datasets['train']:
        premise = data['premise'].strip()
        hypothesis = data['hypothesis'].strip()
        label = data['label']
        if label == 0:
            answer = "네"
        elif label == 2:
            answer = "아니오"
        elif label == 1:
            answer = "애매합니다."
        else:
            raise RuntimeError()

        input_template, output_template = rng.choice(TEMPLATES['klue_nli'])
        all_data.append({
            'input_text': input_template.format(premise=premise, hypothesis=hypothesis, options_=options_, answer=answer),
            'output_text': output_template.format(premise=premise, hypothesis=hypothesis, options_=options_, answer=answer)
        })
    return _tokenize(all_data, tokenizer)


def _generate_mrc_v2(rng, tokenizer, context_ids, question, title, answers, is_impossible, context_max_length=512) -> Iterable[Dict[str, str]]:
    templates = TEMPLATES['klue_mrc']
    if title is None:
        templates = [t for t in templates if '{title}' not in t]

    if is_impossible:
        answer = '답변할 수 없음'
        for context in tokenizer.batch_decode([context_ids[i:i+context_max_length] for i in range(0, len(context_ids), context_max_length)], skip_special_tokens=True):
            input_tmpl, output_tmpl = rng.choice(templates)
            yield {
                'input_text': input_tmpl.format(question=question, title=title, context=context, answer=answer),
                'output_text': output_tmpl.format(question=question, title=title, context=context, answer=answer)
            }
    else:
        for context in tokenizer.batch_decode([context_ids[i:i+context_max_length] for i in range(0, len(context_ids), int(context_max_length/3))], skip_special_tokens=True):
            for answer in answers:
                if answer in context:
                    input_tmpl, output_tmpl = rng.choice(templates)
                    yield {
                        'input_text': input_tmpl.format(question=question, title=title, context=context, answer=answer),
                        'output_text': output_tmpl.format(question=question, title=title, context=context, answer=answer)
                    }
                    break


def klue_mrc(rng, tokenizer: PreTrainedTokenizer, **kwargs):
    all_data = []
    datasets = load_dataset('klue', 'mrc')
    all_context_ids = tokenizer([data['context'].strip() for data in datasets['train']], add_special_tokens=False).input_ids

    for data, context_ids in zip(datasets['train'], all_context_ids):
        question = data['question'].strip()
        title = data['title'].strip()
        is_impossible = bool(data['is_impossible'])
        answers = data['answers']['text']

        for record in _generate_mrc_v2(rng, tokenizer, context_ids, question, title, answers, is_impossible, context_max_length=768):
            all_data.append(record)
    return _tokenize(all_data, tokenizer)


def ai_hub_law_summ(rng, data_path, tokenizer, **kwargs):
    all_data = []
    filenames = ['train_law.json', 'valid_law.json']
    for file_name in tqdm(filenames):
        file_path = f"{data_path}/ai_hub_doc_summ/{file_name}"

        with open(file_path, "rt") as file_object:
            documents = json.load(file_object)["documents"]

        for doc in documents:
            if doc['size'] == 'large':
                continue
            sentences = []
            for d in doc['text']:
                for dd in d:
                    sentences.append(dd['sentence'].strip())
            text = '\n'.join(sentences).strip()
            title = doc['title'].strip()
            abstractive = doc['abstractive'][0].strip()

            input_template, output_template = rng.choice(TEMPLATES['ko_summary2'])
            all_data.append({
                'input_text': input_template.format(title=title, text=text, summary=abstractive),
                'output_text': output_template.format(title=title, text=text, summary=abstractive),
            })

    return _tokenize(all_data, tokenizer, skip_input_max_length=True, output_tokenizer_options={'truncation': True, 'max_length': 256})


def ai_hub_article_and_editorial_summ(rng, data_path, tokenizer, **kwargs):
    all_data = []
    filenames = ['train_editorial.json', 'train_article.json', 'valid_editorial.json', 'valid_article.json']
    for filename in tqdm(filenames):
        file_path = f"{data_path}/ai_hub_doc_summ/{filename}"
        with open(file_path, "rt") as f:
            documents = json.load(f)["documents"]

        for doc in documents:
            if doc['size'] == 'large':
                continue
            extractive_indices = doc['extractive']
            sentences = []
            for d in doc['text']:
                in_ext = False
                paragraph = []
                for dd in d:
                    paragraph.append(dd['sentence'].strip())
                    in_ext = True
                if in_ext:
                    sentences += paragraph

            text = '\n'.join(sentences).strip()
            title = doc['title'].strip()
            abstractive = doc['abstractive'][0].strip()

            input_template, output_template = rng.choice(TEMPLATES['ko_summary_newsroom'])
            all_data.append({
                'input_text': input_template.format(title=title, text=text, summary=abstractive),
                'output_text': output_template.format(title=title, text=text, summary=abstractive),
            })

    return _tokenize(all_data, tokenizer, skip_input_max_length=True, output_tokenizer_options={'truncation': True, 'max_length': 256})


def _decode_contexts(tokenizer, all_context_ids, context_max_length, skip_length):
    all_sub_context_ids = []
    for i, context_ids in enumerate(all_context_ids):
        sub_context_ids = []
        for j in range(0, len(context_ids), skip_length):
            sub_context_ids.append(context_ids[j:j + context_max_length])
        all_sub_context_ids.append(sub_context_ids)

    flatten = [
        (ids2, i, j)
        for i, ids1 in enumerate(all_sub_context_ids)
        for j, ids2 in enumerate(ids1)
    ]
    decoded = tokenizer.batch_decode([t[0] for t in flatten], skip_special_tokens=True)
    all_sub_contexts = all_sub_context_ids
    for text, (_, i, j) in zip(decoded, flatten):
        all_sub_contexts[i][j] = text

    return all_sub_contexts


def mindslab_mrc(rng, data_path, tokenizer, **kwargs):
    file_path = f"{data_path}/mindslab_mrc/"
    files = [os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files]
    templates = TEMPLATES['klue_mrc']
    templates = [t for t in templates if '{title}' not in t[0]]
    context_max_length = 768

    all_input_ids, all_target_ids = [], []
    for file_name in tqdm(files, position=0):
        with open(file_name, "rb") as file_object:
            datasets = json.loads(file_object.read().decode('utf-8'))["data"]
        contexts = [data['paragraphs'][0]['context'].strip() for data in datasets]
        all_context_ids = tokenizer(contexts, add_special_tokens=False).input_ids

        all_contexts_1 = _decode_contexts(tokenizer, all_context_ids, context_max_length, skip_length=context_max_length)
        all_contexts_2 = _decode_contexts(tokenizer, all_context_ids, context_max_length, skip_length=int(context_max_length/3))
        all_data = []

        for data, contexts_1, contexts_2 in tqdm(zip(datasets, all_contexts_1, all_contexts_2), total=len(datasets), position=1):
            paragraph = data["paragraphs"][0]
            for qa in paragraph["qas"]:
                question = qa['question'].strip()
                is_impossible = 'answers' not in qa.keys()
                answers = [a['text'] for a in qa['answers']] if not is_impossible else []

                if is_impossible:
                    answer = '답변할 수 없음'
                    for context in contexts_1:
                        input_tmpl, output_tmpl = rng.choice(templates)
                        all_data.append({
                            'input_text': input_tmpl.format(question=question, context=context, answer=answer),
                            'output_text': output_tmpl.format(question=question, context=context, answer=answer)
                        })
                else:
                    for context in contexts_2:
                        for answer in answers:
                            if answer in context:
                                input_tmpl, output_tmpl = rng.choice(templates)
                                all_data.append({
                                    'input_text': input_tmpl.format(question=question, context=context, answer=answer),
                                    'output_text': output_tmpl.format(question=question, context=context, answer=answer)
                                })
                                break

        input_ids, target_ids = _tokenize(all_data, tokenizer, skip_output_max_length=True)
        all_input_ids += input_ids
        all_target_ids += target_ids
    return all_input_ids, all_target_ids


def ai_hub_book_summ(rng, data_path, tokenizer, **kwargs):
    all_data = []
    file_path = f"{data_path}/ai_hub_book_summ/"
    files = [os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files]
    pbar = tqdm(total=len(files))
    for file_name in files:  # 1 data per 1 file
        with open(f"{file_name}", "rb") as file_object:
            datasets = json.loads(file_object.read().decode('utf-8'))
            passage = datasets['passage'].strip()
            summary = datasets['summary'].strip()

            input_tmpl, output_tmpl = rng.choice(TEMPLATES['ko_summary1'])
            all_data.append({
                'input_text': input_tmpl.format(text=passage, summary=summary),
                'output_text': output_tmpl.format(text=passage, summary=summary),
            })
        pbar.update(1)
    pbar.close()
    return _tokenize(all_data, tokenizer, output_tokenizer_options={'truncation': True, 'max_length': 256})


def _generate_ai_hub_trans(rng, sentences_src, sentences_tgt, lang1, lang2):
    for sent1, sent2 in zip(sentences_src, sentences_tgt):
        sent1 = sent1.strip()
        sent2 = sent2.strip()
        lang1 = lang1.strip()
        lang2 = lang2.strip()

        input_tpl, output_tpl = rng.choice(TEMPLATES['kor_translate'])
        yield {
            'input_text': input_tpl.format(sent1=sent1, sent2=sent2, lang1=lang1, lang2=lang2),
            'output_text': output_tpl.format(sent1=sent1, sent2=sent2, lang1=lang1, lang2=lang2),
        }


def ai_hub_kor2eng(rng, data_path, tokenizer, **kwargs):
    all_data = []
    file_path = f"{data_path}/ai_hub_kor2eng/"
    files = [file_path + f for f in os.listdir(file_path)]
    for data_file in tqdm(files):
        df = pd.read_excel(data_file, header=0)
        all_data += list(_generate_ai_hub_trans(
            rng,
            df['원문'].tolist(),
            df['번역문'].tolist(),
            lang1='한국어',
            lang2='영어',
        ))
    return _tokenize(all_data, tokenizer, output_tokenizer_options={'truncation': True, 'max_length': 256})


def ai_hub_kor2eng_expert(rng, data_path, tokenizer, **kwargs):
    all_data = []
    file_path = f"{data_path}/ai_hub_kor2eng_expert/"
    files = [file_path + f for f in os.listdir(file_path)]
    for data_file in tqdm(files):
        df = pd.read_csv(data_file, header=0)
        all_data += list(_generate_ai_hub_trans(
            rng,
            df['한국어'].tolist(),
            df['영어'].tolist(),
            lang1='한국어',
            lang2='영어',
        ))
    return _tokenize(all_data, tokenizer, output_tokenizer_options={'truncation': True, 'max_length': 256})


def ai_hub_kor2eng_tech_and_social(rng, data_path, **kwargs):
    all_data = []
    for data_name in ['ai_hub_kor2eng_technology', 'ai_hub_kor2eng_socialscience']:
        file_path = f"{data_path}/{data_name}/"
        files = [file_path + f for f in os.listdir(file_path)]
        for data_file in tqdm(files):
            df = pd.read_csv(data_file, header=0)
            all_data += list(_generate_ai_hub_trans(
                rng,
                df['ko'].tolist(),
                df['en'].tolist(),
                lang1='한국어',
                lang2='영어',
            ))
    return _tokenize(all_data, kwargs['tokenizer'], output_tokenizer_options={'truncation': True, 'max_length': 256})


def ai_hub_kor2jpn(rng, data_path, tokenizer, **kwargs):
    all_data = []
    file_path = f"{data_path}/ai_hub_kor2jpn/"
    files = [file_path + f for f in os.listdir(file_path)]
    for data_file in tqdm(files):
        df = pd.read_csv(data_file, header=0)
        all_data += list(_generate_ai_hub_trans(
            rng,
            df['한국어'].tolist(),
            df['일본어'].tolist(),
            '한국어',
            '일본어',
        ))
    return _tokenize(all_data, tokenizer, output_tokenizer_options={'truncation': True, 'max_length': 256})


def ai_hub_kor2chn_tech_and_social(rng, data_path, tokenizer, **kwargs):
    all_data = []
    for data_name in ["ai_hub_kor2chn_technology", "ai_hub_kor2chn_socialscience"]:
        file_path = f"{data_path}/{data_name}/"
        files = [file_path + f for f in os.listdir(file_path)]
        for data_file in tqdm(files):
            df = pd.read_csv(data_file, header=0)
            all_data += list(_generate_ai_hub_trans(
                rng,
                df['한국어'].tolist(),
                df['중국어'].tolist(),
                '한국어',
                '중국어',
            ))
    return _tokenize(all_data, tokenizer, output_tokenizer_options={'truncation': True, 'max_length': 256})


def ai_hub_translation_corpus(rng, data_path, tokenizer, **kwargs):
    all_data = []
    lang2text = {
        'ko': '한국어',
        'en': '영어',
        'jp': '일본어',
        'cn': '중국어',
    }
    data_names = ["ai_hub_food_translation_corpus",
                  "ai_hub_broadcasting_translation_corpus",
                  "ai_hub_casualtalk_translation",
                  "ai_hub_tech_translation_corpus"]
    all_input_ids, all_target_ids = [], []
    for data_name in data_names:
        for data_split in ['train', 'validation']:
            file_path = f"{data_path}/{data_name}/{data_split}/"
            files = [os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files]
            for file_name in tqdm(files):
                with open(f"{file_name}", "rb") as file_object:
                    datasets = json.loads(file_object.read().decode('utf-8'))['data']
                for data in datasets:
                    source_lang = data['source_language']
                    target_lang = data['target_language']

                    if source_lang != 'ko':
                        target_lang = 'ko'

                    if source_lang == 'ko' or target_lang == 'ko':
                        sent1 = data[f'{source_lang}_original'].strip()
                        sent2 = data[target_lang].strip()
                        lang1 = lang2text[source_lang]
                        lang2 = lang2text[target_lang]

                        input_tpl, output_tpl = rng.choice(TEMPLATES['kor_translate'])
                        all_data.append({
                            'input_text': input_tpl.format(sent1=sent1, sent2=sent2, lang1=lang1, lang2=lang2),
                            'output_text': output_tpl.format(sent1=sent1, sent2=sent2, lang1=lang1, lang2=lang2),
                        })
        results = _tokenize(all_data, tokenizer, skip_input_max_length=True, output_tokenizer_options={'truncation': True, 'max_length': 256})
        all_input_ids += results[0]
        all_target_ids += results[1]
    return all_input_ids, all_target_ids


def ai_hub_conversation_summ(rng, data_path, tokenizer, **kwargs):
    all_data = []
    for data_split in ["train", "validation"]:
        file_path = f"{data_path}/ai_hub_conversation_summ/{data_split}/"
        files = [file_path + f for f in os.listdir(file_path)]
        pbar = tqdm(total=len(files))
        for i, file_name in enumerate(files):
            with open(f"{file_name}", "rb") as file_object:
                datasets = json.loads(file_object.read().decode('utf-8'))["data"]

            for data in datasets:
                chat_logs, summary = [], []
                participants = data['header']['participantsInfo']
                participants = {participants[i]['participantID']: participants[i]['age'] + participants[i]['gender'] for i in range(len(participants))}

                for d in data['body']['dialogue']:
                    chat_logs.append(f"{participants[d['participantID']]}: {d['utterance']}")

                summary = data['body']['summary']
                dialogue = '\n'.join(chat_logs)

                input_tpl, output_tpl = rng.choice(TEMPLATES['ko_summary_dialogue'])
                all_data.append({
                    'input_text': input_tpl.format(dialogue=dialogue, summary=summary),
                    'output_text': output_tpl.format(dialogue=dialogue, summary=summary),
                })
            pbar.update(1)
        pbar.close()
    return _tokenize(all_data, tokenizer, skip_input_max_length=True, output_tokenizer_options=dict(truncation=True, max_length=256))


def ai_hub_news_summ(rng, data_path, tokenizer, **kwargs):
    all_data = []
    file_path = f"{data_path}/ai_hub_news_summ/"
    files = [file_path + f for f in os.listdir(file_path) if f.endswith('.json')]

    for file_name in tqdm(files):
        with open(f"{file_name}", "rt") as file_object:
            datasets = json.load(file_object)

        for data in datasets:
            sentences = [t.strip() for t in data['article_original']]
            sentences = sentences[:8]
            text = '\n'.join(sentences)
            summary = data['abstractive'].strip()

            input_tpl, output_tpl = rng.choice(TEMPLATES['ko_summary_news'])
            all_data.append({
                'input_text': input_tpl.format(text=text, summary=summary),
                'output_text': output_tpl.format(text=text, summary=summary),
            })
    return _tokenize(all_data, tokenizer, skip_input_max_length=True, output_tokenizer_options={'truncation': True, 'max_length': 256})


def klue_sts(rng, tokenizer, **kwargs):
    datasets = load_dataset('klue', 'sts')
    label2text = {
        0: "아니오",
        1: "네",
    }
    options_ = "옵션:\n- 네\n- 아니오"
    all_data = []
    for data in datasets['train']:
        sentence1 = data['sentence1'].strip()
        sentence2 = data['sentence2'].strip()
        label = data['labels']['binary-label']
        answer = label2text[label]

        input_tpl, output_tpl = rng.choice(TEMPLATES['klue_sts'])
        all_data.append({
            'input_text': input_tpl.format(sentence1=sentence1, sentence2=sentence2, answer=answer, options_=options_),
            'output_text': output_tpl.format(sentence1=sentence1, sentence2=sentence2, answer=answer, options_=options_),
        })
    return _tokenize(all_data, tokenizer)


def main(output_file: str, data_path: str, seed: int = 42):
    processors = [
        nsmc,
        kor_hate_speech,
        kor_nli,
        kor_sts,
        question_pair,
        klue_sts,

        # ai_hub_news_summ,
        # ai_hub_article_and_editorial_summ,
        # ai_hub_law_summ,
        # ai_hub_book_summ,
        # ai_hub_conversation_summ,

        # ai_hub_kor2eng,
        # ai_hub_kor2eng_expert,
        # ai_hub_kor2eng_tech_and_social,
        # ai_hub_kor2jpn,
        # ai_hub_kor2chn_tech_and_social,
        # ai_hub_translation_corpus,

        # klue_ynat,
        # korquad_v1,
        # klue_mrc,
        # mindslab_mrc,
    ]

    rng = random.Random(seed)
    tokenizer = T5TokenizerFast.from_pretrained("paust/pko-t5-base")
    all_input_ids, all_target_ids = [], []
    for process_fn in processors:
        print(f"Processing for {process_fn}")
        input_ids, target_ids = process_fn(rng=rng, data_path=data_path, tokenizer=tokenizer)
        all_input_ids += input_ids
        all_target_ids += target_ids
    with open(output_file, 'wb') as f:
        pickle.dump({'all_input_ids': all_input_ids, 'all_target_ids': all_target_ids}, f)


def make_db(data_dir: str, db_path: str):
    data_dir = Path(data_dir)
    db = aimrocks.DB(db_path, aimrocks.Options(create_if_missing=True), read_only=False)
    total = 0
    for i in range(4):
        with open(data_dir / f"pko-flant5.{i:02d}.pkl", 'rb') as f:
            data = pickle.load(f)
        batch = aimrocks.WriteBatch()
        for input_ids, target_ids in tqdm(zip(data['all_input_ids'], data['all_target_ids']), total=len(data['all_input_ids'])):
            key = f"{total}".encode('utf-8')
            value = pickle.dumps({'input_ids': input_ids, 'target_ids': target_ids})
            batch.put(key, value)
            total += 1
        db.write(batch)
    db.put(b'metadata', pickle.dumps({'total': total}), sync=True)


class FlanDatasetServicer(pb.FlanProcessingDatasetServicer):
    def __init__(self, db: aimrocks.DB):
        self.db = db

    def Get(self, request: pb.GetReq, context) -> pb.GetResp:
        data = pickle.loads(self.db.get(f"{request.index}".encode('utf-8')))
        return pb.GetResp(
            input_ids=data['input_ids'],
            target_ids=data['target_ids'],
        )

    def Metadata(self, request, context) -> pb.MetadataResp:
        metadata = pickle.loads(self.db.get(b'metadata'))
        return pb.MetadataResp(total=metadata['total'])


def serve(port: int, db_path: str, num_workers=4):
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(num_workers))
    servicer = FlanDatasetServicer(
        aimrocks.DB(db_path, aimrocks.Options(), read_only=True),
    )
    pb.add_FlanProcessingDatasetServicer_to_server(servicer, server)
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    print(f"Start server")
    server.wait_for_termination()


if __name__ == '__main__':
    fire.Fire({
        'run': main,
        'make-db': make_db,
        'serve': serve,
    })

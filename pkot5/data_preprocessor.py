import copy
import re
from pprint import pprint
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import json
import random
import pickle
import os
import fire

from transformers import PreTrainedTokenizer, AutoTokenizer
from datasets import load_dataset
import traceback

tokenizer = AutoTokenizer.from_pretrained("paust/pko-t5-base")
# DATA_PATH = "/home/jovyan/data/ex-pkot5/"
DATA_PATH = "/home/jovyan/temp"
### Template of each dataset
# {
#     data_name : DATA_NAME ,
#     task_family : unsupervised | classification | dialogue | ... <- according to this, add rules to make prefix
#     input_ids : ... ,
#     target_ids : ... , <- make it as text, as possible
# }
#####

def preprocess_dataset(data_name, task_family=None):
    all_data = []
    if data_name == '3i4k':
        file_path = f"{DATA_PATH}/{data_name}/"
        files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
        label_to_text = {
            '0': '조각',
            '1': '설명문',
            '2': '질문',
            '3': '명령',
            '4': '수사적 질문',
            '5': '수사적 명령',
            '6': '억양이 있는 발화',
        }
        
        pbar = tqdm(total = len(files))
        for file_name in files:
            file_object = open(f"{file_name}", "r")
            input_texts = []
            label_texts = []
        
            for line in file_object:
                line = line.split('\t')
                input_texts.append(line[1].strip())
                label_texts.append(label_to_text[line[0].strip()])

            for input_text, label_text in zip(input_texts, label_texts):
                all_data.append({
                    "data_name": data_name,
                    "task_family": "classification",
                    'input_text': input_text,
                    'output_text': label_text,
                    'instruction': "다음의 질문을 보고 한국어의 억양 보조 의도를 식별하는 적절한 응답을 작성하세요.",
                })
            file_object.close()
            pbar.update(1)
        pbar.close()
        
    elif data_name == 'nsmc':
        datasets = load_dataset("nsmc")
        label_to_text = {
            0: '부정',
            1: '긍정',
        }

        for data_split in ['train', 'test']:    
            input_texts = []
            label_texts = []
            for data in datasets[data_split]:
                input_texts.append(data['document'].strip())
                label_texts.append(label_to_text[data['label']])
            
        input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
        label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids
            
        for i in range(len(input_text_ids)):
            all_data.append({
                "data_name": data_name,
                "task_family": "classification",
                'input_ids': input_text_ids[i],
                'output_ids': label_ids[i],
                'input_text': input_texts[i],
                'output_text': label_texts[i],
            })
                
    elif data_name == 'toxic_comment':
        df = pd.read_csv(f"{DATA_PATH}/Toxic_comment_data/ko_train_label.csv", header=0)
        input_texts = []
        label_texts = []
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            for k, v in row.items():
                if v == 1:
                    label_texts.append(k)
            input_texts.append(row['document'].strip())
            
        input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
        label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids

        for i in range(len(input_text_ids)):
            all_data.append({
                "data_name": data_name,
                "task_family": "classification",
                'input_ids': input_text_ids[i],
                'output_ids': label_ids[i],
                'input_text': input_texts[i],
                'output_text': label_texts[i],
            })
            
    elif data_name == 'korean_hate_speech':
        df = pd.read_csv(f"{DATA_PATH}/korean_hate_speech/labeled/train.tsv", sep='\t', header=0)
        input_texts, label_texts = [], []
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            input_texts.append(row['comments'].strip())
            for label_type in ['contain_gender_bias', 'bias', 'hate']:
                if label_type == 'contain_gender_bias':
                    if row[label_type]:
                        label_texts.append('biased')
                    else:
                        label_texts.append('unbiased')
                elif label_type == 'bias':
                    if row[label_type] == 'none':
                        label_texts.append('no bias')
                    else:
                        label_texts.append(row[label_type])
                elif label_type == 'hate':
                    if row[label_type] == 'none':
                        label_texts.append('no hate')
                    else:
                        label_texts.append(row[label_type])
                
        input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
        label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids

        for i in range(len(input_text_ids)):
            all_data.append({
                "data_name": data_name,
                "task_family": "classification",
                'input_ids': input_text_ids[i],
                'output_ids': label_ids[i],
                'input_text': input_texts[i],
                'output_text': label_texts[i],
            })
                
    elif data_name == 'kornli':
        file_path = f"{DATA_PATH}/{data_name}/"
        files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
        print(files)
        pbar = tqdm(total = len(files))
        for file_name in files:
            file_object = open(file_name, 'r')
            file_object.readline()
            input_texts = []
            label_texts = []
            for line in file_object:
                line = line.strip()
                line = line.split('\t')
                input_texts.append(line[0].strip() + ' ' + line[1].strip())
                label_texts.append(line[2].strip())
            
            input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
            label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids

            for i in range(len(input_text_ids)):
                all_data.append({
                    "data_name": data_name,
                    "task_family": "natural language inference",
                    'input_ids': input_text_ids[i],
                    'output_ids': label_ids[i],
                    'input_text': input_texts[i],
                    'output_text': label_texts[i],
                })                
            file_object.close()
            
            pbar.update(1)
        pbar.close()
    
    elif data_name == 'korsts':
        file_path = f"{DATA_PATH}/{data_name}/"
        files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]

        pbar = tqdm(total = len(files))
        for file_name in files:
            file_object = open(file_name, 'r')
            input_texts = []
            label_texts = []
            for line in file_object:
                line = line.strip()
                line = line.split('\t')
                input_texts.append(line[5].strip() + ' ' + line[6].strip())
                label_texts.append(line[4].strip())
            
            input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
            label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids

            for i in range(len(input_text_ids)):
                all_data.append({
                    "data_name": data_name,
                    "task_family": "semantic textual similarity",
                    'input_ids': input_text_ids[i],
                    'output_ids': label_ids[i],
                    'input_text': input_texts[i],
                    'output_text': label_texts[i],
                })                
            file_object.close()
            
            pbar.update(1)
        pbar.close()
     
    elif data_name == 'question_pair':
        df = pd.read_csv(f"{DATA_PATH}/question_pair/kor_pair_train.csv", header=0)
        input_texts, label_texts = [], []
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            input_texts.append(row['question1'].strip() + ' ' + row['question2'].strip())
            if row['is_duplicate'] == 1:
                label_texts.append("duplicate")
            else:
                label_texts.append("not duplicate")
            
        input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
        label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids

        for i in range(len(input_text_ids)):
            all_data.append({
                "data_name": data_name,
                "task_family": "semantic textual similarity",
                'input_ids': input_text_ids[i],
                'output_ids': label_ids[i],
                'input_text': input_texts[i],
                'output_text': label_texts[i],
            })                

    elif data_name == 'paraKQC':
        df = pd.read_csv(f"{DATA_PATH}/paraKQC/data/paraKQC_v1_generated.tsv", sep='\t')
        input_texts, label_texts = [], []
        for i in range(len(df)):
            row = df.iloc[i].to_list()
            input_texts.append(row[0].strip() + ' ' + row[1].strip())
            label_texts.append(row[2].strip())
        
        input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
        label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids

        for i in range(len(input_text_ids)):
            all_data.append({
                "data_name": data_name,
                "task_family": "semantic textual similarity",
                'input_ids': input_text_ids[i],
                'output_ids': label_ids[i],
                'input_text': input_texts[i],
                'output_text': label_texts[i],
            })    
    
    elif data_name == "korquad_v1":
        datasets = load_dataset('squad_kor_v1')
        for data_split in ['train', 'validation']:
            input_texts, label_texts = [], []
            for data in datasets[data_split]:
                input_texts.append(f"질의: {data['question']} 제목: {data['title']} 본문: {data['context']}".strip())
                label_texts.append(data['answers']['text'][0].strip())
                
            input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
            label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids

            for i in range(len(input_text_ids)):
                all_data.append({
                    "data_name": data_name,
                    "task_family": "question answering",
                    'input_ids': input_text_ids[i],
                    'output_ids': label_ids[i],
                    'input_text': input_texts[i],
                    'output_text': label_texts[i],
                })
                
    elif data_name == "korquad_v2": # Not use
        datasets = load_dataset('squad_kor_v2')
        for data_split in ['train', 'validation']:
            input_texts, label_texts = [], []
            for data in datasets[data_split]:
                input_texts.append(f"질의: {data['question']} 제목: {data['title']} 본문: {data['context']}".strip())
                label_texts.append(data['answers']['text'][0].strip())
                
            input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
            label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids

            for i in range(len(input_text_ids)):
                all_data.append({
                    "data_name": data_name,
                    "task_family": "question answering",
                    'input_ids': input_text_ids[i],
                    'output_ids': label_ids[i],
                    'input_text': input_texts[i],
                    'output_text': label_texts[i],
                })
                
    elif data_name == "sci-news-sum-kr-50":
        file_path = f"{DATA_PATH}/{data_name}/"
        files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
        input_texts, label_texts = [], []
        for file in files:
            file_object = open(file, "rb")
            data = json.loads(file_object.read().decode('utf-8'))
            input_texts.append(f"제목: {data['title']} 본문: {'. '.join(data['sentences']).strip()}")
            label_texts.append('. '.join([ data['sentences'][s] for s in data['summaries'] ]).strip())
            file_object.close()
            
        input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
        label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids

        for i in range(len(input_text_ids)):
            all_data.append({
                "data_name": data_name,
                "task_family": "extractive summarization",
                'input_ids': input_text_ids[i],
                'output_ids': label_ids[i],
                'input_text': input_texts[i],
                'output_text': label_texts[i],
            })
    
    elif data_name == "sae4k":
        file_path = f"{DATA_PATH}/{data_name}/data/"
        files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
        for file_name in files:
            file_object = open(file_name, 'r')
            file_object.readline()
            input_texts, label_texts = [], []
            for line in file_object:
                line = line.strip()
                line = line.split('\t')
                if file_name[-6:] == "v1.txt":
                    input_texts.append(line[0].strip())
                    label_texts.append(line[1].strip())
                elif file_name[-6:] == "v2.txt":
                    input_texts.append(line[1].strip())
                    label_texts.append(line[2].strip())
            file_object.close()
            
            input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
            label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids

            for i in range(len(input_text_ids)):
                all_data.append({
                    "data_name": data_name,
                    "task_family": "abstractive summarization", # short -> very short
                    'input_ids': input_text_ids[i],
                    'output_ids': label_ids[i],
                    'input_text': input_texts[i],
                    'output_text': label_texts[i],
                })
            
    elif data_name == "korean_parallel":
        data_object_ko = open(f"{DATA_PATH}/korean_parallel/korean-english-park.train.ko", 'r')
        data_object_en = open(f"{DATA_PATH}/korean_parallel/korean-english-park.train.en", 'r')
        
        input_texts, label_texts = [], []
        for line_ko, line_en in zip(data_object_ko, data_object_en):
            line_ko, line_en = line_ko.strip(), line_en.strip()
            input_texts.append(line_ko)
            label_texts.append(line_en)
            
        data_object_ko.close()
        data_object_en.close()

        input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
        label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids

        for i in range(len(input_text_ids)):
            all_data.append({
                "data_name": data_name,
                "task_family": "translation (ko->en)",
                'input_ids': input_text_ids[i],
                'output_ids': label_ids[i],
                'input_text': input_texts[i],
                'output_text': label_texts[i],
            })
            
            all_data.append({
                "data_name": data_name,
                "task_family": "translation (en->ko)",
                'input_ids': label_ids[i],
                'output_ids': input_text_ids[i],
                'input_text': label_texts[i],
                'output_text': input_texts[i],
            })
        
    elif data_name == "transliteration":
        file_path = f"{DATA_PATH}/transliteration/data/source/"
        files = [ file_path + f for f in os.listdir(file_path) if os.path.isfile(file_path + f) ]
        input_texts, label_texts = [], []
        for file_name in files:
            file_object = open(file_name, "r")
            for line in file_object:
                if line[0] == '#':
                    continue
                else:
                    line = line.split('\t')
                    input_texts.append(line[1].strip())
                    label_texts.append(line[0].strip())
                    
            input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
            label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids
        
            for i in range(len(input_text_ids)):
                all_data.append({
                    "data_name": data_name,
                    "task_family": "transliteration (ko->en)",
                    'input_ids': input_text_ids[i],
                    'output_ids': label_ids[i],
                    'input_text': input_texts[i],
                    'output_text': label_texts[i],
                })
                    
    elif data_name == "Xpersona": # Too low Quality
        file_object = open(f"{DATA_PATH}/Xpersona/dataset/Ko_persona_train_corrected.json", "rb")
        datasets = json.loads(file_object.read().decode('utf-8'))
        file_object.close()
        # Not Implemented

    elif data_name == "common_sense":
        file_object = open(f"{DATA_PATH}/common_sense/ko_wiki_v1_squad.json", "rb")
        datasets = json.loads(file_object.read().decode('utf-8'))["data"]
        input_texts, label_texts = [], []
        for data in datasets:
            paragraph = data["paragraphs"][0]
            for qa in paragraph["qas"]:
                input_texts.append(f"질의: {qa['question']} 제목: {data['title']} 본문: {paragraph['context']}".strip())
                label_texts.append(qa['answers'][0]['text'].strip())
        file_object.close()
                
        input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
        label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids

        for i in range(len(input_text_ids)):
            all_data.append({
                "data_name": data_name,
                "task_family": "question answering",
                'input_ids': input_text_ids[i],
                'output_ids': label_ids[i],
                'input_text': input_texts[i],
                'output_text': label_texts[i],
            })
            
    elif data_name == "mindslab_mrc":
        file_path = f"{DATA_PATH}/{data_name}/"
        files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
        
        pbar = tqdm(total=len(files))
        for file_name in files:
            file_object = open(file_name, "rb")
            datasets = json.loads(file_object.read().decode('utf-8'))["data"]
            file_object.close()
            input_texts, label_texts = [], []
            for data in datasets:
                paragraph = data["paragraphs"][0]
                for qa in paragraph["qas"]:
                    input_texts.append(f"질의: {qa['question']} 제목: {data['title']} 본문: {paragraph['context']}") # title is not article's title, instead it seems raw file name
                    if 'answers' in qa.keys():
                        label_texts.append(qa['answers'][0]['text'])
                    else:
                        label_texts.append("정답없음")
            
            input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
            label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids

            for i in range(len(input_text_ids)):
                all_data.append({
                    "data_name": data_name,
                    "task_family": "question answering",
                    'input_ids': input_text_ids[i],
                    'output_ids': label_ids[i],
                    'input_text': input_texts[i],
                    'output_text': label_texts[i],
                })
            pbar.update(1)
        pbar.close()
    
    elif data_name == "korean_chat":
        file_path = f"{DATA_PATH}/{data_name}/"
        files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
        # passport is not conversation and filter out the chat_logs by only 1 person
        files.remove(file_path + 'passport.xlsx')
        files.remove(file_path + 'transport.xlsx')
        files.remove(file_path + 'water_supply.xlsx')
        files.remove(file_path + 'car_licence.xlsx')
        
        for data_file in files:
            df = pd.read_excel(data_file, header=0)
            prev_intention, history = "", ""
            speaker_list = []
            for i in range(len(df)):
                row = df.iloc[i].to_list()
                speaker = str(row[0]).strip()
                if speaker == 44:
                    continue
                intention = str(row[7]).strip() # used for split
                if intention != intention:
                    continue
                mention = [ m for m in row[11:15] if m == m ]
                if len(mention): # Filter out a missing value
                    if intention == prev_intention:
                        history += f"{speaker}: {str(mention[0]).strip()}\n" # in order to split history by '\n'
                        speaker_list.append(speaker)
                    else:
                        prev_intention = intention
                        chat_logs = history.strip().split('\n')
                        if len(set(speaker_list)) > 1:
                            chat_logs_tok = tokenizer(chat_logs, add_special_tokens=False).input_ids
                            chat_logs_tok_flatten = []
                            for j in range(1, len(chat_logs_tok)):
                                chat_logs_tok_flatten += chat_logs_tok[j-1]
                                all_data.append({
                                    "data_name": data_name,
                                    "task_family": "dialogue",
                                    'input_ids': chat_logs_tok_flatten,
                                    'output_ids': chat_logs_tok[j],
                                    'input_text': ' '.join(chat_logs[:j]),
                                    'output_text': chat_logs[j],
                                })
                                
                        history = f"{speaker}: {mention[0].strip()}\n"
                        speaker_list = []

    elif data_name == "ai_hub_kor2eng":
        file_path = f"{DATA_PATH}/ai_hub_kor2eng/"
        files = [ file_path + f for f in os.listdir(file_path) ]
        pbar = tqdm(total = len(files))
        for data_file in files:
            df = pd.read_excel(data_file, header=0)
            input_texts, label_texts = [], []
            for i in range(len(df['원문'])):
                input_texts.append(df['원문'][i].strip())
                label_texts.append(df['번역문'][i].strip())
                
            input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
            label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids

            for i in range(len(input_text_ids)):
                all_data.append({
                    "data_name": data_name,
                    "task_family": "translation (ko->en)",
                    'input_ids': input_text_ids[i],
                    'output_ids': label_ids[i],
                    'input_text': input_texts[i],
                    'output_text': label_texts[i],
                })
                
                all_data.append({
                    "data_name": data_name,
                    "task_family": "translation (en->ko)",
                    'input_ids': label_ids[i],
                    'output_ids': input_text_ids[i],
                    'input_text': label_texts[i],
                    'output_text': input_texts[i],
                })
                
            pbar.update(1)
        pbar.close()
    
    elif data_name == "ai_hub_sentiment_conversation": # Should be checked by runs
        file_path = f"{DATA_PATH}/ai_hub_sentiment_conversation/"
        files = [ file_path + f for f in os.listdir(file_path) ]
        for data_file in files:
            df = pd.read_excel(data_file, header=0)
            conv_header = []
            for h in list(df):
                if '사람문장' in h or '시스템응답' in h:
                    conv_header.append(h)
            sentiment = df['감정_대분류']
            convs = df[conv_header]
            pbar = tqdm(total = len(df['감정_대분류']))
            chat_logs, label_texts = [], []
            for i in range(len(df['감정_대분류'])):
                chat_logs.append(convs.iloc[i].to_list())#[::2] + convs.iloc[i].to_list()[1::2]
                label_texts.append(sentiment[i])
                
                for j, c in enumerate(chat_logs):
                    if c == c:
                        if j%2:
                            chat_logs[j] = '시스템: ' + str(c).strip()
                        else:
                            chat_logs[j] = '사람: ' + str(c).strip()
                
            chat_logs_tok = tokenizer(chat_logs, add_special_tokens=False).input_ids
                    
            if task_family == "dialogue":
                chat_logs_tok_flatten = []
                for j in range(1, len(chat_logs_tok)):
                    chat_logs_tok_flatten += chat_logs_tok[j-1]
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "dialogue",
                        'input_ids': chat_logs_tok_flatten[:-1],
                        'output_ids': chat_logs_tok[j][:-1],
                        'input_text': ' '.join(chat_logs[:j]),
                        'output_text': chat_logs[j],
                    })

            if task_family == "classification":
                label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids
                for j in range(1, len(chat_logs_tok)):
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "classification",
                        'input_ids': chat_logs_tok[j],
                        'output_ids': label_ids[j],
                        'input_text': ' '.join(chat_logs),
                        'output_text': label_texts,
                    })
                pbar.update(1)
            pbar.close()

    elif data_name == "ai_hub_kor2eng_expert":
        file_path = f"{DATA_PATH}/ai_hub_kor2eng_expert/"
        files = [ file_path + f for f in os.listdir(file_path) ]
        pbar = tqdm(total = len(files))
        for data_file in files:
            df = pd.read_csv(data_file, header=0)
            input_texts, label_texts = [], []
            for i in range(len(df['한국어'])):
                input_texts.append(df['한국어'][i].strip())
                label_texts.append(df['영어'][i].strip())
                
            input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
            label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids
            
            for i in range(len(input_text_ids)):
                all_data.append({
                    "data_name": data_name,
                    "task_family": "translation (ko->en)",
                    'input_ids': input_text_ids[i],
                    'output_ids': label_ids[i],
                    'input_text': input_texts[i],
                    'output_text': label_texts[i],
                })
                
                all_data.append({
                    "data_name": data_name,
                    "task_family": "translation (en->ko)",
                    'input_ids': label_ids[i],
                    'output_ids': input_text_ids[i],
                    'input_text': label_texts[i],
                    'output_text': input_texts[i],
                })
                
            pbar.update(1)
        pbar.close()
    
    elif data_name == "ai_hub_doc_summ":
        file_path = f"{DATA_PATH}/ai_hub_doc_summ/"
        files = [ file_path + f for f in os.listdir(file_path) ]
        pbar = tqdm(total = len(files))
        for i, file_name in enumerate(files):
            file_object = open(f"{file_name}", "rb")
            datasets = json.loads(file_object.read().decode('utf-8'))["documents"]
            input_texts, label_ext, label_abs = [], [], []
            for data in datasets:
#                 contents = [ d[0]['sentence'] for d in data['text'] ]
                contents = []
                for d in data['text']:
                    for dd in d:
                        contents.append(dd['sentence'].strip())
                input_texts.append(f"제목: {data['title']} 본문: {' '.join(contents)}")
                label_ext.append(' '.join([ contents[i] for i in data['extractive'] if i is not None ]))
                label_abs.append(data['abstractive'][0].strip())
            file_object.close()
            
            input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
            label_ext_ids = tokenizer(label_ext, add_special_tokens=False).input_ids
            label_abs_ids = tokenizer(label_abs, add_special_tokens=False).input_ids
            
            for j in range(len(input_text_ids)):
                all_data.append({
                    "data_name": data_name,
                    "task_family": "extractive summarization",
                    'input_ids': input_text_ids[j],
                    'output_ids': label_ext_ids[j],
                    'input_text': input_texts[j],
                    'output_text': label_ext[j],
                })
                    
                all_data.append({
                    "data_name": data_name,
                    "task_family": "abstractive summarization",
                    'input_ids': input_text_ids[j],
                    'output_ids': label_abs_ids[j],
                    'input_text': input_texts[j],
                    'output_text': label_abs[j],
                })
            pbar.update(1)
        pbar.close()

    elif data_name == "ai_hub_thesis_summ":
        for data_split in ["train", "valid"]:
            file_path = f"{DATA_PATH}/ai_hub_thesis_summ/{data_split}/"
            files = [ file_path + f for f in os.listdir(file_path) ]
            for i, file_name in enumerate(files):
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))['data']
                pbar = tqdm(total = len(datasets))
                input_texts, label_abs, label_ext = [], [], []
                for data in datasets:
                    if f"논문요약" in file_name:
                        input_texts.append(data["summary_entire"][0]['orginal_text'].strip())
                        label_abs.append(data["summary_entire"][0]['summary_text'].strip())
                        
                    input_texts.append(data["summary_section"][0]['orginal_text'].strip())
                    label_ext.append(data["summary_section"][0]['summary_text'].strip())
                file_object.close()
                
                input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
                label_ext_ids = tokenizer(label_ext, add_special_tokens=False).input_ids
                label_abs_ids = tokenizer(label_abs, add_special_tokens=False).input_ids
                
                for j in range(len(input_text_ids)):
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "abstractive summarization",
                        'input_ids': input_text_ids[j],
                        'output_ids': label_abs_ids[j],
                        'input_text': input_texts[j],
                        'output_text': label_abs[j],
                    })

                    all_data.append({
                        "data_name": data_name,
                        "task_family": "extractive summarization",
                        'input_ids': input_text_ids[j],
                        'output_ids': label_ext_ids[j],
                        'input_text': input_texts[j],
                        'output_text': label_ext[j],
                    })
                        
                    pbar.update(1)
                pbar.close()
                
    elif data_name == "ai_hub_book_mrc":
            file_path = f"{DATA_PATH}/ai_hub_book_mrc/"
            files = [ file_path + f for f in os.listdir(file_path) ]
            pbar = tqdm(total = len(files))
            for data_file in files:
                file_object = open(f"{data_file}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))["data"]
                input_texts, label_texts = [], []
                for data in datasets:
                    paragraph = data["paragraphs"][0]
                    for qa in paragraph["qas"]:
                        input_texts.append(f"질의: {qa['question'].strip()} 제목: {data['title'].strip()} 본문: {paragraph['context'].strip()}")
                        label_texts.append('정답없음' if qa['is_impossible'] else qa['answers'][0]['text'])

                input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
                label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids

                for j in range(len(input_text_ids)):
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "question answering",
                        'input_ids': input_text_ids[j],
                        'output_ids': label_ids[j],
                        'input_text': input_texts[j],
                        'output_text': label_texts[j],
                    })
                pbar.update(1)
            pbar.close()

    elif data_name == "ai_hub_book_summ":
        file_path = f"{DATA_PATH}/{data_name}/"
        files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
        pbar = tqdm(total = len(files))
        input_texts, label_texts = [], []
        for file_name in files: # 1 data per 1 file
            file_object = open(f"{file_name}", "rb")
            datasets = json.loads(file_object.read().decode('utf-8'))
            title = datasets['doc_name'].strip() if "doc_name" in datasets.keys() else ''
            input_texts.append(f"제목: {title} 본문: {datasets['passage'].strip()}")
            label_texts.append(datasets['summary'].strip())
            file_object.close()
            pbar.update(1)
        pbar.close()
                
        input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
        label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids
                
        for j in range(len(input_text_ids)):         
            all_data.append({
                "data_name": data_name,
                "task_family": "abstractive summarization",
                'input_ids': input_text_ids[j],
                'output_ids': label_ids[j],
                'input_text': input_texts[j],
                'output_text': label_texts[j],
            })

    elif data_name == "ai_hub_callcenter_dialogue":
            file_path = f"{DATA_PATH}/{data_name}/"
            files = [ file_path + f for f in os.listdir(file_path) ]
            pbar = tqdm(total = len(files))
            for i, file_name in enumerate(files):
                try:
                    file_object = open(f"{file_name}", "rb")
                    datasets = json.loads(file_object.read().decode('utf-8'))
                except:
                    traceback.print_exc()
                    pbar.update(1)
                    continue
                file_object.close()
                
                chat_id = datasets[0]['대화셋일련번호']
                chat_logs = []
                chats = ""
                for data in datasets:
                    if data['대화셋일련번호'] != chat_id:
                        chats = chats.strip()
                        chat_logs.append(chats)
                        chats = ""

                    chat_id = data['대화셋일련번호']
                    speaker = data['화자']
                    any_chats = data['고객질문(요청)'] + data['상담사질문(요청)'] + data['고객답변'] + data['상담사답변']
                    chats += f"{speaker}: {any_chats} "
                # Add the last data
                chats = chats.strip()
                chat_logs.append(chats)
                                    
                chat_logs_tok = tokenizer(chat_logs, add_special_tokens=False).input_ids

                chat_logs_tok_flatten = []
                for j in range(1, len(chat_logs_tok)):
                    chat_logs_tok_flatten += chat_logs_tok[j-1]
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "dialogue",
                        'input_ids': chat_logs_tok_flatten[:-1],
                        'output_ids': chat_logs_tok[j][:-1],
                        'input_text': ' '.join(chat_logs[:j]),
                        'output_text': chat_logs[j],
                    })
                pbar.update(1)
            pbar.close()
    
    elif data_name == "ai_hub_ordering_dialogue":
            file_path = f"{DATA_PATH}/{data_name}/"
            files = [ file_path + f for f in os.listdir(file_path) ]
            pbar = tqdm(total = len(files))
            for i, file_name in enumerate(files):
                df = pd.read_csv(f"{file_name}", header=0)
                chat_id = df['QA번호'][0]
                chat_logs = []
                chats = ""
                for i in range(len(df)):
                    if df['QA번호'][i] != chat_id:
                        chat_logs.append(chats)
                        chats = ""
                        
                    chat_id = df['QA번호'][i]
                    speaker = "고객" if df['발화자'][i] == 'c' else "점원"
                    chat = df['발화문'][i]
                    chats += f"{speaker}: {chats}"
                
                # Add the last data
                chat_logs.append(chats)
                chats = ""
                
                chat_logs_tok = tokenizer(chat_logs, add_special_tokens=False).input_ids

                chat_logs_tok_flatten = []
                for j in range(1, len(chat_logs_tok)):
                    chat_logs_tok_flatten += chat_logs_tok[j-1]
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "dialogue",
                        'input_ids': chat_logs_tok_flatten[:-1],
                        'output_ids': chat_logs_tok[j][:-1],
                        'input_text': '\n'.join(chat_logs[:j]),
                        'output_text': chat_logs[j],
                    })

                pbar.update(1)
            pbar.close()
            
    elif data_name == "ai_hub_koreansns_dialogue":
        for data_split in ["train", "validation"]:
            file_path = f"{DATA_PATH}/{data_name}/{data_split}/"
            files = [ file_path + f for f in os.listdir(file_path) ]
            pbar = tqdm(total = len(files))
            for i, file_name in enumerate(files):
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))["data"]
                file_object.close()
                for data in datasets:
                    chat_logs = []
                    participants = data['header']['participantsInfo']
                    participants = { participants[i]['participantID']: participants[i]['age'] + participants[i]['gender'] for i in range(len(participants)) }
                    for d in data['body']:
                        chat_logs.append(f"{participants[d['participantID']]}: {d['utterance']}")
                
                    chat_logs_tok = tokenizer(chat_logs, add_special_tokens=False).input_ids

                    chat_logs_tok_flatten = []
                    for j in range(1, len(chat_logs_tok)):
                        chat_logs_tok_flatten += chat_logs_tok[j-1]
                        all_data.append({
                            "data_name": data_name,
                            "task_family": "dialogue",
                            'input_ids': chat_logs_tok_flatten[:-1],
                            'output_ids': chat_logs_tok[j][:-1],
                            'input_text': ' '.join(chat_logs[:j]),
                            'output_text': chat_logs[j],
                        })
                    
                pbar.update(1)
            pbar.close()
                
    elif data_name == "ai_hub_conversation_summ":
        for data_split in ["train", "validation"]:
            file_path = f"{DATA_PATH}/{data_name}/{data_split}/"
            files = [ file_path + f for f in os.listdir(file_path) ]
            pbar = tqdm(total = len(files))
            for i, file_name in enumerate(files):
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))["data"]
                file_object.close()

                for data in datasets:
                    chat_logs, summary = [], []
                    participants = data['header']['participantsInfo']
                    participants = { participants[i]['participantID']: participants[i]['age'] + participants[i]['gender'] for i in range(len(participants)) }
                    
                    for d in data['body']['dialogue']:
                        chat_logs.append(f"{participants[d['participantID']]}: {d['utterance']}")
                    if task_family == "summarization":
                        summary.append(data['body']['summary'])
                
                    chat_logs_tok = tokenizer(chat_logs, add_special_tokens=False).input_ids

                    if task_family == "dialogue":
                        chat_logs_tok_flatten = []
                        for j in range(1, len(chat_logs_tok)):
                            chat_logs_tok_flatten += chat_logs_tok[j-1]
                            all_data.append({
                                "data_name": data_name,
                                "task_family": "dialogue",
                                'input_ids': chat_logs_tok_flatten[:-1],
                                'output_ids': chat_logs_tok[j][:-1],
                                'input_text': ' '.join(chat_logs[:j]),
                                'output_text': chat_logs[j],
                            })
                
                    if task_family == "summarization" or "summarization" in task_family:
                        summary_tok = tokenizer(summary, add_special_tokens=False).input_ids
                        chat_logs_tok_flatten = []
                        for j in range(1, len(chat_logs_tok)):
                            chat_logs_tok_flatten += chat_logs_tok[j-1]

                        all_data.append({
                            "data_name": data_name,
                            "task_family": "abstractive summarization",
                            'input_ids': chat_logs_tok_flatten,
                            'output_ids': summary_tok,
                            'input_text': ' '.join(chat_logs),
                            'output_text': data['body']['summary'],
                        })
                pbar.update(1)
            pbar.close()                    
                
    elif data_name == "ai_hub_kor2eng_technology"\
            or data_name == "ai_hub_kor2eng_socialscience":
        file_path = f"{DATA_PATH}/{data_name}/"
        files = [ file_path + f for f in os.listdir(file_path) ]
        for data_file in files:
            df = pd.read_csv(data_file, header=0)
            pbar = tqdm(total = len(df['ko']))
            input_texts, label_texts = [], []
            for i in range(len(df['ko'])):
                input_texts.append(df['ko'][i].strip())
                label_texts.append(df['en'][i].strip())
                
            input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
            label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids

            for j in range(len(input_text_ids)):
                all_data.append({
                    "data_name": data_name,
                    "task_family": "translation (ko->en)",
                    'input_ids': input_text_ids[j],
                    'output_ids': label_ids[j],
                    'input_text': input_texts[j],
                    'output_text': label_texts[j],
                })
                
                all_data.append({
                    "data_name": data_name,
                    "task_family": "translation (en->ko)",
                    'input_ids': label_ids[j],
                    'output_ids': input_text_ids[j],
                    'input_text': label_texts[j],
                    'output_text': input_texts[j],
                })

                pbar.update(1)
            pbar.close()
            
    elif data_name == "ai_hub_kor2jpn"\
        or data_name == "ai_hub_kor2chn_technology"\
        or data_name == "ai_hub_kor2chn_socialscience":
        file_path = f"{DATA_PATH}/{data_name}/"
        files = [ file_path + f for f in os.listdir(file_path) ]
        target_lang = '일본어' if data_name[-3:] == 'jpn' else '중국어'
        target_code = 'jp' if data_name[-3:] == 'jpn' else 'cn'
        pbar = tqdm(total = len(files))
        for data_file in files:
            df = pd.read_csv(data_file, header=0)
            input_texts, label_texts = [], []
            for i in range(len(df['한국어'])):
                input_texts.append(df['한국어'][i].strip())
                label_texts.append(df[target_lang][i].strip())
                
            input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
            label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids
            
            for j in range(len(input_text_ids)):
                all_data.append({
                    "data_name": data_name,
                    "task_family": f"translation (ko->{target_code})",
                    'input_ids': input_text_ids[j],
                    'output_ids': label_ids[j],
                    'input_text': input_texts[j],
                    'output_text': label_texts[j],
                })
                
                all_data.append({
                    "data_name": data_name,
                    "task_family": f"translation ({target_code}->ko)",
                    'input_ids': label_ids[j],
                    'output_ids': input_text_ids[j],
                    'input_text': label_texts[j],
                    'output_text': input_texts[j],
                })
                
            pbar.update(1)
        pbar.close()
    
    elif data_name == "ai_hub_command":
        file_path = f"{DATA_PATH}/{data_name}/"
        files = [ file_path + f for f in os.listdir(file_path) ]
        for data_file in files:
            df = pd.read_excel(data_file, header=0)
            pbar = tqdm(total = len(df['문장']))
            input_texts, label_texts = [], []
            for i in range(len(df['문장'])):
                input_texts.append(df['문장'][i].strip())
                label_texts.append(df["의도 (Intention)"][i].strip())
                
            input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
            label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids
            
            for j in range(len(input_text_ids)):
                all_data.append({
                    "data_name": data_name,
                    "task_family": "classification",
                    'input_ids': input_text_ids[j],
                    'output_ids': label_ids[j],
                    'input_text': input_texts[j],
                    'output_text': label_texts[j],
                })
                pbar.update(1)
            pbar.close()
            
    elif data_name == "ai_hub_broadcasting_conversation" or data_name == "ai_hub_domain_conversation":
        for data_split in ['train', 'validation']:
            file_path = f"{DATA_PATH}/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))
                file_object.close()
                participants = datasets['speaker']
                participants = { participants[i]['id']: participants[i]['age'] + ' ' + participants[i]['role'] + ' ' + participants[i]['sex'] for i in range(len(participants)) }
                
                chat_logs = []
                for d in datasets['utterance']:
                    if d['speaker_id'] == '?': # Remove
                        continue
                    chat_logs.append(f"{participants[d['speaker_id']]}: {d['original_form']}")
                
                chat_logs_tok = tokenizer(chat_logs, add_special_tokens=False).input_ids
                
                chat_logs_tok_flatten = []
                for j in range(1, len(chat_logs_tok), 10): # Could be out-of-Memory
                    chat_logs_tok_flatten += chat_logs_tok[j-1]
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "dialogue",
                        'input_ids': chat_logs_tok_flatten[:-1],
                        'output_ids': chat_logs_tok[j][:-1],
                        'input_text': ' '.join(chat_logs[:j]),
                        'output_text': chat_logs[j],
                    })
                pbar.update(1)
            pbar.close()
            
    elif data_name == "ai_hub_casual_domain_conversation" or data_name == "ai_hub_goal_oriented_dialogue":
        for data_split in ['train', 'validation']:
            file_path = f"{DATA_PATH}/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                try:
                    datasets = json.loads(file_object.read().decode('utf-8'))["info"]
                except json.decoder.JSONDecodeError: # Filter out invalid data
                    traceback.print_exc()
                    pbar.update(1)
                    continue
                file_object.close()
                    
                chat_logs = []
                chat_by_line = datasets[0]["annotations"]["lines"]
                for c in chat_by_line:
                    norm_text = c["norm_text"].replace('\xa0', ' ')
                    if len(norm_text) > 1:
                        if norm_text[1] == '.':
                            norm_text = norm_text[2:]
                    chat_logs.append(c["speaker"]["age"] + c["speaker"]["sex"] + ': ' + norm_text)

                chat_logs_tok = tokenizer(chat_logs, add_special_tokens=False).input_ids
                
                chat_logs_tok_flatten = []
                for j in range(1, len(chat_logs_tok)):
                    chat_logs_tok_flatten += chat_logs_tok[j-1]
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "dialogue",
                        'input_ids': chat_logs_tok_flatten[:-1],
                        'output_ids': chat_logs_tok[j][:-1],
                        'input_text': ' '.join(chat_logs[:j]),
                        'output_text': chat_logs[j],
                    })
                pbar.update(1)
            pbar.close()
    
    elif data_name == "ai_hub_essay_evaluation":
        for data_split in ['train', 'validation']:
            file_path = f"{DATA_PATH}/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            input_texts, label_texts = [], []
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))
                file_object.close()
                
                ### cleaning
                text = [ p['paragraph_txt'].strip() for p in datasets['paragraph'] ]
                text = '\n'.join(text)
                text = text.replace('#@문장구분#', '\n').replace(' .\n', '.\n').replace('.\n ', '.\n').replace('\n\n', '\n')
                text = text.replace(' \n', '\n').replace('\n ', '\n').strip()
                input_texts.append(text)
                label_texts.append(str(np.round(datasets['score']['essay_scoreT_avg'], 1)))
                pbar.update(1)
                #####
                
            input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
            label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids
            
            for j in range(len(input_text_ids)):
                all_data.append({
                    "data_name": data_name,
                    "task_family": "classification",
                    'input_ids': input_texts[j],
                    'output_ids': label_ids[j],
                    'input_text': input_texts[j],
                    'output_text': label_texts[j],
                })
            pbar.close()
    
    elif data_name == "ai_hub_casual_kor2chn2jpn_corpus":
        for data_split in ['train', 'validation']:
            file_path = f"{DATA_PATH}/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                input_texts, label_texts = [], []
                file_object = open(f"{file_name}", "rb")
                try:
                    datasets = json.loads(file_object.read().decode('utf-8'))
                except UnicodeDecodeError: # Filter out invalid data
                    traceback.print_exc()
                    pbar.update(1)
                    continue
                file_object.close()
                
                task_family = 'translation '
                if datasets[0]['S_Code'][-2:] == 'JP':
                    task_family += "(jp->"
                elif datasets[0]['S_Code'][-2:] == 'KR':
                    task_family += "(ko->"
                elif datasets[0]['S_Code'][-2:] == 'CN':
                    task_family += "(cn->"
                if datasets[0]['T_Code'][-2:] == 'JP':
                    task_family += "jp)"
                elif datasets[0]['T_Code'][-2:] == 'KR':
                    task_family += "ko)"
                elif datasets[0]['T_Code'][-2:] == 'CN':
                    task_family += "cn)"
                    
                for data in datasets:
                    input_texts.append(data['원문'])
                    label_texts.append(data['최종번역문'])
                    
                input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
                label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids
                
                for j in range(len(input_text_ids)):
                    all_data.append({
                        "data_name": data_name,
                        "task_family": task_family,
                        'input_ids': input_text_ids[j],
                        'output_ids': label_texts[j],
                        'input_text': input_texts[j],
                        'output_text': label_texts[j],
                    })
                pbar.update(1)
            pbar.close()
    
    elif data_name == "ai_hub_ethical_text":
        for data_split in ['train', 'validation']:
            file_path = f"{DATA_PATH}/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))
                file_object.close()
                for data in datasets:
                    dialogue_history = []
                    dialogue_history_label = []
                    for turn in data['sentences']:
                        dialogue_history.append(f"화자{turn['speaker']}: {turn['text']}")
                        dialogue_history_label.append('비윤리적' if turn['is_immoral'] else '윤리적') # Type is so ambiguous
                
                    dialogue_history_tok = tokenizer(dialogue_history, add_special_tokens=False).input_ids
                    dialogue_history_label_tok = tokenizer(dialogue_history_label, add_special_tokens=False).input_ids
                
                    for j in range(1, len(dialogue_history_tok)): # Could be OOM
                        all_data.append({
                            "data_name": data_name,
                            "task_family": "classification",
                            'input_ids': dialogue_history_tok[:j],
                            'output_ids': dialogue_history_label_tok[j],
                            'input_text': ' '.join(dialogue_history[:j]),
                            'output_text': dialogue_history_label[j],
                        })
                pbar.update(1)
            pbar.close()
    
    elif data_name == "ai_hub_patent_eng2kor":
        for data_split in ['train', 'validation']:
            file_path = f"{DATA_PATH}/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))["labeled_data"]
                file_object.close()
                input_texts, label_texts = [], []
                for data in datasets:
                    input_texts.append(f"title: {data['invention_title_eng']}\nabstract: {data['astrt_cont_eng']}\nclaim: {data['claim_eng']}")
                    label_texts.append(f"제목: {data['invention_title_kor']}\n초록: {data['astrt_cont_kor']}\n청구항: {data['claim_kor']}")
                
                input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
                label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids
                
                for j in range(len(input_text_ids)):
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "translation (en->ko)",
                        'input_ids': input_text_ids[j],
                        'output_ids': label_ids,
                        'input_text': input_texts[j],
                        'output_text': label_texts[j],
                    })
                    
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "translation (ko->en)",
                        'input_ids': label_ids[j],
                        'output_ids': input_text_ids[j],
                        'input_text': label_texts[j],
                        'output_text': input_texts[j],
                    })
                pbar.update(1)
            pbar.close()
            
    elif data_name == "ai_hub_admin_document_mrc" or data_name == "ai_hub_newsarticle_mrc":
        for data_split in ['train', 'validation']:
            file_path = f"{DATA_PATH}/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                try:
                    datasets = json.loads(file_object.read().decode('utf-8'))['data']
                except json.decoder.JSONDecodeError: # Filter out invalid data
                    traceback.print_exc()
                    pbar.update(1)
                    continue
                file_object.close()
                input_texts, label_texts = [], []
                for data in datasets:
                    for qa in data['paragraphs'][0]['qas']:
                        input_texts.append(f"질의: {qa['question']} 제목: {data['doc_title']} 본문: {data['paragraphs'][0]['context']}")
                        label_texts.append("" if qa['is_impossible'] else qa['answers']['text'])

                input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
                label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids

                for j in range(len(input_text_ids)):
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "question answering",
                        'input_ids': input_text_ids[j],
                        'output_ids': label_ids[j],
                        'input_text': input_texts[j],
                        'output_text': label_texts[j],
                    })
                pbar.update(1)
            pbar.close()
            
    elif data_name == "ai_hub_lowquality_stt_dialogue":
        for data_split in ['train', 'validation']:
            file_path = f"{DATA_PATH}/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))['dataSet']
                file_object.close()
                if len(datasets['dialogs']) == 0:
                    pbar.update(1)
                    continue
                participants = datasets['typeInfo']['speakers']
                participants = { participants[i]['id'] : f"{participants[i]['age']} {participants[i]['gender']}자 {participants[i]['type'][:-1]}" for i in range(len(participants)) }
                
                chat_logs = []
                for data in datasets['dialogs']:
                    chat_logs.append(f"{participants[data['speaker']]}: {data['text']}")
                chat_logs_tok = tokenizer(chat_logs, add_special_tokens=False).input_ids
                    
                for j in range(1, len(chat_logs_tok), 2):
                    chat_logs_tok_flatten = [c for sublist in chat_logs_tok[:j] for c in sublist]
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "dialogue",
                        'input_ids': chat_logs_tok_flatten,
                        'output_ids': chat_logs_tok[j],
                        'input_text': ' '.join(chat_logs[:j]),
                        'output_text': chat_logs[j],
                    })
                pbar.update(1)
            pbar.close()

    elif data_name == "ai_hub_summary_report_generation":
        for data_split in ['train', 'validation']:
            file_path = f"{DATA_PATH}/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            input_texts, summary1, summary2, summary3 = [], [], [], []
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))
                file_object.close()
                
                input_texts.append(f"제목: {datasets['Meta(Acqusition)']['doc_name']}\n본문: {datasets['Meta(Refine)']['passage']}")
                for k, v in datasets['Annotation'].items():
                    # summary1 is abstractive
                    if k == "summary1":
                        summary1.append(v.strip() if v != "null" and v is not None else "")
                    elif k == "summary2":
                        summary2.append(v.strip() if v != "null" and v is not None else "")
                    elif k == "summary3":
                        summary3.append(v.strip() if v != "null" and v is not None else "")
                
            input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
            summary1_ids = tokenizer(summary1, add_special_tokens=False).input_ids
            summary2_ids = tokenizer(summary2, add_special_tokens=False).input_ids
            summary3_ids = tokenizer(summary3, add_special_tokens=False).input_ids

            for j in range(len(input_text_ids)):
                if len(summary1_ids[j]):
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "abstractive summarization",
                        'input_ids': input_text_ids[j],
                        'output_ids': summary1_ids[j],
                        'input_text': input_texts[j],
                        'output_text': summary1[j],
                    })

                if len(summary2_ids[j]):
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "abstractive summarization",
                        'input_ids': input_text_ids[j],
                        'output_ids': summary2_ids[j],
                        'input_text': input_texts[j],
                        'output_text': summary2[j],
                    })

                if len(summary3_ids[j]):
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "abstractive summarization",
                        'input_ids': input_text_ids[j],
                        'output_ids': summary3_ids[j],
                        'input_text': input_texts[j],
                        'output_text': summary3[j],
                    })
                    
                pbar.update(1)
            pbar.close()
            
    elif data_name == "ai_hub_script_summary":
        for data_split in ['train', 'validation']:
            file_path = f"{DATA_PATH}/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            input_texts, summary1, summary2, summary3 = [], [], [], []
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))
                file_object.close()
                
                input_texts.append(f"제목: {datasets['Meta']['doc_name']}\n본문: {datasets['Meta']['passage']}")
                for k, v in datasets['Annotation'].items():
                    # summary1 is abstractive
                    if k == "Summary1":
                        summary1.append(v.strip() if v != "null" and v is not None else "")
                    elif k == "Summary2":
                        summary2.append(v.strip() if v != "null" and v is not None else "")
                    elif k == "Summary3":
                        summary3.append(v.strip() if v != "null" and v is not None else "")
                pbar.update(1)
            pbar.close()
                
            input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
            summary1_ids = tokenizer(summary1, add_special_tokens=False).input_ids
            summary2_ids = tokenizer(summary2, add_special_tokens=False).input_ids
            summary3_ids = tokenizer(summary3, add_special_tokens=False).input_ids

            pbar = tqdm(total = len(input_text_ids))
            for j in range(len(input_text_ids)):
                if len(summary1_ids[j]):
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "abstractive summarization",
                        'input_ids': input_text_ids[j],
                        'output_ids': summary1_ids[j],
                        'input_text': input_texts[j],
                        'output_text': summary1[j],
                    })

                if len(summary2_ids[j]):
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "abstractive summarization",
                        'input_ids': input_text_ids[j],
                        'output_ids': summary2_ids[j],
                        'input_text': input_texts[j],
                        'output_text': summary2[j],
                    })

                if len(summary3_ids[j]):
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "abstractive summarization",
                        'input_ids': input_text_ids[j],
                        'output_ids': summary3_ids[j],
                        'input_text': input_texts[j],
                        'output_text': summary3[j],
                    })
                pbar.update(1)
            pbar.close()
    
    elif data_name == "ai_hub_multilingual_speaking_translation":
        for data_split in ['train', 'validation']:
            file_path = f"{DATA_PATH}/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))
                file_object.close()
                task_family = 'translation '
                if datasets[0]['S_Code'][-2:] == 'KR':
                    task_family += "(ko->"
                elif datasets[0]['S_Code'][-2:] == 'DE':
                    task_family += "(de->"
                elif datasets[0]['S_Code'][-2:] == 'ES':
                    task_family += "(es->"
                elif datasets[0]['S_Code'][-2:] == 'FR':
                    task_family += "(fr->"
                else:
                    print(datasets[0]['S_Code'][-2:]) # Nothing                  
                    return
                if datasets[0]['T_Code'][-2:] == 'KR':
                    task_family += "ko)"
                elif datasets[0]['T_Code'][-2:] == 'DE':
                    task_family += "de)"
                elif datasets[0]['T_Code'][-2:] == 'ES':
                    task_family += "es)"
                elif datasets[0]['T_Code'][-2:] == 'FR':
                    task_family += "fr)"
                else:
                    print(datasets[0]['T_Code'][-2:]) # Nothing
                    return
                    
                input_texts, label_texts = [], []
                for data in datasets:
                    if label_texts == 0.0: # Filter out
                        continue
                    input_texts.append(str(data['원문']).strip())
                    label_texts.append(str(data['최종번역문']).strip())
                        
                input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
                label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids
                
                for j in range(len(input_text_ids)):
                    all_data.append({
                        "data_name": data_name,
                        "task_family": task_family,
                        'input_ids': input_text_ids[j],
                        'output_ids': label_ids[j],
                        'input_text': input_texts[j],
                        'output_text': label_texts[j],
                    })
                pbar.update(1)
            pbar.close()
                
    elif data_name == "ai_hub_complaint_automation":
        for data_split in ['train', 'validation']:
            file_path = f"{DATA_PATH}/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))['documents']
                file_object.close()
                input_texts, label_texts = [], []
                for data in datasets:
                    input_texts.append(data['Q_refined'].strip())
                    label_texts.append(data['labeling']['intent']['category'].strip())
                    
                input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
                label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids
                
                for j in range(len(input_text_ids)):
                    all_data.append({
                        "data_name": data_name,
                        "task_family": 'classification',
                        'input_ids': input_text_ids[j],
                        'output_ids': label_ids[j],
                        'input_text': input_texts[j],
                        'output_text': label_texts[j],
                    })
                pbar.update(1)
            pbar.close()
    
    elif data_name == "ai_hub_food_translation_corpus" or\
        data_name == "ai_hub_broadcasting_translation_corpus" or\
        data_name == "ai_hub_casualtalk_translation" or\
        data_name == "ai_hub_tech_translation_corpus":
        for data_split in ['train', 'validation']:
            file_path = f"{DATA_PATH}/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            input_texts, label_texts = [], []
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))['data']
                file_object.close()
                source_lang = datasets[0]['source_language']
                target_lang = datasets[0]['target_language']
                
                if source_lang != 'ko':
                    for data in datasets:
                        input_texts.append(data[f'{source_lang}_original'].strip())
                        label_texts.append(data['ko'].strip())
                
                else:
                    for data in datasets:
                        input_texts.append(data[f'{source_lang}_original'].strip())
                        label_texts.append(data[target_lang].strip())
                        
                input_text_ids = tokenizer(input_texts, add_special_tokens=False).input_ids
                label_ids = tokenizer(label_texts, add_special_tokens=False).input_ids

                for j in range(len(input_text_ids)):
                    all_data.append({
                        "data_name": data_name,
                        "task_family": f'translation ({source_lang}->{target_lang})',
                        'input_ids': input_text_ids[j],
                        'output_ids': label_ids[j],
                        'input_text': input_texts[j],
                        'output_text': label_texts[j],
                    })
                pbar.update(1)
            pbar.close()

                        
    else:
        print("out of predefined dataset", data_name)
        
    print(all_data[-5:])
    
    return all_data

def main(data_name=None, task_family=None):
    dataset = []
    if task_family is None:
        dataset += preprocess_dataset(data_name) # 1 data
    
    else:
        if "classification" in task_family or "all" in task_family: # total 9
            dataset += preprocess_dataset('3i4k')
            dataset += preprocess_dataset('nsmc')
            dataset += preprocess_dataset('toxic_comment')
            dataset += preprocess_dataset('korean_hate_speech')
            dataset += preprocess_dataset('ai_hub_sentiment_conversation', "classification")
            dataset += preprocess_dataset('ai_hub_command')
            dataset += preprocess_dataset('ai_hub_essay_evaluation')
            dataset += preprocess_dataset('ai_hub_ethical_text')
            dataset += preprocess_dataset('ai_hub_complaint_automation')

        if "natural language inference" in task_family or "all" in task_family: # total 1
            dataset += preprocess_dataset('kornli')

        if "semantic textual similarity" in task_family or "all" in task_family: # total 3
            dataset += preprocess_dataset('korsts')
            dataset += preprocess_dataset('question_pair')
            dataset += preprocess_dataset('paraKQC')

        if "question answering" in task_family or "all" in task_family: # total 6
            dataset += preprocess_dataset('korquad_v1')
#             dataset += preprocess_dataset('korquad_v2') # Not use
            dataset += preprocess_dataset('common_sense')
            dataset += preprocess_dataset('mindslab_mrc')
            dataset += preprocess_dataset('ai_hub_book_mrc')
            dataset += preprocess_dataset('ai_hub_admin_document_mrc')
            dataset += preprocess_dataset('ai_hub_newsarticle_mrc')

        if "summarization" in task_family or "all" in task_family: # total 8
            dataset += preprocess_dataset("sci-news-sum-kr-50")
            dataset += preprocess_dataset("sae4k")
            dataset += preprocess_dataset("ai_hub_doc_summ")
            dataset += preprocess_dataset("ai_hub_thesis_summ")
            dataset += preprocess_dataset("ai_hub_book_summ")
            dataset += preprocess_dataset("ai_hub_conversation_summ", 'summarization')
            dataset += preprocess_dataset("ai_hub_summary_report_generation")
            dataset += preprocess_dataset("ai_hub_script_summary")

        if "translation" in task_family or "all" in task_family: # 13 here +1 at below
            dataset += preprocess_dataset("korean_parallel")
            dataset += preprocess_dataset("ai_hub_kor2eng")
            dataset += preprocess_dataset("ai_hub_kor2eng_expert")
            dataset += preprocess_dataset("ai_hub_kor2eng_socialscience")
            dataset += preprocess_dataset("ai_hub_kor2eng_technology")
            dataset += preprocess_dataset("ai_hub_kor2jpn")
            dataset += preprocess_dataset("ai_hub_patent_eng2kor")
            dataset += preprocess_dataset("ai_hub_multilingual_speaking_translation")
            dataset += preprocess_dataset("ai_hub_casual_kor2chn2jpn_corpus")
            dataset += preprocess_dataset("ai_hub_food_translation_corpus")
            dataset += preprocess_dataset("ai_hub_broadcasting_translation_corpus")
            dataset += preprocess_dataset("ai_hub_casualtalk_translation")
            dataset += preprocess_dataset("ai_hub_tech_translation_corpus")
            dataset += preprocess_dataset("ai_hub_kor2chn_technology")
            dataset += preprocess_dataset("ai_hub_kor2chn_socialscience")

        if "transliteration" in task_family or "all" in task_family: # +1 at translation
            dataset += preprocess_dataset("transliteration")

        if "dialogue" in task_family or "all" in task_family: # total 11
            dataset += preprocess_dataset("korean_chat")
            dataset += preprocess_dataset('ai_hub_sentiment_conversation', 'dialogue')
            dataset += preprocess_dataset("ai_hub_callcenter_dialogue")
            dataset += preprocess_dataset("ai_hub_ordering_dialogue")
            dataset += preprocess_dataset("ai_hub_koreansns_dialogue")
            dataset += preprocess_dataset("ai_hub_conversation_summ", 'dialogue')
            dataset += preprocess_dataset("ai_hub_broadcasting_conversation")
            dataset += preprocess_dataset("ai_hub_domain_conversation")
            dataset += preprocess_dataset("ai_hub_casual_domain_conversation")
            dataset += preprocess_dataset("ai_hub_goal_oriented_dialogue")
            dataset += preprocess_dataset("ai_hub_lowquality_stt_dialogue")

#     with open('./processed/train_data_all.pkl', 'wb') as f:
#         pickle.dump(dataset, f)
        
if __name__ == '__main__':
    fire.Fire(main)
    # to use: python data_preprocessor.py --data_name="{DATA_NAME}" --task_family="{TASK_NAME}"
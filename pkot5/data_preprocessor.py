import copy
import re
from pprint import pprint
from typing import Optional

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
import json
import random
import pickle
import os

from transformers import PreTrainedTokenizer, AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("paust/pko-t5-base")

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
        data_object = open("./raw/3i4k/data/train_val_test/fci_train_val.txt", 'r')
        label_to_text = {
            '0': 'fragment',
            '1': 'statement',
            '2': 'question',
            '3': 'command',
            '4': 'rhetorical question',
            '5': 'rhetorical command',
            '6': 'intonation-depedent utterance',
        }
        
        for line in data_object:
            line = line.split('\t')
            label = line[0].strip()
            input_text = line[1].strip()
            
            all_data.append({
                "data_name": data_name,
                "task_family": "classification",
                'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                'output_ids': tokenizer.encode(label_to_text[label], add_special_tokens=False),
                'input_text': input_text,
                'output_text': label_to_text[label],
            })
        data_object.close()
            
    elif data_name == 'nsmc':
        datasets = load_dataset("nsmc")
        label_to_text = {
            0: 'negative',
            1: 'positive',
        }
        
#         for data_split in ['train', 'test']:
        for data_split in ['train']:
            for data in datasets[data_split]:
                input_text = data['document'].strip()
                label = data['label']
                all_data.append({
                    "data_name": data_name,
                    "task_family": "classification",
                    'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'output_ids': tokenizer.encode(label_to_text[label], add_special_tokens=False),
                    'input_text': input_text,
                    'output_text': label_to_text[label],
                })
                
    elif data_name == 'toxic_comment':
        df = pd.read_csv("./raw/Toxic_comment_data/ko_train_label.csv", header=0)
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            for k, v in row.items():
                if v == 1:
                    label = k
            input_text = row['document'].strip()
            all_data.append({
                    "data_name": data_name,
                    "task_family": "classification",
                    'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'output_ids': tokenizer.encode(label, add_special_tokens=False),
                    'input_text': input_text,
                    'output_text': label,
            })
            
    elif data_name == 'korean_hate_speech':
        df = pd.read_csv("./raw/korean_hate_speech/labeled/train.tsv", sep='\t', header=0)
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            input_text = row['comments'].strip()
            for label_type in ['contain_gender_bias', 'bias', 'hate']:
                if label_type == 'contain_gender_bias':                    
                    label = 'biased' if row[label_type] else 'unbiased'
                elif label_type == 'bias':
                    label = 'no bias' if row[label_type] == 'none' else row[label_type]
                elif label_type == 'hate':
                    label = 'no hate' if row[label_type] == 'none' else row[label_type]
                all_data.append({
                            "data_name": data_name,
                            "task_family": "classification",
                            'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                            'output_ids': tokenizer.encode(label, add_special_tokens=False),
                            'input_text': input_text,
                            'output_text': label,
                })
                
    elif data_name == 'kornli':
        for file_path in ["./raw/kornli/multinli.train.ko.tsv", "./raw/kornli/snli_1.0_train.ko.tsv"]:
            data_object = open(file_path, 'r')
            data_object.readline()
            for line in data_object:
                line = line.strip()
                line = line.split('\t')
                input_text = line[0] + ' ' + line[1]
                label = line[2]
                all_data.append({
                    "data_name": data_name,
                    "task_family": "natural language inference",
                    'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'output_ids': tokenizer.encode(label, add_special_tokens=False),
                    'input_text': input_text,
                    'output_text': label,
                })
            data_object.close()
    
    elif data_name == 'korsts':
        for file_path in ["./raw/korsts/sts-train.tsv"]:
            data_object = open(file_path, 'r')
            for line in data_object:
                line = line.strip()
                line = line.split('\t')
                input_text = line[5] + ' ' + line[6]
                label = line[4]
                all_data.append({
                    "data_name": data_name,
                    "task_family": "semantic textual similarity",
                    'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'output_ids': tokenizer.encode(label, add_special_tokens=False),
                    'input_text': input_text,
                    'output_text': label,
                })
            data_object.close()
     
    elif data_name == 'question_pair':
        df = pd.read_csv("./raw/question_pair/kor_pair_train.csv", header=0)
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            input_text = row['question1'].strip() + ' ' + row['question2'].strip()
            label = "duplicate" if row['is_duplicate'] == 1 else "not duplicate"
            all_data.append({
                "data_name": data_name,
                "task_family": "semantic textual similarity",
                'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                'output_ids': tokenizer.encode(label, add_special_tokens=False),
                'input_text': input_text,
                'output_text': label,
            })
    
    elif data_name == 'paraKQC':
        df = pd.read_csv("./raw/paraKQC/data/paraKQC_v1_generated.tsv", sep='\t')
        pbar = tqdm(total = len(df))
        for i in range(len(df)):
            row = df.iloc[i].to_list()
            input_text = row[0].strip() + ' ' + row[1].strip()
            label = row[2].strip()
            all_data.append({
                "data_name": data_name,
                "task_family": "semantic textual similarity",
                'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                'output_ids': tokenizer.encode(label, add_special_tokens=False),
                'input_text': input_text,
                'output_text': label,
            })
            pbar.update(1)
        pbar.close()
    
    elif data_name == "korquad_v1":
        datasets = load_dataset('squad_kor_v1')
        for data_split in ['train', 'validation']:
            for data in datasets[data_split]:
                input_text = f"질의: {data['question']} 제목: {data['title']} 본문: {data['context']}"
                label = data['answers']['text'][0]
                all_data.append({
                    "data_name": data_name,
                    "task_family": "question answering",
                    'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'output_ids': tokenizer.encode(label, add_special_tokens=False),
                    'input_text': input_text,
                    'output_text': label,
                })
                
    elif data_name == "korquad_v2": # Not use
        datasets = load_dataset('squad_kor_v2')
        for data_split in ['train', 'validation']:
            for data in datasets[data_split]:
                input_text = f"질의: {data['question']} 제목: {data['title']} 본문: {data['context']}"
                label = data['answers']['text'][0]
                all_data.append({
                    "data_name": data_name,
                    "task_family": "question answering",
                    'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'output_ids': tokenizer.encode(label, add_special_tokens=False),
                    'input_text': input_text,
                    'output_text': label,
                })
                
    elif data_name == "sci-news-sum-kr-50":
        file_path = "./raw/sci-news-sum-kr-50/data/"
        files = [ file_path + f for f in os.listdir(file_path) ]

        for file in files:
            file_object = open(file, "rb")
            data = json.loads(file_object.read().decode('utf-8'))
            input_text = f"제목: {data['title']} 본문: {'. '.join(data['sentences'])}"
            label = '. '.join([ data['sentences'][s] for s in data['summaries'] ])
            all_data.append({
                "data_name": data_name,
                "task_family": "extractive summarization",
                'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                'output_ids': tokenizer.encode(label, add_special_tokens=False),
                'input_text': input_text,
                'output_text': label,
            })
    
    elif data_name == "sae4k":
        for file_path in ["./raw/sae4k/data/sae4k_v1.txt", "./raw/sae4k/data/sae4k_v2.txt"]:
            data_object = open(file_path, 'r')
            data_object.readline()
            for line in data_object:
                line = line.strip()
                line = line.split('\t')
                if file_path[-6:] == "v1.txt":
                    input_text = line[0].strip()
                    label = line[1].strip()
                elif file_path[-6:] == "v2.txt":
                    input_text = line[1].strip()
                    label = line[2].strip()
                    
                all_data.append({
                    "data_name": data_name,
                    "task_family": "summarization", # short -> very short
                    'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'output_ids': tokenizer.encode(label, add_special_tokens=False),
                    'input_text': input_text,
                    'output_text': label,
                })
            data_object.close()
            
    elif data_name == "korean_parallel":
        data_object_ko = open("./raw/korean_parallel/korean-english-park.train.ko", 'r')
        data_object_en = open("./raw/korean_parallel/korean-english-park.train.en", 'r')
        
        for line_ko, line_en in zip(data_object_ko, data_object_en):
            line_ko, line_en = line_ko.strip(), line_en.strip()
            input_text = line_ko
            label = line_en
            
            all_data.append({
                "data_name": data_name,
                "task_family": "translation (ko->en)",
                'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                'output_ids': tokenizer.encode(label, add_special_tokens=False),
                'input_text': input_text,
                'output_text': label,
            })
            all_data.append({
                "data_name": data_name,
                "task_family": "translation (en->ko)",
                'input_ids': tokenizer.encode(label, add_special_tokens=False),
                'output_ids': tokenizer.encode(input_text, add_special_tokens=False),
                'input_text': label,
                'output_text': input_text,
            })
            
        data_object_ko.close()
        data_object_en.close()
        
    elif data_name == "transliteration":
        file_path = "./raw/transliteration/data/source/"
        files = [ file_path + f for f in os.listdir(file_path) if os.path.isfile(file_path + f) ]
        for file in files:
            file_object = open(file, "r")
            for line in file_object:
                if line[0] == '#':
                    continue
                else:
                    line = line.split('\t')
                    input_text = line[1].strip()
                    label = line[0].strip()
                    
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "transliteration (ko->en)",
                        'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                        'output_ids': tokenizer.encode(label, add_special_tokens=False),
                        'input_text': input_text,
                        'output_text': label,
                    })
                    
    elif data_name == "Xpersona": # Too low Quality
        file_object = open("./raw/Xpersona/dataset/Ko_persona_train_corrected.json", "rb")
        datasets = json.loads(file_object.read().decode('utf-8'))
        # Not Implemented

    elif data_name == "common_sense":
        file_object = open("./raw/common_sense/ko_wiki_v1_squad.json", "rb")
        datasets = json.loads(file_object.read().decode('utf-8'))["data"]
        for data in datasets:
            paragraph = data["paragraphs"][0]
            for qa in paragraph["qas"]:
                input_text = f"질의: {qa['question']} 제목: {data['title']} 본문: {paragraph['context']}"
                label = qa['answers'][0]['text']
                all_data.append({
                    "data_name": data_name,
                    "task_family": "question answering",
                    'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'output_ids': tokenizer.encode(label, add_special_tokens=False),
                    'input_text': input_text,
                    'output_text': label,
                })
    
    elif data_name == "mindslab_mrc":
        for file_name in ["ko_nia_normal_squad_all", "ko_nia_noanswer_squad_all", "ko_nia_clue0529_squad_all"]:
            file_object = open(f"./raw/mindslab_mrc/{file_name}.json", "rb")
            datasets = json.loads(file_object.read().decode('utf-8'))["data"]
            for data in datasets:
                paragraph = data["paragraphs"][0]
                for qa in paragraph["qas"]:
                    input_text = f"질의: {qa['question']} 제목: {data['title']} 본문: {paragraph['context']}"
                    label = qa['answers'][0]['text']
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "question answering",
                        'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                        'output_ids': tokenizer.encode(label, add_special_tokens=False),
                        'input_text': input_text,
                        'output_text': label,
                    })
    
    elif data_name == "korean_chat":
        file_path = "./raw/korean_chat/"
        files = [ file_path + f for f in os.listdir(file_path) ]
        # passport is not conversation and filter out the chat_logs by only 1 person
        files.remove(file_path + 'passport.xlsx')
        files.remove(file_path + 'transport.xlsx')
        files.remove(file_path + 'water_supply.xlsx')
        files.remove(file_path + 'car_licence.xlsx')
        
        for data_file in files:
            df = pd.read_excel(data_file, header=0)
            prev_intention, input_text = "", ""
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
                        input_text += f"{speaker}: {str(mention[0]).strip()}\n"
                        speaker_list.append(speaker)
                    else:
                        prev_intention = intention
                        chat_logs = input_text.strip().split('\n')
                        if len(set(speaker_list)) > 1:
                            chat_logs_tok = []
                            for c in chat_logs:
                                chat_logs_tok.append(tokenizer.encode(c + '\n', add_special_tokens=False))

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
                                
                        input_text = f"{speaker}: {mention[0].strip()}\n"
                        speaker_list = []

    elif data_name == "ai_hub_kor2eng":
        file_path = "./raw/ai_hub_kor2eng/"
        files = [ file_path + f for f in os.listdir(file_path) ]
        for data_file in files:
            df = pd.read_excel(data_file, header=0)
            for i in range(len(df['원문'])):
                input_text = df['원문'][i].strip()
                label = df['번역문'][i].strip()
                
                all_data.append({
                    "data_name": data_name,
                    "task_family": "translation (ko->en)",
                    'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'output_ids': tokenizer.encode(label, add_special_tokens=False),
                    'input_text': input_text,
                    'output_text': label,
                })
                
                all_data.append({
                    "data_name": data_name,
                    "task_family": "translation (en->ko)",
                    'input_ids': tokenizer.encode(label, add_special_tokens=False),
                    'output_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'input_text': label,
                    'output_text': input_text,
                })
    
    elif data_name == "ai_hub_sentiment_conversation":
        file_path = "./raw/ai_hub_sentiment_conversation/"
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
            for i in range(len(df['감정_대분류'])):
                chat_logs = convs.iloc[i].to_list()#[::2] + convs.iloc[i].to_list()[1::2]
                
                for j, c in enumerate(chat_logs):
                    if c == c:
                        if j%2:
                            chat_logs[j] = '시스템: ' + str(c).strip()
                        else:
                            chat_logs[j] = '사람: ' + str(c).strip()
                    else:
                        chat_logs[j] = ''
                
                    chat_logs_tok = []
                    for c in chat_logs:
                        chat_logs_tok.append(tokenizer.encode(c + '\n', add_special_tokens=False))
                        
                    
                    if task_family == "dialogue":
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

                if task_family == "classification":
                    label = sentiment[i]
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "classification",
                        'input_ids': tokenizer.encode('\n'.join(chat_logs), add_special_tokens=False),
                        'output_ids': tokenizer.encode(label, add_special_tokens=False),
                        'input_text': '\n'.join(chat_logs),
                        'output_text': label,
                    })
                pbar.update(1)
            pbar.close()
                    
    elif data_name == "ai_hub_kor2eng_expert":
        file_path = "./raw/ai_hub_kor2eng_expert/"
        files = [ file_path + f for f in os.listdir(file_path) ]
        for data_file in files:
            df = pd.read_csv(data_file, header=0)
            pbar = tqdm(total=len(df['한국어']))
            for i in range(len(df['한국어'])):
                input_text = df['한국어'][i].strip()
                label = df['영어'][i].strip()
                all_data.append({
                    "data_name": data_name,
                    "task_family": "translation (ko->en)",
                    'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'output_ids': tokenizer.encode(label, add_special_tokens=False),
                    'input_text': input_text,
                    'output_text': label,
                })
                
                all_data.append({
                    "data_name": data_name,
                    "task_family": "translation (en->ko)",
                    'input_ids': tokenizer.encode(label, add_special_tokens=False),
                    'output_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'input_text': label,
                    'output_text': input_text,
                })
                pbar.update(1)
            pbar.close()
    
    elif data_name == "ai_hub_doc_summ":
        file_path = "./raw/ai_hub_doc_summ/"
        files = [ file_path + f for f in os.listdir(file_path) ]
        for i, file_name in enumerate(files):
            file_object = open(f"{file_name}", "rb")
            datasets = json.loads(file_object.read().decode('utf-8'))["documents"]
            pbar = tqdm(total = len(datasets))
            for data in datasets:
#                 contents = [ d[0]['sentence'] for d in data['text'] ]
                contents = []
                for d in data['text']:
                    for dd in d:
                        contents.append(dd['sentence'].strip())
                input_text = f"제목: {data['title']} 본문: {' '.join(contents)}"
                
                label_ext = [ contents[i] for i in data['extractive'] if i is not None ]
                label_ext = ' '.join(label_ext)
                all_data.append({
                    "data_name": data_name,
                    "task_family": "extractive summarization",
                    'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'output_ids': tokenizer.encode(label_ext, add_special_tokens=False),
                    'input_text': input_text,
                    'output_text': label_ext,
                })
                    
                label_abs = data['abstractive'][0]
                all_data.append({
                    "data_name": data_name,
                    "task_family": "abstractive summarization",
                    'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'output_ids': tokenizer.encode(label_abs, add_special_tokens=False),
                    'input_text': input_text,
                    'output_text': label_abs,
                })
                pbar.update(1)
            pbar.close()

    elif data_name == "ai_hub_thesis_summ":
        for data_split in ["train", "valid"]:
            file_path = f"./raw/ai_hub_thesis_summ/{data_split}/"
            files = [ file_path + f for f in os.listdir(file_path) ]
            for i, file_name in enumerate(files):
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))['data']
                pbar = tqdm(total = len(datasets))
                for data in datasets:
                    if f"논문요약" in file_name:
                        input_text = data["summary_entire"][0]['orginal_text'].strip()
                        label = data["summary_entire"][0]['summary_text'].strip()
                        
                        all_data.append({
                            "data_name": data_name,
                            "task_family": "abstractive summarization",
                            'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                            'output_ids': tokenizer.encode(label, add_special_tokens=False),
                            'input_text': input_text,
                            'output_text': label,
                        })
                    
                    input_text = data["summary_section"][0]['orginal_text'].strip()
                    label = data["summary_section"][0]['summary_text'].strip()

                    all_data.append({
                        "data_name": data_name,
                        "task_family": "extractive summarization",
                        'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                        'output_ids': tokenizer.encode(label, add_special_tokens=False),
                        'input_text': input_text,
                        'output_text': label,
                    })
                        
                    pbar.update(1)
                pbar.close()
                
    elif data_name == "ai_hub_book_mrc":
            file_path = f"./raw/ai_hub_book_mrc/"
            files = [ file_path + f for f in os.listdir(file_path) ]
            for data_file in files:
                file_object = open(f"{data_file}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))["data"]
                pbar = tqdm(total = len(datasets))
                for data in datasets:
                    paragraph = data["paragraphs"][0]
                    for qa in paragraph["qas"]:
                        input_text = f"질의: {qa['question'].strip()} 제목: {data['title'].strip()} 본문: {paragraph['context'].strip()}"
                        label = '' if qa['is_impossible'] else qa['answers'][0]['text']
                        all_data.append({
                            "data_name": data_name,
                            "task_family": "question answering",
                            'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                            'output_ids': tokenizer.encode(label, add_special_tokens=False),
                            'input_text': input_text,
                            'output_text': label,
                        })
                    pbar.update(1)
                pbar.close()

    elif data_name == "ai_hub_book_summ":
        for data_split in ["기술과학", "기타", "사회과학", "예술"]:
            file_path = f"./raw/{data_name}/{data_split}/"
            files = [ file_path + f for f in os.listdir(file_path) ]
            all_data = []
            pbar = tqdm(total = len(files))
            for i, file_name in enumerate(files):
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))
                title = datasets['doc_name'].strip() if "doc_name" in datasets.keys() else ''
                input_text = f"제목: {title} 본문: {datasets['passage'].strip()}"
                label = datasets['summary'].strip()
                        
                all_data.append({
                    "data_name": data_name,
                    "task_family": "abstractive summarization",
                    'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'output_ids': tokenizer.encode(label, add_special_tokens=False),
                    'input_text': input_text,
                    'output_text': label,
                })
                    
                pbar.update(1)
            pbar.close()
    
    elif data_name == "ai_hub_callcenter_dialogue":
            file_path = f"./raw/{data_name}/"
            files = [ file_path + f for f in os.listdir(file_path) ]
            all_data = []
            pbar = tqdm(total = len(files))
            for i, file_name in enumerate(files):
                try:
                    file_object = open(f"{file_name}", "rb")
                    datasets = json.loads(file_object.read().decode('utf-8'))
                except UnicodeDecodeError:
                    print('UnicodeDecodeError @', file_name)
                    pbar.update(1)
                    continue
                except json.decoder.JSONDecodeError:
                    print('json.decoder.JSONDecodeError @', file_name)
                    pbar.update(1)
                    continue
                
                chat_id = datasets[0]['대화셋일련번호']
                chat_logs = []
                for data in datasets:
                    if data['대화셋일련번호'] != chat_id:
                        chat_logs_tok = []
                        for c in chat_logs:
                            chat_logs_tok.append(tokenizer.encode(c + '\n', add_special_tokens=False))

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
                        chat_logs = []
                        
                    chat_id = data['대화셋일련번호']
                    speaker = data['화자']
                    any_chats = data['고객질문(요청)'] + data['상담사질문(요청)'] + data['고객답변'] + data['상담사답변']
                    chat_logs.append(f"{speaker}: {any_chats}")
                                    
                chat_logs_tok = []
                for c in chat_logs:
                    chat_logs_tok.append(tokenizer.encode(c + '\n', add_special_tokens=False))

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
                chat_logs = []
                pbar.update(1)
            pbar.close()
    
    elif data_name == "ai_hub_ordering_dialogue":
            file_path = f"./raw/{data_name}/"
            files = [ file_path + f for f in os.listdir(file_path) ]
            all_data = []
            pbar = tqdm(total = len(files))
            for i, file_name in enumerate(files):
                df = pd.read_csv(f"{file_name}", header=0)
                chat_id = df['QA번호'][0]
                chat_logs = []
                for i in range(len(df)):
                    if df['QA번호'][i] != chat_id:
                        for j in range(1,len(chat_logs)):
                            all_data.append({
                                "data_name": data_name,
                                "task_family": "dialogue",
                                'input_ids': tokenizer.encode('\n'.join(chat_logs[:j]), add_special_tokens=False),
                                'output_ids': tokenizer.encode(chat_logs[j], add_special_tokens=False),
                                'input_text': '\n'.join(chat_logs[:j]),
                                'output_text': chat_logs[j],
                        })
                        chat_logs = []
                        
                    chat_id = df['QA번호'][i]
                    speaker = "고객" if df['발화자'][i] == 'c' else "점원"
                    chats = df['발화문'][i]
                    chat_logs.append(f"{speaker}: {chats}")
                
                chat_logs_tok = []
                for c in chat_logs:
                    chat_logs_tok.append(tokenizer.encode(c + '\n', add_special_tokens=False))

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
            file_path = f"./raw/{data_name}/{data_split}/"
            files = [ file_path + f for f in os.listdir(file_path) ]
            all_data = []
            for i, file_name in enumerate(files):
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))["data"]
                pbar = tqdm(total = len(datasets))
                for data in datasets:
                    participants = data['header']['participantsInfo']
                    participants = { participants[i]['participantID']: participants[i]['age'] + participants[i]['gender'] for i in range(len(participants)) }
                    chat_logs = []
                    for d in data['body']:
                        chat_logs.append(f"{participants[d['participantID']]}: {d['utterance']}")
                        
                    chat_logs_tok = []
                    for c in chat_logs:
                        chat_logs_tok.append(tokenizer.encode(c + '\n', add_special_tokens=False))

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
                
    elif data_name == "ai_hub_conversation_summ":
        for data_split in ["train", "validation"]:
            file_path = f"./raw/{data_name}/{data_split}/"
            files = [ file_path + f for f in os.listdir(file_path) ]
            all_data = []
            for i, file_name in enumerate(files):
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))["data"]
                pbar = tqdm(total = len(datasets))
                for data in datasets:
                    participants = data['header']['participantsInfo']
                    participants = { participants[i]['participantID']: participants[i]['age'] + participants[i]['gender'] for i in range(len(participants)) }
                    
                    chat_logs = []
                    for d in data['body']['dialogue']:
                        chat_logs.append(f"{participants[d['participantID']]}: {d['utterance']}")
                    
                    chat_logs_tok = []
                    for c in chat_logs:
                        chat_logs_tok.append(tokenizer.encode(c + '\n', add_special_tokens=False))

                    if task_family == "dialogue":
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
                    
                    elif task_family == "summarization" or "summarization" in task_family:
                        all_data.append({
                            "data_name": data_name,
                            "task_family": "abstractive summarization",
                            'input_ids': tokenizer.encode('\n'.join(chat_logs), add_special_tokens=False),
                            'output_ids': tokenizer.encode(data['body']['summary'], add_special_tokens=False),
                            'input_text': '\n'.join(chat_logs),
                            'output_text': data['body']['summary'],
                        })
                    
                    pbar.update(1)
                pbar.close()
                
    elif data_name == "ai_hub_kor2eng_technology"\
            or data_name == "ai_hub_kor2eng_socialscience":
        file_path = f"./raw/{data_name}/"
        files = [ file_path + f for f in os.listdir(file_path) ]
        for data_file in files:
            df = pd.read_csv(data_file, header=0)
            pbar = tqdm(total = len(df['ko']))
            for i in range(len(df['ko'])):
                input_text = df['ko'][i].strip()
                label = df['en'][i].strip()
                
                all_data.append({
                    "data_name": data_name,
                    "task_family": "translation (ko->en)",
                    'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'output_ids': tokenizer.encode(label, add_special_tokens=False),
                    'input_text': input_text,
                    'output_text': label,
                })
                
                all_data.append({
                    "data_name": data_name,
                    "task_family": "translation (en->ko)",
                    'input_ids': tokenizer.encode(label, add_special_tokens=False),
                    'output_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'input_text': label,
                    'output_text': input_text,
                })

                pbar.update(1)
            pbar.close()
            
    elif data_name == "ai_hub_kor2jpn"\
        or data_name == "ai_hub_kor2chn_technology"\
        or data_name == "ai_hub_kor2chn_socialscience":
        file_path = f"./raw/{data_name}/"
        files = [ file_path + f for f in os.listdir(file_path) ]
        target_lang = '일본어' if data_name[-3:] == 'jpn' else '중국어'
        target_code = 'jp' if data_name[-3:] == 'jpn' else 'cn'
        for data_file in files:
            df = pd.read_csv(data_file, header=0)
            pbar = tqdm(total = len(df['한국어']))
            for i in range(len(df['한국어'])):
                input_text = df['한국어'][i].strip()
                label = df[target_lang][i].strip()
                
                all_data.append({
                    "data_name": data_name,
                    "task_family": f"translation (ko->{target_code})",
                    'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'output_ids': tokenizer.encode(label, add_special_tokens=False),
                    'input_text': input_text,
                    'output_text': label,
                })
                
                all_data.append({
                    "data_name": data_name,
                    "task_family": f"translation ({target_code}->ko)",
                    'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'output_ids': tokenizer.encode(label, add_special_tokens=False),
                    'input_text': input_text,
                    'output_text': label,
                })
                
                pbar.update(1)
            pbar.close()
    
    elif data_name == "ai_hub_command":
        file_path = f"./raw/{data_name}/"
        files = [ file_path + f for f in os.listdir(file_path) ]
        for data_file in files:
            df = pd.read_excel(data_file, header=0)
            pbar = tqdm(total = len(df['문장']))
            for i in range(len(df['문장'])):
                input_text = df['문장'][i].strip()
                label = df["의도 (Intention)"][i].strip()
                all_data.append({
                    "data_name": data_name,
                    "task_family": "classification",
                    'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'output_ids': tokenizer.encode(label, add_special_tokens=False),
                    'input_text': input_text,
                    'output_text': label,
                })
                pbar.update(1)
            pbar.close()
            
    elif data_name == "ai_hub_broadcasting_conversation" or data_name == "ai_hub_domain_conversation":
        for data_split in ['train', 'validation']:
            file_path = f"./raw/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))
                participants = datasets['speaker']
                participants = { participants[i]['id']: participants[i]['age'] + ' ' + participants[i]['role'] + ' ' + participants[i]['sex'] for i in range(len(participants)) }
                
                chat_logs = []
                for d in datasets['utterance']:
                    if d['speaker_id'] == '?':
                        continue
                    chat_logs.append(f"{participants[d['speaker_id']]}: {d['original_form']}")
                
                chat_logs_tok = []
                for c in chat_logs:
                    chat_logs_tok.append(tokenizer.encode(c + '\n', add_special_tokens=False))

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
            
    elif data_name == "ai_hub_casual_domain_conversation" or data_name == "ai_hub_goal_oriented_dialogue":
        for data_split in ['train', 'validation']:
            file_path = f"./raw/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                try:
                    datasets = json.loads(file_object.read().decode('utf-8'))["info"]
                except json.decoder.JSONDecodeError: # Filter out invalid data
                    continue
                    
                chat_logs = []
                chat_by_line = datasets[0]["annotations"]["lines"]
                for c in chat_by_line:
                    norm_text = c["norm_text"].replace('\xa0', ' ')
                    if norm_text[1] == '.':
                        norm_text = norm_text[2:]
                    chat_logs.append(c["speaker"]["age"] + c["speaker"]["sex"] + ': ' + norm_text)
                    
                chat_logs_tok = []
                for c in chat_logs:
                    chat_logs_tok.append(tokenizer.encode(c + '\n', add_special_tokens=False))
                
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
    
    elif data_name == "ai_hub_essay_evaluation":
        for data_split in ['train', 'validation']:
            file_path = f"./raw/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))
                
                input_text = [ p['paragraph_txt'].strip() for p in datasets['paragraph'] ]
                input_text = '\n'.join(input_text)
                input_text = input_text.replace('#@문장구분#', '\n').replace(' .\n', '.\n').replace('.\n ', '.\n').replace('\n\n', '\n')
                input_text = input_text.replace(' \n', '\n').replace('\n ', '\n').strip()
                label = str(np.round(datasets['score']['essay_scoreT_avg'], 1))
                all_data.append({
                    "data_name": data_name,
                    "task_family": "classification",
                    'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                    'output_ids': tokenizer.encode(label, add_special_tokens=False),
                    'input_text': input_text,
                    'output_text': label,
                })
                pbar.update(1)
            pbar.close()
    
    elif data_name == "ai_hub_casual_kor2chn2jpn_corpus":
        for data_split in ['train', 'validation']:
            file_path = f"./raw/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                try:
                    datasets = json.loads(file_object.read().decode('utf-8'))
                except UnicodeDecodeError: # Filter out invalid data
                    pbar.update(1)
                    continue
                    
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
                    input_text = data['원문']
                    label = data['최종번역문']
                    
                    all_data.append({
                        "data_name": data_name,
                        "task_family": task_family,
                        'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                        'output_ids': tokenizer.encode(label, add_special_tokens=False),
                        'input_text': input_text,
                        'output_text': label,
                    })
                pbar.update(1)
            pbar.close()
    
    ###
    elif data_name == "ai_hub_ethical_text":
        for data_split in ['train', 'validation']:
            file_path = f"./raw/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))
                for data in datasets:
                    dialogue_history, dialogue_history_tok = [], []
                    dialogue_history_label, dialogue_history_label_tok = [], []
                    for turn in data['sentences']:
                        dialogue_history.append(f"화자{turn['speaker']}: {turn['text']}")
                        dialogue_history_label.append('윤리적' if turn['is_immoral'] else '비윤리적') # Type is so ambiguous
                        dialogue_history_tok.append(tokenizer.encode(dialogue_history[-1] + '\n', add_special_tokens=False))
                        dialogue_history_label_tok.append(tokenizer.encode(dialogue_history_label[-1], add_special_tokens=False))
                    
                    for j in range(1, len(dialogue_history)):
                        all_data.append({
                            "data_name": data_name,
                            "task_family": "classification",
                            'input_ids': dialogue_history_tok[:j],
                            'output_ids': dialogue_history_label_tok[j],
                            'input_text': '\n'.join(dialogue_history[:j]),
                            'output_text': dialogue_history_label[j],
                        })
                pbar.update(1)
            pbar.close()
    
    elif data_name == "ai_hub_patent_eng2kor":
        for data_split in ['train', 'validation']:
            file_path = f"./raw/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))["labeled_data"]
                
                for data in datasets:
                    input_text = f"title: {data['invention_title_eng']}\nabstract: {data['astrt_cont_eng']}\nclaim: {data['claim_eng']}"
                    label = f"제목: {data['invention_title_kor']}\n초록: {data['astrt_cont_kor']}\n청구항: {data['claim_kor']}"
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "translation (en->ko)",
                        'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                        'output_ids': tokenizer.encode(label, add_special_tokens=False),
                        'input_text': input_text,
                        'output_text': label,
                    })
                    
                    all_data.append({
                        "data_name": data_name,
                        "task_family": "translation (ko->en)",
                        'input_ids': tokenizer.encode(label, add_special_tokens=False),
                        'output_ids': tokenizer.encode(input_text, add_special_tokens=False),
                        'input_text': label,
                        'output_text': input_text,
                    })
                pbar.update(1)
            pbar.close()
            
    elif data_name == "ai_hub_admin_document_mrc" or data_name == "ai_hub_newsarticle_mrc":
        for data_split in ['train', 'validation']:
            file_path = f"./raw/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                try:
                    datasets = json.loads(file_object.read().decode('utf-8'))['data']
                except json.decoder.JSONDecodeError: # Filter out invalid data
                    pbar.update(1)
                    continue
                for data in datasets:
                    for qa in data['paragraphs'][0]['qas']:
                        input_text = f"질의: {qa['question']}\n제목: {data['doc_title']}\n본문: {data['paragraphs'][0]['context']}"
                        label = "" if qa['is_impossible'] else qa['answers']['text']
                        all_data.append({
                            "data_name": data_name,
                            "task_family": "question answering",
                            'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                            'output_ids': tokenizer.encode(label, add_special_tokens=False),
                            'input_text': input_text,
                            'output_text': label,
                        })
                pbar.update(1)
            pbar.close()
            
    elif data_name == "ai_hub_lowquality_stt_dialogue":
        for data_split in ['train', 'validation']:
            file_path = f"./raw/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))['dataSet']
                participants = datasets['typeInfo']['speakers']
                participants = { participants[i]['id'] : f"{participants[i]['age']} {participants[i]['gender']}자 {participants[i]['type'][:-1]}" for i in range(len(participants)) }
                chat_logs = []
                for data in datasets['dialogs']:
                    chat_logs.append(f"{participants[data['speaker']]}: {data['text']}")
                    
                chat_logs_tok = []
                for c in chat_logs:
                    chat_logs_tok.append(tokenizer.encode(c + '\n', add_special_tokens=False))
                    
                chat_logs_tok_flatten = []
                for j in range(1, len(chat_logs_tok), 2):
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

    elif data_name == "ai_hub_summary_report_generation" or data_name == "ai_hub_script_summary":
        for data_split in ['train', 'validation']:
            file_path = f"./raw/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))
                input_text = f"제목: {datasets['Meta(Acqusition)']['doc_name']}\n본문: {datasets['Meta(Refine)']['passage']}"
                for k, v in datasets['Annotation'].items():
                    if k == "summary1" and v != "null" and v is not None:
                        all_data.append({
                            "data_name": data_name,
                            "task_family": "abstractive summarization",
                            'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                            'output_ids': tokenizer.encode(v, add_special_tokens=False),
                            'input_text': input_text,
                            'output_text': v,
                        })
                    elif k == "summary2" and v != "null" and v is not None:
                        all_data.append({
                            "data_name": data_name,
                            "task_family": "extractive summarization",
                            'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                            'output_ids': tokenizer.encode(v, add_special_tokens=False),
                            'input_text': input_text,
                            'output_text': v,
                        })
                    elif k == "summary3" and v != "null" and v is not None:
                        all_data.append({
                            "data_name": data_name,
                            "task_family": "extractive summarization",
                            'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                            'output_ids': tokenizer.encode(v, add_special_tokens=False),
                            'input_text': input_text,
                            'output_text': v,
                        })
                pbar.update(1)
            pbar.close()
    
    elif data_name == "ai_hub_multilingual_speaking_translation":
        for data_split in ['train', 'validation']:
            file_path = f"./raw/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))
                
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
                    print(datasets[0]['S_Code'][-2:])                    
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
                    print(datasets[0]['T_Code'][-2:])
                    return
                    
                for data in datasets:
                    input_text = data['원문']
                    label = data['최종번역문']
                    if label == 0.0:
                        continue
                    all_data.append({
                        "data_name": data_name,
                        "task_family": task_family,
                        'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                        'output_ids': tokenizer.encode(label, add_special_tokens=False),
                        'input_text': input_text,
                        'output_text': label,
                    })
                pbar.update(1)
            pbar.close()
                
    elif data_name == "ai_hub_complaint_automation":
        for data_split in ['train', 'validation']:
            file_path = f"./raw/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))['documents']
                for data in datasets:
                    input_text = data['Q_refined']
                    label = data['labeling']['intent']['category']
                    all_data.append({
                        "data_name": data_name,
                        "task_family": 'classification',
                        'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                        'output_ids': tokenizer.encode(label, add_special_tokens=False),
                        'input_text': input_text,
                        'output_text': label,
                    })
                pbar.update(1)
            pbar.close()
    
    elif data_name == "ai_hub_food_translation_corpus" or\
        data_name == "ai_hub_broadcasting_translation_corpus" or\
        data_name == "ai_hub_casualtalk_translation" or\
        data_name == "ai_hub_tech_translation_corpus":
        for data_split in ['train', 'validation']:
            file_path = f"./raw/{data_name}/{data_split}/"
            files = [ os.path.join(path, name) for path, subdirs, files in os.walk(file_path) for name in files ]
            pbar = tqdm(total = len(files))
            for file_name in files:
                file_object = open(f"{file_name}", "rb")
                datasets = json.loads(file_object.read().decode('utf-8'))['data']
                
                if datasets[0]['source_language'] == 'en':
                    for data in datasets:
                        input_text = data['en_original']
                        label = data['ko']
                        
                        all_data.append({
                            "data_name": data_name,
                            "task_family": 'translation (en->ko)',
                            'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                            'output_ids': tokenizer.encode(label, add_special_tokens=False),
                            'input_text': input_text,
                            'output_text': label,
                        })
                        
                elif datasets[0]['source_language'] == 'cn':
                    for data in datasets:
                        input_text = data['cn_original']
                        label = data['ko']
                        
                        all_data.append({
                            "data_name": data_name,
                            "task_family": 'translation (cn->ko)',
                            'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                            'output_ids': tokenizer.encode(label, add_special_tokens=False),
                            'input_text': input_text,
                            'output_text': label,
                        })
                
                elif datasets[0]['source_language'] == 'jp':
                    for data in datasets:
                        input_text = data['jp_original']
                        label = data['ko']
                        
                        all_data.append({
                            "data_name": data_name,
                            "task_family": 'translation (jp->ko)',
                            'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                            'output_ids': tokenizer.encode(label, add_special_tokens=False),
                            'input_text': input_text,
                            'output_text': label,
                        })
                        
                elif datasets[0]['source_language'] == 'ko':
                    if "cn" in datasets[0].keys():
                        for data in datasets:
                            input_text = data['ko_original']
                            label = data['cn']
                            all_data.append({
                                "data_name": data_name,
                                "task_family": 'translation (ko->cn)',
                                'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                                'output_ids': tokenizer.encode(label, add_special_tokens=False),
                                'input_text': input_text,
                                'output_text': label,
                            })
                            
                    elif "jp" in datasets[0].keys():
                        for data in datasets:
                            input_text = data['ko_original']
                            label = data['jp']
                            all_data.append({
                                "data_name": data_name,
                                "task_family": 'translation (ko->jp)',
                                'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                                'output_ids': tokenizer.encode(label, add_special_tokens=False),
                                'input_text': input_text,
                                'output_text': label,
                            })
                    
                    elif "en" in datasets[0].keys():
                        for data in datasets:
                            input_text = data['ko_original']
                            label = data['en']
                            all_data.append({
                                "data_name": data_name,
                                "task_family": 'translation (ko->en)',
                                'input_ids': tokenizer.encode(input_text, add_special_tokens=False),
                                'output_ids': tokenizer.encode(label, add_special_tokens=False),
                                'input_text': input_text,
                                'output_text': label,
                            })
                pbar.update(1)
            pbar.close()

                        
    else:
        print("out of predefined dataset", data_name)
    print(all_data[:5])
    return all_data

def main():
#     selected_task_family = [ "classification" ]
    selected_task_family = [ "dialogue" ]
    
    dataset = []
    if "classification" in selected_task_family: # total 9
        dataset += preprocess_dataset('3i4k')
        dataset += preprocess_dataset('nsmc')
        dataset += preprocess_dataset('toxic_comment')
        dataset += preprocess_dataset('korean_hate_speech')
        dataset += preprocess_dataset('ai_hub_sentiment_conversation')
        dataset += preprocess_dataset('ai_hub_command')
        dataset += preprocess_dataset('ai_hub_essay_evaluation')
        dataset += preprocess_dataset('ai_hub_ethical_text')
        dataset += preprocess_dataset('ai_hub_complaint_automation')
        
    if "natural language inference" in selected_task_family: # total 1
        dataset += preprocess_dataset('kornli')
        
    if "semantic textual similarity" in selected_task_family: # total 3
        dataset += preprocess_dataset('korsts')
        dataset += preprocess_dataset('question_pair')
        dataset += preprocess_dataset('paraKQC')
    
    if "question answering" in selected_task_family: # total 6
        dataset += preprocess_dataset('korquad_v1')
#         dataset += preprocess_dataset('korquad_v2') # Not use
        dataset += preprocess_dataset('common_sense')
        dataset += preprocess_dataset('mindslab_mrc')
        dataset += preprocess_dataset('ai_hub_book_mrc')
        dataset += preprocess_dataset('ai_hub_admin_document_mrc')
        dataset += preprocess_dataset('ai_hub_newsarticle_mrc')

    if "summarization" in selected_task_family: # total 8
        dataset += preprocess_dataset("sci-news-sum-kr-50")
        dataset += preprocess_dataset("sae4k")
        dataset += preprocess_dataset("ai_hub_doc_summ")
        dataset += preprocess_dataset("ai_hub_thesis_summ")
        dataset += preprocess_dataset("ai_hub_book_summ")
        dataset += preprocess_dataset("ai_hub_conversation_summ")
        dataset += preprocess_dataset("ai_hub_summary_report_generation")
        dataset += preprocess_dataset("ai_hub_script_summary")
        
    if "translation" in selected_task_family: # 13 here +1 at below
        dataset += preprocess_dataset("korean_parallel")
        dataset += preprocess_dataset("ai_hub_kor2eng")
        dataset += preprocess_dataset("ai_hub_kor2eng_expert")
        dataset += preprocess_dataset("ai_hub_kor2eng_technology")
        dataset += preprocess_dataset("ai_hub_kor2eng_socialscience")
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
        
    if "transliteration" in selected_task_family: # +1 at translation
        dataset += preprocess_dataset("transliteration")
        
    if "dialogue" in selected_task_family: # total 11
        dataset += preprocess_dataset("korean_chat")
        dataset += preprocess_dataset('ai_hub_sentiment_conversation')
        dataset += preprocess_dataset("ai_hub_callcenter_dialogue")
        dataset += preprocess_dataset("ai_hub_ordering_dialogue")
        dataset += preprocess_dataset("ai_hub_koreansns_dialogue")
        dataset += preprocess_dataset("ai_hub_conversation_summ")
        dataset += preprocess_dataset("ai_hub_broadcasting_conversation")
        dataset += preprocess_dataset("ai_hub_domain_conversation")
        dataset += preprocess_dataset("ai_hub_casual_domain_conversation")
        dataset += preprocess_dataset("ai_hub_goal_oriented_dialogue")
        dataset += preprocess_dataset("ai_hub_lowquality_stt_dialogue")

    with open('./processed/train_data_all.pkl', 'wb') as f:
        pickle.dump(dataset, f)
        
if __name__ == '__main__':
    main()
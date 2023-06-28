[한국어](README.md)

# pko-t5

This repository, pko-T5 is trained [T5 v1.1](https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/released_checkpoints.md) model for korean only.
The model is trained on large-scale korean corpus without english.
And then, we would be expected the model is improved performance than mT5 or byT5. 

We use pretrained BBPE for tokenization in Korean without OOV was used instead of sentencepiece.
And, we parse various Korean data (나무위키, 위키피디아, 모두의말뭉치, ...) to the tokenizer. The data is applied to the task for T5 pretraining with span corruption task.

- Flan T5: [Source Code](https://github.com/paust-team/pko-t5/tree/main/pkot5/flan/README.md) / [Model](https://huggingface.co/paust/pko-flan-t5-large)
- Chat T5: [Source Code](https://github.com/paust-team/pko-t5/tree/main/pkot5/chat/README.md) / [Model](https://huggingface.co/paust/pko-chat-t5-large)

## Download Model
| Model        | transformers                                               |
|--------------|------------------------------------------------------------|
| pko-t5-small | [transformers](https://huggingface.co/paust/pko-t5-small)  |
| pko-t5-base  | [transformers](https://huggingface.co/paust/pko-t5-base)   |
| pko-t5-large | [transformers](https://huggingface.co/paust/pko-t5-large)  |


## Usage
The models can be downloaded by transformers API. But, you should use `T5TokenizerFast` instead of `T5Tokenizer` when using tokenization for text.

### Pre-training Example
- Single-node pre-training
```bash
python3 -m torch.distributed.launch \
  --use_env \
  --nproc_per_node 8
  --logidr {specific log directory}
  -m pkot5.pretraining \
  --grpc_endpoint <grpc_endpoint>
  --model_size <model_size>  # ex: small, base, large
  --resume_checkpoint <ckpt_step>  # ex: 200000
```

- multi-node pre-training
```bash
python3 -m torch.distributed.launch \
  --use_env \
  --nproc_per_node 8
  --nnode <num nodes>
  --node_rank <node_rank>
  --master_addr <master_addr>  # ex: node 0
  --logidr {specific log directory}
  -m pkot5.pretraining \
  --grpc_endpoint <grpc_endpoint>
  --model_size <model_size>  # ex: small, base, large
  --resume_checkpoint <ckpt_step>  # ex: 200000
```

### Fine-tuning Example
```python
from transformers import T5TokenizerFast, T5ForConditionalGeneration

tokenizer = T5TokenizerFast.from_pretrained('paust/pko-t5-base')
model = T5ForConditionalGeneration.from_pretrained('paust/pko-t5-base')

input_ids = tokenizer(["qa question: 당신의 이름은 무엇인가요?"]).input_ids
labels = tokenizer(["T5 입니다."]).input_ids
outputs = model(input_ids=input_ids, labels=labels)

print(f"loss={outputs.loss} logits={outputs.logits}")
```


## [KLUE](https://arxiv.org/pdf/2105.09680.pdf) Evaluation (dev)

|     | Model                                                            | ynat (macro F1) | sts (pearsonr/F1) | nli (acc) | ner (entity-level F1) | re (micro F1) | dp (LAS)  | mrc (EM/F1) |
|-----|------------------------------------------------------------------|-----------------|-------------------|-----------|-----------------------|---------------|-----------|-------------|
|     | Baseline                                                         | **87.30**       | **93.20/86.13**   | **89.50** | 86.06                 | 71.06         | 87.93     | **75.26/-** |
| FT  | [pko-t5-small](https://huggingface.co/paust/pko-t5-small) (77M)  | 86.21           | 77.99/77.01       | 69.20     | 82.60                 | 66.46         | 93.15     | 43.81/46.58 |
| FT  | [pko-t5-base](https://huggingface.co/paust/pko-t5-base) (250M)   | 87.29           | 90.25/83.43       | 79.73     | 87.80                 | 67.23         | 97.28     | 61.53/64.74 |
| FT  | [pko-t5-large](https://huggingface.co/paust/pko-t5-large) (800M) | 87.12           | 92.05/85.24       | 84.96     | **88.18**             | **75.17**     | **97.60** | 68.01/71.44 |
| MT  | pko-t5-small                                                     | 84.54           | 68.50/72/02       | 51.16     | 74.69                 | 66.11         | 80.40     | 43.60/46.28 |
| MT  | pko-t5-base                                                      | 86.89           | 83.96/80.30       | 72.03     | 85.27                 | 66.59         | 95.05     | 61.11/63.94 |
| MT  | pko-t5-large                                                     | 87.57           | 91.93/86.29       | 83.63     | 87.41                 | 71.34         | 96.99     | 70.70/73.72 |

- FT: Single-task finetuning / MT: multi-task finetuning
- [Baseline](https://arxiv.org/abs/2105.09680): SOTA scores on dev introduced by the KLUE paper.

The klue evaluation table is trained on max_length of input_ids was 1300. 
On the setting, encoded `context` got over 98.4% of training set and 100% fo dev set.

### Additional MRC evaluation
1. An experiment with context sliding of max_length=512
2. An experiment with context and title

|       | (1) EM / F1 | (2) EM / F1 |
|-------|-------------|-------------|
| small | 42.20/45.03 | 46.85/50.46 |
| base  | 57.06/60.20 | 63.12/67.38 |
| large | 61.53/64.94 | 70.15/74.20 |

When an experiment of inclusion of title, the model performance improved. but, the model needed more sequence length.

## License
This repository code and model are published by [MIT license](https://github.com/paust-team/pko-t5/blob/main/LICENSE).

## References
- [Google T5 v1.1](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md)
- [Transformers](https://github.com/huggingface/transformers)

## Citation
```bibtex
@software{paust_pkot5_v1,
  author = {Dennis Park},
  month = {5},
  title = {pko-t5: PAUST Korean T5 for text-to-text unified framework},
  url = {https://github.com/paust-team/pko-t5},
  version = {1.0},
  year = {2022}
}
```
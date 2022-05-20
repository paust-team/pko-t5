# pko-t5

pko-t5 는 한국어 전용 데이터로 학습한 [t5 v1.1 모델](https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/released_checkpoints.md)입니다.

한국어를 tokenize 하기 위해서 sentencepiece 대신 OOV 가 없는 BBPE 를 사용했으며 한국어 데이터 (나무위키, 위키피디아, 모두의말뭉치 등..) 를 T5 의 span corruption task 를 사용해서 unsupervised learning 만 적용하여 학습을 진행했습니다.

pko-t5 를 사용하실 때는 실제 task 에 파인튜닝하여 사용하시기 바랍니다.

## Download Model
|Model| transformers                                              |
|---|-----------------------------------------------------------|
|pko-t5-small| [transformers](https://huggingface.co/paust/pko-t5-small) |
|pko-t5-base| [transformers](https://huggingface.co/paust/pko-t5-base)         |
|pko-t5-large| [transformers](https://huggingface.co/paust/pko-t5-large)        |

## Usage
transformers 의 API 를 사용하여 접근 가능합니다. tokenizer 를 사용할때는 `T5Tokenizer` 가 아니라 `T5TokenizerFast` 를 사용해주십시오. model 은 T5ForConditionalGeneration 를 그대로 활용하시면 됩니다.

### Example
```python
from transformers import T5TokenizerFast, T5ForConditionalGeneration

tokenizer = T5TokenizerFast.from_pretrained('paust/pko-t5-base')
model = T5ForConditionalGeneration.from_pretrained('paust/pko-t5-base')

input_ids = tokenizer(["nsmc sentence: 당신의 이름은 무엇인가요?"]).input_ids
labels = tokenizer(["T5 입니다."]).input_ids
outputs = model(input_ids, labels)

print(f"loss={outputs.loss} logits={outputs.logits}")
```
    

## Klue 평가 (dev)

|  | Model | ynat (macro F1) | sts (pearsonr/F1) | nli (acc) | ner (entity-level F1) | re (micro F1) | dp (LAS) | mrc (EM/F1) |
| --- | --- |-----------------| --- | --- | --- | --- | --- | --- |
|  | Baseline | **87.30**       | **93.20/86.13** | **89.50** | 86.06 | 71.06 | 87.93 | 75.26/- |
| FT | pko-t5-small (77M) | 86.21           | 77.99/77.01 | 69.20 | 82.60 | 62.95 | 93.15 | 43.81/46.58 |
| FT | pko-t5-base (250M) | 87.29           | 90.25/83.43 | 79.73 | 87.80 | 72.94 | 97.28 | 61.53/64.74 |
| FT | pko-t5-large (800M) | 87.12           | 92.05/85.24 | 84.96 | **88.18** | 72.26 | 97.60 | 68.01/71.44 |
| MT | pko-t5-small | 85.85           | 79.12/77.81 | 66.8 | 81.53 | 67.93 | 91.38 | 44.97/48.07 |
| MT | pko-t5-base | 86.86           | 87.61/81.42 | 75.46 | 86.85 | 71.85 | 96.32 | 61.95/65.06 |
| MT | pko-t5-large | 87.25           | 91.05/84.58 | 82.16 | 87.63 | **74.78** | **97.33** | **69.18/71.92** |

- FT: 싱글태스크 파인튜닝 / MT: 멀티태스크 파인튜닝
- [Baseline](https://arxiv.org/abs/2105.09680): KLUE 논문에서 소개된 dev set 에 대한 SOTA 점수

위의 klue 평가는 input_ids 의 max length를 1300 으로 하여 학습했습니다. 이렇게 하면 encoding 된 context 가 train=98.4% / dev=100% 로 커버가 됩니다.

### MRC에 대한 추가 실험 평가

1. max length를 512 로 하여 context 를 슬라이딩해서 학습
2. context 이외에 title 을 포함하여 학습

|  | (1) EM / F1 | (2) EM / F1 |
| --- | --- | --- |
| small | 42.20/45.03 | 46.85/50.46 |
| base | 57.06/60.20 | 63.12/67.38 |
| large | 61.53/64.94 | 70.15/74.20 |

title 을 포함했을 때 성능이 좀 더 좋지만 그에 따라 sequence length 가 늘어났습니다.

## License
PAUST에서 만든 pko-t5는 [MIT license](https://github.com/paust-team/pko-t5/blob/main/LICENSE) 하에 공개되어 있습니다.

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
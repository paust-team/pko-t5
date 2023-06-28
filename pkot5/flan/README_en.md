[한국어](https://github.com/paust-team/pko-t5/tree/main/pkot5/flan/README.md)

# FLAN T5

FLAN T5 is a model based on [paust/pko-t5-large](https://huggingface.co/paust/pko-t5-large) that has been trained on various tasks through the Instruction-Finetuning.

We are currently performing Instruction-Finetuning while updating the model with intermediate results.

### Trained Tasks

| Task name                  | Task type      | 
|----------------------------|----------------|
| NSMC                       | Classification |
| Klue Ynat                  | Classification |
| KorNLI                     | Classification |
| KorSTS                     | Classification |
| QuestionPair               | Classification |
| Klue STS                   | Classification |
| AIHub news Summary         | Summarization  |
| AIHub document Summary     | Summarization  |
| AIHub book Summary         | Summarization  |
| AIHub conversation Summary | Summarization  |
| AIHub ko-to-en             | Translation    |
| AIHub ko-to-en Expert      | Translation    |
| AIHub ko-to-en Tech        | Translation    |
| AIHub ko-to-en social      | Translation    |
| AIHub ko-to-jp             | Translation    |
| AIHub ko-to-cn Tech        | Translation    |
| AIHub Translation Corpus   | Translation    |
| korquad                    | QA             |
| Klue MRC                   | QA             |
| AIHub mindslab's MRC       | QA             |


### Model
- [Hugginface 링크](https://huggingface.co/paust/pko-flan-t5-large)


### Example
```python
from transformers import T5ForConditionalGeneration, T5TokenizerFast

tokenizer = T5TokenizerFast.from_pretrained('paust/pko-flan-t5-large')
model = T5ForConditionalGeneration.from_pretrained('paust/pko-flan-t5-large', device_map='cuda')

prompt = """서울특별시(서울特別市, 영어: Seoul Metropolitan Government)는 대한민국 수도이자 최대 도시이다. 선사시대부터 사람이 거주하였으나 본 역사는 백제 첫 수도 위례성을 시초로 한다. 삼국시대에는 전략적 요충지로서 고구려, 백제, 신라가 번갈아 차지하였으며, 고려 시대에는 왕실의 별궁이 세워진 남경(南京)으로 이름하였다.
한국의 수도는 어디입니까?"""
input_ids = tokenizer(prompt, add_special_tokens=True, return_tensors='pt').input_ids
output_ids = model.generate(input_ids=input_ids.cuda(), max_new_tokens=32, num_beams=12)
text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
print(text)  # 서울특별시
```
[ENGLISH](https://github.com/paust-team/pko-t5/tree/main/pkot5/chat/README_en.md)

# Chat T5

Chat T5 는 [pko-flan-t5-large](https://huggingface.co/paust/pko-flan-t5-large) 를 기반으로 만들었습니다.

[KoAlpaca](https://github.com/beomi/koalpaca) 에서 제공하는 데이터셋과 [evolve-instruct](https://github.com/lcw99/evolve-instruct) 에서 제공하는 데이터셋을 학습했습니다.
좋은 데이터를 공개해주셔서 감사합니다.


### Model
- [Huggingface](https://huggingface.co/paust/pko-chat-t5-large)

### Example
```python
from transformers import T5TokenizerFast, T5ForConditionalGeneration
tokenizer = T5TokenizerFast.from_pretrained("paust/pko-chat-t5-large")
model = T5ForConditionalGeneration.from_pretrained("paust/pko-chat-t5-largee", device_map='cuda')

prompt_tpl = "사용자가 한 말을 읽고 그에 질문에 답하거나 명령에 응답하는 비서입니다.\n\n사용자:\n{text}\n\n비서:\n"
prompt = prompt_tpl.format(text="한국의 수도는 어디인가요?")
input_ids = tokenizer(prompt, return_tensors='pt').input_ids
logits = model.generate(
    input_ids,
    max_new_tokens=1024,
    temperature=0.5,
    no_repeat_ngram_size=6,
    do_sample=True,
    num_return_sequences=1,
)
text = tokenizer.batch_decode(logits, skip_special_tokens=True)[0]
print(text)  # 한국의 수도는 서울입니다.
```
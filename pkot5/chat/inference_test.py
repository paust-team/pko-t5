import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import safetensors

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context.\n"
        "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
        "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### Instruction(명령어):\n{instruction}\n\n### Input(입력):\n{input}\n\n### Response(응답):"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task.\n"
        "아래는 작업을 설명하는 명령어입니다.\n\n"
        "Write a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### Instruction(명령어):\n{instruction}\n\n### Response(응답):"
    ),
}


def gen(tokenizer, model: T5ForConditionalGeneration, prompt, max_new_tokens=128, temperature=0.5):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda:0')
    gen_tokens = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        temperature=temperature,
        no_repeat_ngram_size=6,
        do_sample=True,
    )
    gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)

    return gen_text


@torch.no_grad()
def main():
    model_name_or_path = "./pko-t5-large-chat"
    tokenizer = T5TokenizerFast.from_pretrained(model_name_or_path)
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map='cuda')

    print(model.generate(
        input_ids=tokenizer('안녕하세요? ', return_tensors='pt').to('cuda:0').input_ids,
        max_new_tokens=10,
    ))

    # Example usage:
    prompt_tpl = "사용자가 한 말을 읽고 그에 질문에 답하거나 명령에 응답하는 비서입니다.\n\n사용자:\n{text}\n\n비서:\n"
    prompt = prompt_tpl.format(text="파이썬으로 어떻게 현재 시간을 출력하는 프로그램의 코드를 작성해주세요.")
    generated_text = gen(tokenizer, model, prompt, max_new_tokens=1024)
    print(generated_text)
    print('---')

    prompt = prompt_tpl.format(text="한국전쟁은 언제 시작해서 언제 끝났나요?")
    generated_text = gen(tokenizer, model, prompt, max_new_tokens=1024)
    print(generated_text)
    print('---')

    prompt = prompt_tpl.format(text="한국의 수도는 어디인가요?")
    generated_text = gen(tokenizer, model, prompt, max_new_tokens=1024)
    print(generated_text)


if __name__ == '__main__':
    main()
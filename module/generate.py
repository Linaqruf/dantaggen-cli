from contextlib import nullcontext
from random import shuffle

import torch
from llama_cpp import Llama
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase


def generate(
    model: PreTrainedModel | Llama,
    tokenizer: PreTrainedTokenizerBase,
    prompt="",
    temperature=0.5,
    top_p=0.95,
    top_k=45,
    repetition_penalty=1.17,
    max_new_tokens=128,
    autocast_gen=lambda: torch.autocast("cpu", enabled=False),
    **kwargs,
):
    if isinstance(model, Llama):
        result = model.create_completion(
            prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            repeat_penalty=repetition_penalty or 1,
        )
        return prompt + result["choices"][0]["text"]

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        **kwargs,
    )
    with torch.no_grad(), autocast_gen(), torch.cuda.amp.autocast(enabled=False):
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output


def tag_gen(
    text_model,
    tokenizer,
    prompt,
    prompt_tags,
    len_target,
    black_list,
    temperature=0.5,
    top_p=0.95,
    top_k=100,
    max_new_tokens=256,
    max_retry=5,
):
    prev_len = 0
    retry = max_retry
    llm_gen = ""

    while True:
        llm_gen = generate(
            model=text_model,
            tokenizer=tokenizer,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=None,
            max_new_tokens=max_new_tokens,
            stream_output=False,
            autocast_gen=lambda: (
                torch.autocast("cuda") if torch.cuda.is_available() else nullcontext()
            ),
            prompt_lookup_num_tokens=10,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        llm_gen = llm_gen.replace("</s>", "").replace("<s>", "")
        extra = llm_gen.split("<|input_end|>")[-1].strip().strip(",")
        extra_tokens = [
            tok.strip() for tok in extra.split(",") if tok.strip() not in black_list
        ]
        extra_tokens = list(set(extra_tokens))
        llm_gen = llm_gen.replace(extra, ", ".join(extra_tokens))

        yield llm_gen, extra_tokens

        if len(prompt_tags) + len(extra_tokens) >= len_target:
            break

        if len(extra_tokens) == prev_len and prev_len > 0:
            retry -= 1
            if retry < 0:
                break
        else:
            retry = max_retry

        shuffle(extra_tokens)
        prev_len = len(extra_tokens)
        prompt = llm_gen.strip().replace("  <|", " <|")

    yield llm_gen, extra_tokens

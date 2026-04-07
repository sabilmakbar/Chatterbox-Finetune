import re
import torch


def split_text(text: str):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def generate_tts(model, text_input, gen_kwargs: dict = {}):
    return model.generate(text_input, **gen_kwargs)


def generate_tts_from_splits(model, text_input, silence_ms: int = 300, gen_kwargs: dict = {}):
    silence = torch.zeros(int(model.sr * silence_ms / 1000), dtype=torch.float32)
    chunks = []

    print("Text Input:", text_input)
    split_texts = split_text(text_input)
    total_splits = len(split_texts)

    for i, text in enumerate(split_texts, start=1):
        print(f"Segment {i} out of {total_splits}: Text to be generated: {text}")
        wav = generate_tts(model=model, text_input=text, gen_kwargs=gen_kwargs)

        if not isinstance(wav, torch.Tensor):
            wav = torch.tensor(wav)

        wav = wav.detach().cpu().float().flatten()
        chunks.append(wav)

        if i < len(split_texts):
            chunks.append(silence)

    return torch.cat(chunks, dim=0)

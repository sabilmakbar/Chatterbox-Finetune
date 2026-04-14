import re
import torch


# ===========================================================================
# simple regex-based sentence splitter
# Used by the original generate_tts_from_splits() below.
# Splits only on .!? — does not batch short sentences or handle abbreviations.
# ===========================================================================

def split_text(text: str):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_into_sentences(text: str) -> list[str]:
    """NLTK sent_tokenize with regex fallback (same regex as legacy split_text)."""
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except ImportError:
        return re.split(r"(?<=[.!?])\s+", text.strip())


def split_long_sentence(sentence: str, max_len: int = 300, seps=None) -> list[str]:
    """Recursively split a sentence that exceeds max_len on separator hierarchy."""
    if seps is None:
        seps = [";", ":", "-", ",", " "]

    sentence = sentence.strip()
    if len(sentence) <= max_len:
        return [sentence]

    if not seps:
        return [sentence[i : i + max_len].strip() for i in range(0, len(sentence), max_len)]

    sep = seps[0]
    parts = sentence.split(sep)
    if len(parts) == 1:
        return split_long_sentence(sentence, max_len, seps=seps[1:])

    chunks = []
    current = parts[0].rstrip()
    for part in parts[1:]:
        part = part.lstrip()
        candidate = current + sep + part
        if len(candidate) > max_len:
            if sep == " ":
                chunks.append(current + " ")
            else:
                chunks.extend(split_long_sentence(current, max_len, seps=seps[1:]))
            current = part
        else:
            current = candidate
    if current:
        if sep == " ":
            chunks.append(current)
        else:
            chunks.extend(split_long_sentence(current, max_len, seps=seps[1:]))

    return chunks


def group_sentences(sentences: list[str], max_chars: int = 150) -> list[str]:
    """Batch short sentences together up to max_chars to reduce model.generate() calls."""
    chunks = []
    current_chunk: list[str] = []
    current_length = 0

    for sentence in sentences:
        if not sentence:
            continue
        sentence = sentence.strip()
        sentence_len = len(sentence)

        if sentence_len > 300:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            for chunk in split_long_sentence(sentence, 300):
                if len(chunk) > max_chars:
                    for i in range(0, len(chunk), max_chars):
                        chunks.append(chunk[i : i + max_chars])
                else:
                    chunks.append(chunk)
            continue

        if sentence_len > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            chunks.append(sentence)
            current_chunk = []
            current_length = 0
        elif current_length + sentence_len + (1 if current_chunk else 0) <= max_chars:
            current_chunk.append(sentence)
            current_length += sentence_len + (1 if current_chunk else 0)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def enforce_min_chunk_length(chunks: list[str], min_len: int = 20, max_len: int = 300) -> list[str]:
    """Merge short chunks with their neighbour to avoid tiny TTS segments."""
    out = []
    i = 0
    while i < len(chunks):
        current = chunks[i].strip()
        if len(current) >= min_len or i == len(chunks) - 1:
            out.append(current)
            i += 1
        else:
            if i + 1 < len(chunks):
                merged = current + " " + chunks[i + 1]
                if len(merged) <= max_len:
                    out.append(merged)
                    i += 2
                else:
                    out.append(current)
                    i += 1
            else:
                out.append(current)
                i += 1
    return out


def normalize_text(
    text: str,
    max_chars: int = 150,
    min_chunk_length: int = 20,
    enable_batching: bool = True,
) -> list[str]:
    """
    Split and chunk text for TTS inference.

    Replaces the legacy split_text() with a pipeline that:
      1. Splits sentences via NLTK (or regex fallback)
      2. Batches short sentences up to max_chars
      3. Merges tiny leftover chunks

    Returns a list of text chunks ready for model.generate().
    """
    text = text.strip()
    if not text:
        raise ValueError("Input text cannot be empty")

    sentences = split_into_sentences(text)

    if enable_batching:
        chunks = group_sentences(sentences, max_chars=max_chars)
        chunks = enforce_min_chunk_length(chunks, min_len=min_chunk_length)
    else:
        chunks = [s.strip() for s in sentences if s.strip()]

    return chunks


# ===========================================================================
# Generation Functions
# ===========================================================================

def generate_tts(model, text_input, gen_kwargs: dict = {}):
    return model.generate(text_input, **gen_kwargs)


# uses split_text() (regex only, no grouping on short splits)
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


# uses normalize_text() (NLTK + batching + min-length grouping)
def generate_tts_normalized(model, text_input, silence_ms: int = 300, gen_kwargs: dict = {}):
    silence = torch.zeros(int(model.sr * silence_ms / 1000), dtype=torch.float32)
    chunks = []

    text_chunks = normalize_text(text_input)
    total = len(text_chunks)

    for i, text in enumerate(text_chunks, start=1):
        print(f"Segment {i} out of {total}: {text}")
        wav = generate_tts(model=model, text_input=text, gen_kwargs=gen_kwargs)

        if not isinstance(wav, torch.Tensor):
            wav = torch.tensor(wav)

        wav = wav.detach().cpu().float().flatten()
        chunks.append(wav)

        if i < total:
            chunks.append(silence)

    return torch.cat(chunks, dim=0)

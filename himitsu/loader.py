from typing import List, Tuple

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer


def load_model(language: str, device: str = "cpu") -> transformers.PreTrainedModel:
    if language == "en":
        model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
    elif language == "ja":
        model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
    elif language == "ru":
        model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
    else:
        raise ValueError(f"Invalid language: {language}")

    model.to(device)
    model.eval()
    return model


def load_tokenizer(language: str) -> Tuple[transformers.PreTrainedTokenizer, bool, List[str]]:
    if language == "en":
        tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
        special_tokens = ["<|endoftext|>", "\u010a", "\u010a\u010a", "\u010a\u00c2\u0142"]
    elif language == "ja":
        tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
        tokenizer.do_lower_case = True
        special_tokens = ["<unk>", "<s>", "</s>", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "▁"]
    elif language == "ru":
        tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
        special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<mask>", "Ċ", "ċ", "ĠĊ", "ÂłĊ"]
    else:
        raise ValueError(f"Invalid language: {language}")

    # gpt2-medium and rugpt3medium_based_on_gpt2 use byte-level vocab and need to be handled slight
    # differently when encoding/decoding
    byte_level_vocab = language in ["en", "ru"]

    return tokenizer, byte_level_vocab, special_tokens

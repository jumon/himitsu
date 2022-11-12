from typing import List

import torch
import torch.nn.functional as F
import transformers


def topk_tokens(
    tokenizer: transformers.PreTrainedTokenizer, count: int, indices: torch.Tensor
) -> List[str]:
    tokens = []
    for i in range(count):
        token_id = indices[i]
        token = tokenizer.convert_ids_to_tokens(token_id.item())
        tokens.append(token)
    return tokens


def resolve_collision(tokens: List[str]) -> List[int]:
    indices = []
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if i == j:
                continue
            if tokens[j].startswith(tokens[i]):
                break
        else:
            indices.append(i)
    return indices


def validate_secret(secret: str):
    for c in secret:
        if c not in {"0", "1"}:
            raise ValueError(
                f"Invalid character: {c} in secret. It should be a string of 0s and 1s."
            )


def supress_special_tokens(
    tokenizer: transformers.PreTrainedTokenizer, special_tokens: List[str], logits: torch.Tensor
) -> None:
    for special_token in special_tokens:
        id_ = tokenizer.convert_tokens_to_ids(special_token)
        logits[id_] = -1e12


def encode(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    secret: str,
    prompt: str = "Hi Bob.",
    min_prob: float = 0.01,
    special_tokens: List[str] = [],
    byte_level_vocab: bool = False,
) -> str:
    validate_secret(secret)

    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = input_ids.to(model.device)
    past_key_values = None
    generated_ids = torch.tensor([], dtype=torch.long, device=model.device)
    secret_index = 0

    while secret_index < len(secret):
        output = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        logits, past_key_values = output.logits, output.past_key_values  # logits: (1, Time, Vocab)
        logits = logits[0, -1, :]  # (Vocab,)
        supress_special_tokens(tokenizer, special_tokens, logits)
        probabilities = F.softmax(logits, dim=-1)
        probabilities, indices = torch.sort(probabilities, descending=True)  # (Vocab,), (Vocab,)

        candidate_count = max(torch.sum(probabilities >= min_prob).item(), 1)
        candidate_tokens = topk_tokens(tokenizer, candidate_count, indices)
        new_candidate_indices = resolve_collision(candidate_tokens)

        # bit_count is the largest n s.t. 2^n <= candidate_count
        bit_count = len(new_candidate_indices).bit_length() - 1

        if bit_count == 0:
            selected_token_id = indices[0]  # use the most probable token
            input_ids = selected_token_id.reshape(1, -1)
            generated_ids = torch.cat((generated_ids, selected_token_id.unsqueeze(0)))
            continue

        secret_to_encode = secret[secret_index : secret_index + bit_count]
        if len(secret_to_encode) < bit_count:
            secret_to_encode += "0" * (bit_count - len(secret_to_encode))
        selected_index = new_candidate_indices[int(secret_to_encode, base=2)]
        selected_token_id = indices[selected_index]

        input_ids = selected_token_id.reshape(1, -1)
        generated_ids = torch.cat((generated_ids, selected_token_id.unsqueeze(0)))
        secret_index += bit_count

    if byte_level_vocab:
        cover_text = tokenizer.decode(generated_ids, clean_up_tokenization_spaces=False)
    else:
        cover_text = "".join([t for t in tokenizer.convert_ids_to_tokens(generated_ids.tolist())])
        cover_text = cover_text.replace("▁", " ")
    return cover_text


def decode(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    cover_text: str,
    prompt: str = "Hi Bob.",
    min_prob: float = 0.01,
    special_tokens: List[str] = [],
    byte_level_vocab: bool = False,
) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = input_ids.to(model.device)
    past_key_values = None
    secret = ""
    current_index = 0

    if byte_level_vocab:
        cover_text = "".join(["".join(tokenizer.tokenize(c)) for c in cover_text])
    else:
        cover_text = cover_text.replace(" ", "▁")

    while current_index < len(cover_text):
        output = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        logits, past_key_values = output.logits, output.past_key_values  # logits: (1, Time, Vocab)
        logits = logits[0, -1, :]  # (Vocab,)
        supress_special_tokens(tokenizer, special_tokens, logits)
        probabilities = F.softmax(logits, dim=-1)
        probabilities, indices = torch.sort(probabilities, descending=True)  # (Vocab,), (Vocab,)

        candidate_count = max(torch.sum(probabilities >= min_prob).item(), 1)
        candidate_tokens = topk_tokens(tokenizer, candidate_count, indices)
        new_candidate_indices = resolve_collision(candidate_tokens)

        # bit_count is the largest n s.t. 2^n <= candidate_count
        bit_count = len(new_candidate_indices).bit_length() - 1

        if bit_count == 0:
            selected_token_id = indices[0]  # use the most probable token
            selected_token = tokenizer.convert_ids_to_tokens(selected_token_id.item())
            current_index += len(selected_token)
            input_ids = selected_token_id.reshape(1, -1)
            continue

        next_secret = None
        for i in range(2**bit_count):
            selected_idx = new_candidate_indices[i]
            selected_token_id = indices[selected_idx]
            selected_token = tokenizer.convert_ids_to_tokens(selected_token_id.item())
            if cover_text[current_index:].startswith(selected_token):
                next_secret = bin(i)[2:]
                next_secret = "0" * (bit_count - len(next_secret)) + next_secret
                secret += next_secret
                break

        if next_secret is None:
            raise ValueError("Decoding failed. Check if the parameters are the same as encoding.")

        input_ids = selected_token_id.reshape(1, -1)
        current_index += len(selected_token)

    return secret

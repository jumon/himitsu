import pytest

import himitsu


@pytest.mark.parametrize(
    ("language", "prompt", "cover_text", "num_trailing_zeros"),
    [
        ("en", "Hi Bob.", " It would appear there's nothing new.", 2),
        ("ja", "こんにちは", "、お元気様です、最近、寒暖差の", 0),
        ("ru", "Доброе утро", "! Как ваши выходные проходят, что успели за день?", 1),
    ],
)
def test_encode_and_decode(language: str, prompt: str, cover_text: str, num_trailing_zeros: int):
    model = himitsu.load_model(language)
    tokenizer, byte_level_vocab, special_tokens = himitsu.load_tokenizer(language)
    secret = "010101011111010101101010"
    min_prob = 0.01

    encoded = himitsu.encode(
        model=model,
        tokenizer=tokenizer,
        secret=secret,
        prompt=prompt,
        min_prob=min_prob,
        special_tokens=special_tokens,
        byte_level_vocab=byte_level_vocab,
    )
    assert encoded == cover_text

    decoded = himitsu.decode(
        model=model,
        tokenizer=tokenizer,
        cover_text=cover_text,
        prompt=prompt,
        min_prob=min_prob,
        special_tokens=special_tokens,
        byte_level_vocab=byte_level_vocab,
    )
    # There could be some trailing 0s in the decoded message.
    assert decoded == secret + "0" * num_trailing_zeros

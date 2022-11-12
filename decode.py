import argparse

import himitsu


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode cover text into a secret message.")
    parser.add_argument("cover_text", type=str, help="Cover text to decode.")
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "ja", "ru"],
        help="Language of the GPT-2 model",
    )
    parser.add_argument(
        "--prompt", type=str, default="Hi Bob.", help="A context prompt used to generate cover text"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.01,
        help="a minimum probability for a token to be a generated candidate",
    )
    return parser


def main():
    args = get_parser().parse_args()

    model = himitsu.load_model(args.language, args.device)
    tokenizer, byte_level_vocab, special_tokens = himitsu.load_tokenizer(args.language)

    decoded = himitsu.decode(
        model=model,
        tokenizer=tokenizer,
        cover_text=args.cover_text,
        prompt=args.prompt,
        min_prob=args.min_prob,
        special_tokens=special_tokens,
        byte_level_vocab=byte_level_vocab,
    )
    print(decoded)


if __name__ == "__main__":
    main()

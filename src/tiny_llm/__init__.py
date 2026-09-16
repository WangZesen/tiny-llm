"""Small, inspectable language models and reproducible training experiments."""


def main() -> None:
    import sys
    import time

    training = sys.argv[1:2] == ["train"]
    start = time.monotonic()
    if training:
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | INFO    | Startup Python entry",
            file=sys.stderr,
            flush=True,
        )
    from tiny_llm.cli import main as cli_main

    if training:
        print(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | INFO    | "
            f"Startup CLI imports: {time.monotonic() - start:.3f}s",
            file=sys.stderr,
            flush=True,
        )
    cli_main()

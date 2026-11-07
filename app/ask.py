"""第1回: 素のLLMに質問し、会話履歴を持って続けて問いかけるスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 既定質問を1回だけ実行する
    uv run python -m app.ask --chat     # 会話履歴を保ちながら複数質問を続ける

第1回では「ツールも会話履歴も持たない」単発質問だったが、宿題では
会話履歴を保持したまま次の質問に答えられるようにする。
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def _normalize_history(history):
    """会話履歴を Gemini の contents 形式へ変換する。

    受け入れる形式:
    - [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    - [("user", "..."), ("assistant", "...")]
    - [{"role": "user", "text": "..."}]
    """
    if not history:
        return []

    normalized = []
    for entry in history:
        if isinstance(entry, tuple) and len(entry) == 2:
            role, content = entry
        elif isinstance(entry, dict):
            role = entry.get("role")
            content = entry.get("content", entry.get("text"))
        else:
            continue

        if content is None:
            continue

        role_name = str(role or "user").lower()
        if role_name in {"assistant", "model", "ai"}:
            role_name = "model"
        elif role_name not in {"user", "model"}:
            role_name = "user"

        normalized.append({"role": role_name, "parts": [{"text": str(content)}]})

    return normalized


def ask(question: str, history=None) -> str:
    """質問をし、必要なら過去の会話履歴を続けてモデルに渡す。

    history には、以下の形式の要素を入れられる:
      - {"role": "user", "content": "..."}
      - {"role": "assistant", "content": "..."}
      - ("user", "...")
      - ("assistant", "...")
    """
    client = genai.Client(api_key=get_api_key())

    if history:
        contents = _normalize_history(history)
        contents.append({"role": "user", "parts": [{"text": question}]})
    else:
        contents = question

    response = client.models.generate_content(model=MODEL, contents=contents)

    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        response_tokens = None
        for attr in (
            "response_token_count",
            "candidates_token_count",
            "completion_token_count",
            "output_token_count",
        ):
            response_tokens = getattr(usage, attr, None)
            if response_tokens is not None:
                break

        if prompt_tokens is not None:
            print(f"入力トークン数: {prompt_tokens}")
        if response_tokens is not None:
            print(f"出力トークン数: {response_tokens}")

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--chat", action="store_true", help="会話履歴を保ちながら複数の質問を続ける")
    args = parser.parse_args()

    if args.chat:
        print(f"[model] {MODEL}")
        print("会話モードです。質問を入力してください。終了は 'exit' または Ctrl-D")
        history = []
        while True:
            try:
                question = input("質問> ").strip()
            except EOFError:
                print()
                break

            if not question:
                continue
            if question.lower() in {"exit", "quit", "q"}:
                break

            try:
                answer = ask(question, history)
            except errors.APIError as e:
                print(api_error_message(e.code, e.message))
                raise SystemExit(1)

            print(answer)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
        return

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

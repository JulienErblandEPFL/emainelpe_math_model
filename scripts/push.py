"""Upload a local merged-model directory to a Hugging Face Hub repo.

This is the push half of the former ``scripts/merge_and_push.py``. It
uploads ``--model-dir`` verbatim — weights, config, tokenizer, chat
template, and (required) ``generation_config.json`` — to the HF repo
given by ``--hf-repo``. It does NOT modify the checkpoint and does
NOT default to any team repo; the repo must be specified explicitly.

Usage::

    python scripts/push.py \\
        --model-dir <path/to/merged_model> \\
        --hf-repo   <your-org/your-repo>

Authentication: the script requires either the ``HF_TOKEN`` env var or
a prior ``huggingface-cli login``. Absence of both is detected up front
and the run aborts before any upload is attempted.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("push")


def default_commit_message() -> str:
    return "merged model checkpoint"


def verify_model_dir(model_dir: Path) -> None:
    """Ensure the directory exists and contains generation_config.json.

    The course CI requires generation_config.json to grade the model;
    refusing to push without it prevents the "uploaded but ungradeable"
    failure mode. Raises ``RuntimeError`` with a clear message.
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise RuntimeError(f"--model-dir does not exist: {model_dir}")
    if not (model_dir / "generation_config.json").is_file():
        raise RuntimeError(
            f"Missing generation_config.json under {model_dir}. "
            "The course CI requires this file; refusing to push an "
            "ungradeable checkpoint. Run scripts/merge.py first."
        )


def has_hf_auth() -> bool:
    """True iff HF_TOKEN is set OR a prior ``huggingface-cli login`` exists."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import HfFolder
    except ImportError:
        return False
    try:
        return bool(HfFolder.get_token())
    except Exception:
        return False


def push_to_hub(
    model_dir: Path, hf_repo: str, commit_message: str
) -> str:
    """Upload ``model_dir`` as a folder, then re-upload generation_config.json.

    The explicit re-upload guards against transformers version drift
    during folder upload silently rewriting generation_config.json.
    """
    from huggingface_hub import HfApi

    model_dir = Path(model_dir)
    api = HfApi()

    logger.info("Ensuring HF repo %s exists", hf_repo)
    api.create_repo(repo_id=hf_repo, exist_ok=True, repo_type="model")

    logger.info("Uploading folder %s to %s", model_dir, hf_repo)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=hf_repo,
        repo_type="model",
        commit_message=commit_message,
    )

    gen_cfg_path = model_dir / "generation_config.json"
    logger.info("Re-uploading %s explicitly", gen_cfg_path.name)
    api.upload_file(
        path_or_fileobj=str(gen_cfg_path),
        path_in_repo="generation_config.json",
        repo_id=hf_repo,
        repo_type="model",
        commit_message=f"{commit_message} (re-upload generation_config.json)",
    )

    return f"https://huggingface.co/{hf_repo}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload a local merged-model directory to a HF Hub repo.",
    )
    p.add_argument(
        "--model-dir", type=Path, required=True,
        help="Path to the local merged-model directory to upload.",
    )
    p.add_argument(
        "--hf-repo", required=True,
        help="HF repo to push to (e.g. 'your-org/your-repo'). No default; "
             "refusing to guess.",
    )
    p.add_argument(
        "--commit-message", default=None,
        help=f"HF commit message. Default: {default_commit_message()!r}.",
    )
    return p.parse_args(argv)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(levelname)s: %(message)s",
        stream=sys.stdout,
    )


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    args = _parse_args(argv)

    commit_message = args.commit_message or default_commit_message()

    # 1. Verify the directory and required files first — cheapest check.
    try:
        verify_model_dir(args.model_dir)
    except RuntimeError as e:
        logger.error("%s", e)
        return 2

    # 2. Auth preflight — fail before any network call.
    if not has_hf_auth():
        logger.error(
            "No Hugging Face credentials found. Set HF_TOKEN in the "
            "environment, or run `huggingface-cli login` first. "
            "Refusing to attempt upload."
        )
        return 3

    logger.info("model_dir      = %s", args.model_dir)
    logger.info("hf_repo        = %s", args.hf_repo)
    logger.info("commit_message = %s", commit_message)

    try:
        repo_url = push_to_hub(args.model_dir, args.hf_repo, commit_message)
    except Exception as e:
        logger.exception("HF push failed: %s", e)
        return 4

    print("\n" + "=" * 60)
    print("push: SUCCESS")
    print("=" * 60)
    print(f"  model_dir : {args.model_dir}")
    print(f"  hf_repo   : {args.hf_repo}")
    print(f"  url       : {repo_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

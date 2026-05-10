"""
Module entrypoint for the bam_processing CLI.
"""

from __future__ import annotations

import importlib.util
import importlib.metadata
import os


def _configure_jax_platform() -> None:
    """
    Normalize JAX platform selection and fail fast on unsupported Metal runs.
    """

    jax_platforms = os.environ.get("JAX_PLATFORMS")
    if not jax_platforms:
        return

    normalized_tokens: list[str] = []
    for platform in jax_platforms.split(","):
        normalized = platform.strip()
        if not normalized:
            continue
        if normalized.lower() == "metal":
            normalized_tokens.append("METAL")
        else:
            normalized_tokens.append(normalized.lower())

    normalized_platforms = ",".join(normalized_tokens)
    if not normalized_platforms:
        return

    os.environ["JAX_PLATFORMS"] = normalized_platforms

    requested_platforms = {
        platform.strip()
        for platform in normalized_platforms.split(",")
        if platform.strip()
    }
    if "METAL" not in requested_platforms:
        return

    try:
        importlib.metadata.version("jax-metal")
    except importlib.metadata.PackageNotFoundError:
        if importlib.util.find_spec("jax_plugins.metal_plugin") is not None:
            return
        raise SystemExit(
            "JAX_PLATFORMS requests 'METAL', but 'jax-metal' is not installed in "
            "the bam_processing environment. Install the macOS extra with "
            "`uv sync --project bam_processing --extra macos`, or unset "
            "JAX_PLATFORMS to let JAX choose an available backend."
        )


def main() -> None:
    """
    Run the bam_processing CLI.

    Returns:
        None:
            This function does not return a value.
    """

    _configure_jax_platform()

    from bam_processing.cli import cli

    cli()


if __name__ == "__main__":
    main()

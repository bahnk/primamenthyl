"""
Module entrypoint for the bam_processing CLI.
"""

from bam_processing.cli import cli


def main() -> None:
    """
    Run the bam_processing CLI.

    Returns:
        None:
            This function does not return a value.
    """

    cli()


if __name__ == "__main__":
    main()

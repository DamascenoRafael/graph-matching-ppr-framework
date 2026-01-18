import argparse
from pathlib import Path


def create_argparse() -> argparse.ArgumentParser:
    def is_valid_file(arg: str, flag: str, name: str) -> Path:
        filepath = Path(arg)
        if not filepath.is_file():
            raise argparse.ArgumentTypeError(f"argument {flag}/{name}: can't open '{arg}'")
        return filepath

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--edgelist-a",
        dest="edgelist_a",
        type=lambda arg: is_valid_file(arg, "-a", "--edgelist-a"),
        help="edgelist filepath for graph A",
        required=True,
        metavar="",
    )
    parser.add_argument(
        "-b",
        "--edgelist-b",
        dest="edgelist_b",
        type=lambda arg: is_valid_file(arg, "-b", "--edgelist-b"),
        help="edgelist filepath for graph B",
        required=True,
        metavar="",
    )
    parser.add_argument(
        "-o",
        "--progress-out",
        dest="progress_output",
        help="progress output filepath",
        default=None,
        metavar="",
    )
    parser.add_argument(
        "-m",
        "--matching-out",
        dest="matching_output",
        help="matching output filepath",
        default=None,
        metavar="",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="is_verbose",
        action="store_true",
        help="enable verbose output",
        default=False,
    )
    return parser

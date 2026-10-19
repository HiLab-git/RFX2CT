#!/usr/bin/env python3
"""Offline metric entry point for RFX2CT saved reconstructions."""

from common_metrics import build_arg_parser, run_cli


def main() -> None:
    parser = build_arg_parser("RFX2CT")
    args = parser.parse_args()
    run_cli("RFX2CT", args)


if __name__ == "__main__":
    main()

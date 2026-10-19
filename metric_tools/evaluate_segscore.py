#!/usr/bin/env python3
"""SegScore entry point for RFX2CT saved reconstructions."""

from common_segscore import build_segscore_arg_parser, run_segscore_cli


def main() -> None:
    parser = build_segscore_arg_parser("RFX2CT")
    args = parser.parse_args()
    run_segscore_cli("RFX2CT", args)


if __name__ == "__main__":
    main()

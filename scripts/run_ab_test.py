"""Deprecated CLI wrapper. Use `python -m undogmatic.eval_ab` instead."""

from __future__ import annotations

import sys

from undogmatic import eval_ab


def main() -> None:  # pragma: no cover - wrapper
    eval_ab.main(sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    main()

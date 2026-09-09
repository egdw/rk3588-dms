#!/usr/bin/env python3
"""根目录入口: 等价于 rk3588_dms/tools/inspect_model.py。

用法:
  python tools/model_inspect.py path/to/model.onnx
  python tools/model_inspect.py model.onnx --json
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TARGET = _ROOT / "rk3588_dms" / "tools" / "inspect_model.py"

if not _TARGET.exists():
    print(f"[FAIL] 找不到实现文件: {_TARGET}", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(_TARGET.parent))
import inspect_model  # noqa: E402

if __name__ == "__main__":
    sys.exit(inspect_model.main())

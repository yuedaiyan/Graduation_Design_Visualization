"""
将浮点数数组格式化为 InDesign 等宽对齐文本
正数前补空格，与负号对齐，每行 5 个数值，并保留向量里的逗号分隔

用法：
  python format_for_indesign.py input.txt output.txt

  input.txt  — 逗号分隔的浮点数（可跨行）
  output.txt — 格式化后的文本，直接粘贴进 InDesign

也可直接修改下方 INLINE_DATA 变量内联运行。
"""

import sys
import re
import unicodedata
from pathlib import Path

# ── 参数 ──────────────────────────────────────────────
COLS = 5  # 每行列数
DECIMALS = 17  # 保留小数位数
SEPARATOR = ""  # 列间距；默认只保留正数前的符号占位空格
# ─────────────────────────────────────────────────────

INLINE_DATA = ""  # 如不使用文件，把数据粘贴到这里的引号内

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "input.txt"
DEFAULT_OUTPUT = SCRIPT_DIR / "output.txt"


def clean_text(source: str) -> str:
    source = unicodedata.normalize("NFKC", source)
    source = "".join(ch for ch in source if unicodedata.category(ch) != "Cf")
    return (
        source.replace("−", "-").replace("–", "-").replace("—", "-").replace("＋", "+")
    )


def load_numbers(source: str) -> list[float]:
    source = clean_text(source)
    source = re.sub(r"(\d)\s+(\d)", r"\1\2", source)  # 修复 OCR 空格断字
    tokens = re.findall(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", source)
    numbers = []
    for t in tokens:
        numbers.append(float(t))
    return numbers


def format_numbers(numbers: list[float], cols: int, decimals: int, sep: str) -> str:
    lines = []
    for i, num in enumerate(numbers):
        # 正数前加空格占位，负数正常显示
        if num >= 0:
            formatted = f" {num:.{decimals}f}"
        else:
            formatted = f"{num:.{decimals}f}"

        # 除最后一个数外保留向量逗号。
        cell = formatted
        if i < len(numbers) - 1:
            cell += ","

        if i % cols == 0:
            lines.append([cell])
        else:
            lines[-1].append(cell)

    return "\n".join(sep.join(row) for row in lines)


def main():
    output_path = None

    if len(sys.argv) >= 2:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            raw = f.read()
    elif INLINE_DATA.strip():
        raw = INLINE_DATA
    elif DEFAULT_INPUT.exists():
        raw = DEFAULT_INPUT.read_text(encoding="utf-8")
        output_path = DEFAULT_OUTPUT
    else:
        print("用法: python format_for_indesign.py input.txt [output.txt]")
        print("或在脚本同目录放 input.txt，或在脚本中设置 INLINE_DATA 变量")
        sys.exit(1)

    numbers = load_numbers(raw)
    print(f"解析到 {len(numbers)} 个数值", file=sys.stderr)

    result = format_numbers(numbers, COLS, DECIMALS, SEPARATOR)

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])

    if output_path:
        output_path.write_text(result, encoding="utf-8")
        print(f"已写入 {output_path}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()

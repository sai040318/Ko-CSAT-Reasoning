#!/usr/bin/bash
set -euo pipefail

# jupytext로 .ipynb -> .py 변환 (셀 마커 유지).

usage() {
  echo "사용법: $(basename "$0") 입력.ipynb [출력.py]" >&2
  echo "예시: $(basename "$0") notebooks/example.ipynb" >&2
  echo "      $(basename "$0") notebooks/example.ipynb out/example.py" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "인자가 올바르지 않습니다." >&2
  usage
  exit 2
fi

input_path="$1"
if [[ ! -f "$input_path" ]]; then
  echo "입력 파일을 찾을 수 없습니다: $input_path" >&2
  exit 1
fi

if ! command -v jupytext >/dev/null 2>&1; then
  echo "jupytext가 필요합니다. 먼저 설치해 주세요." >&2
  echo "설치: uv add jupytext" >&2
  exit 1
fi

if [[ $# -eq 2 ]]; then
  output_path="$2"
  mkdir -p "$(dirname "$output_path")"
  echo "변환 중: $input_path -> $output_path"
  jupytext --to py "$input_path" -o "$output_path"
  echo "완료: $output_path"
else
  output_path="${input_path%.ipynb}.py"
  echo "변환 중: $input_path -> $output_path"
  jupytext --to py "$input_path"
  echo "완료: $output_path"
fi

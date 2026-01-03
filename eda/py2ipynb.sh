#!/usr/bin/bash
set -euo pipefail

# jupytext로 .py -> .ipynb 변환 (# %% 셀 마커 지원).

usage() {
  echo "사용법: $(basename "$0") 입력.py [출력.ipynb]" >&2
  echo "예시: $(basename "$0") scripts/example.py" >&2
  echo "      $(basename "$0") scripts/example.py out/example.ipynb" >&2
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

output_path="${2-}"

if ! command -v jupytext >/dev/null 2>&1; then
  echo "jupytext가 필요합니다. 먼저 설치해 주세요." >&2
  echo "설치: uv add jupytext" >&2
  exit 1
fi

if [[ -n "$output_path" ]]; then
  mkdir -p "$(dirname "$output_path")"
  echo "변환 중: $input_path -> $output_path"
  jupytext --to ipynb "$input_path" -o "$output_path"
  echo "완료: $output_path"
else
  output_path="${input_path%.py}.ipynb"
  echo "변환 중: $input_path -> $output_path"
  jupytext --to ipynb "$input_path"
  echo "완료: $output_path"
fi

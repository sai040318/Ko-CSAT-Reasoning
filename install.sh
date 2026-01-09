#!/bin/bash
set -e
chmod +x scripts/*.sh

echo "========================================================"
echo "   NLP for generation Project Setup (Simplified)"
echo "========================================================"

# 1. 디렉토리 구조 및 링크
bash scripts/01_setup_dirs.sh
echo ""

# 2. 공용 가상환경 생성
bash scripts/02_init_venv.sh
echo ""

# 3. .bashrc 설정
bash scripts/03_setup_env.sh
echo ""

echo "========================================================"
echo "🎉 설정 완료! 아래 명령어로 적용하세요:"
echo ""
echo "    source ~/.bashrc"
echo "========================================================"
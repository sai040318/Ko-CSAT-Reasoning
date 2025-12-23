#!/bin/bash
# [1] 디렉토리 및 심볼릭 링크 생성 (수정본: 루트 공유 방식)
set -e

SHARED_ROOT="/data/ephemeral/home/shared"
HOME_ROOT="/data/ephemeral/home"

echo ">>> [1/3] 공용 디렉토리 및 사용자별 공간 생성"

# 1. 공용 데이터셋 폴더 생성
mkdir -p "$SHARED_ROOT/data"

# 2. 데이터 링크 연결
if [ -L "./data" ]; then rm ./data; elif [ -d "./data" ]; then mv ./data ./data_backup; fi
ln -sfn "$SHARED_ROOT/data" ./data
echo "    ✅ ./data -> $SHARED_ROOT/data"
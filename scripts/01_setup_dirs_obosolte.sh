#!/usr/bin/bash
# [1] 디렉토리 및 심볼릭 링크 생성 (수정본: 루트 공유 방식)
set -e

SHARED_ROOT="/data/ephemeral/home/shared"
HOME_ROOT="/data/ephemeral/home"

# 사용자 목록
USERS=("dahyeong" "minseok" "taewon" "seunghwan" "junbeom" "sehun")

echo ">>> [1/3] 공용 디렉토리 및 사용자별 공간 생성"

# 1. 공용 데이터셋 폴더 생성
mkdir -p "$SHARED_ROOT/datasets/embeddings"

# 2. 사용자별 디렉토리 일괄 생성
#    (미리 만들어둬야 팀원들이 자기 방인 줄 알고 찾아들어갑니다)
echo "    - 사용자별 폴더 확인 및 생성 중..."
for USER in "${USERS[@]}"; do
    mkdir -p "$SHARED_ROOT/outputs/$USER"
    mkdir -p "$HOME_ROOT/$USER"
done
echo "    ✅ 모든 유저(6명)의 디렉토리 생성이 완료되었습니다."

echo ""
echo ">>> [2/3] 현재 프로젝트 심볼릭 링크 연결"

# 3. 데이터 링크 연결
if [ -L "./data" ]; then rm ./data; elif [ -d "./data" ]; then mv ./data ./data_backup; fi
ln -sfn "$SHARED_ROOT/datasets" ./data
echo "    ✅ ./data -> $SHARED_ROOT/datasets"

# 4. Outputs 링크 연결 (수정된 부분!)
#    개별 유저 폴더가 아니라, 'outputs' 전체를 연결합니다.
if [ -L "./outputs" ]; then rm ./outputs; elif [ -d "./outputs" ]; then mv ./outputs ./outputs_backup; fi
ln -sfn "$SHARED_ROOT/outputs" ./outputs
echo "    ✅ ./outputs -> $SHARED_ROOT/outputs (팀 전체 공유 공간)"

echo ""
echo "👉 [사용법 안내]"
echo "   내 모델 저장:   python train.py --output_dir ./outputs/{내이름}/실험명"
echo "   남 모델 사용:   python inference.py --model_name_or_path ./outputs/{팀원}/실험명"
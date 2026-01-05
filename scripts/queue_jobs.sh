#!/bin/bash

# set -e는 중간에 에러나면 스크립트를 멈추지만, 
# tsp에 작업을 '등록'하는 과정 자체는 에러가 날 일이 거의 없으므로 빼두는 게 유연합니다.

# 1. tsp 설치 확인
if ! command -v tsp &> /dev/null; then
    echo "tsp 명령어를 찾을 수 없습니다. 설치를 확인해 주세요."
    exit 1
fi

# 2. 인자 확인
if [ "$#" -eq 0 ]; then
    echo "사용법: ./queue_jobs.sh configs/*.yaml"
    exit 1
fi

# 3. 로그 폴더 생성
mkdir -p logs

# 4. 큐에 작업 추가 루프 (한 번만 돌립니다)
for yaml_path in "$@"; do
    yaml_name=$(basename "$yaml_path" .yaml)

    # tsp로 작업을 등록하면서 해당 작업의 ID를 변수에 저장합니다.
    # PYTHONUNBUFFERED=1을 넣어야 실시간으로 로그 파일에 기록됩니다.
    job_id=$(tsp bash -c "export PYTHONUNBUFFERED=1; uv run run_inference.py --config-name=$yaml_name")
    
    echo "Added to Queue: $yaml_name (ID: $job_id, Log: logs/${yaml_name}.log)"
done

echo "--------------------------------------------------"
echo "모든 작업이 큐에 등록되었습니다."
echo "가장 먼저 실행될 작업(ID: 0번 혹은 첫 번째 ID)의 출력을 모니터링합니다..."
echo "모니터링을 종료하려면 Ctrl+C를 누르세요. (실험은 중단되지 않습니다.)"
echo "--------------------------------------------------"

# 5. 가장 첫 번째 작업 혹은 현재 실행 중인 작업의 로그를 자동으로 보여줌
# 만약 이미 다른 프로세스가 GPU를 쓰고 있다면, 이 상태에서 대기하게 됩니다.
tsp -t
import time
import sys
import subprocess
import logging


def log_gpu_status(logger: logging.Logger = None):
    """
    현재 GPU 상태(메모리, 사용률 등)를 로그로 출력하는 함수.
    Args:
        logger: 사용할 logger 인스턴스. None이면 print로 출력.
    Returns:
        dict: GPU 상태 정보 (memory_used, memory_total, memory_free, gpu_util, name)
    """
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
        lines = result.strip().split("\n")
        gpu_info_list = []

        for idx, line in enumerate(lines):
            parts = [p.strip() for p in line.split(",")]
            gpu_info = {
                "gpu_index": idx,
                "name": parts[0],
                "memory_used": int(parts[1]),
                "memory_total": int(parts[2]),
                "memory_free": int(parts[3]),
                "gpu_util": int(parts[4]) if parts[4].isdigit() else 0,
            }
            gpu_info_list.append(gpu_info)

            log_msg = (
                f"[GPU {idx}] {gpu_info['name']} | "
                f"Memory: {gpu_info['memory_used']}/{gpu_info['memory_total']}MB "
                f"(Free: {gpu_info['memory_free']}MB) | "
                f"Util: {gpu_info['gpu_util']}%"
            )

            if logger:
                logger.info(log_msg)
            else:
                print(log_msg)

        return gpu_info_list[0] if len(gpu_info_list) == 1 else gpu_info_list

    except FileNotFoundError:
        err_msg = "nvidia-smi를 찾을 수 없습니다. NVIDIA 드라이버가 설치되어 있는지 확인하세요."
        if logger:
            logger.warning(err_msg)
        else:
            print(f"⚠️ {err_msg}")
        return None
    except Exception as e:
        err_msg = f"GPU 상태 확인 중 오류 발생: {e}"
        if logger:
            logger.warning(err_msg)
        else:
            print(f"⚠️ {err_msg}")
        return None


def wait_for_gpu_availability(threshold_mb=3000, check_interval=100):
    """
    GPU 메모리가 threshold_mb 이하로 떨어질 때까지 대기하는 함수.
    Args:
        threshold_mb (int): 이 용량보다 적게 사용 중이어야 실행 (기본 2GB)
        check_interval (int): 확인 주기 (초)
    """
    print(f"\n🛡️ [GPU Safety Guard] GPU 상태를 모니터링합니다... (기준: {threshold_mb}MB 미만)")

    while True:
        try:
            # nvidia-smi로 메모리 사용량 조회
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                encoding="utf-8",
            )
            # 첫 번째 GPU 메모리 파싱
            used_mem = int(result.strip().split("\n")[0])

            if used_mem < threshold_mb:
                print(f"✅ GPU 확보 완료! (현재 사용량: {used_mem}MB). 학습을 시작합니다.\n")
                break
            else:
                # 대기 메시지 (줄바꿈 없이 덮어쓰기)
                sys.stdout.write(f"\r⏳ 다른 팀원이 사용 중입니다... (현재: {used_mem}MB) - 대기 중...")
                sys.stdout.flush()
                time.sleep(check_interval)

        except Exception as e:
            print(f"\n⚠️ GPU 확인 중 에러 발생 (무시하고 진행): {e}")
            break

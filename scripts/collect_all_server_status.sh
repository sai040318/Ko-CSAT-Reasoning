#!/usr/bin/env bash
# collect_unsloth_status.sh
# 목적: Unsloth 설치 전 서버/컨테이너/GPU/파이썬(uv)/PyTorch 상태를 "개행/따옴표 이슈 없이" 한 번에 수집
# 포인트:
# - bash -lc "....\n...." 같은 형태를 제거(개행 깨짐 방지)
# - 멀티라인 코드는 파일(.sh/.py)로 저장 후 실행
# - 어떤 커맨드가 실패해도 스크립트 전체는 계속 진행(로그에 WARN 기록)

set -u
set -o pipefail

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUTDIR="${OUTDIR:-unsloth_env_audit_${TS}}"
mkdir -p "${OUTDIR}"

# python 선택 (python3 우선)
PYBIN="${PYBIN:-python3}"
command -v "${PYBIN}" >/dev/null 2>&1 || PYBIN="python"
command -v "${PYBIN}" >/dev/null 2>&1 || PYBIN=""

log() { echo "[$(date -u '+%F %T UTC')] $*" | tee -a "${OUTDIR}/_run.log" >/dev/null; }

run_cmd() {
  local name="$1"; shift
  local outfile="${OUTDIR}/${name}.txt"
  {
    echo "## ${name}"
    echo "## date_utc: $(date -u -Is)"
    echo "## cmd: $*"
    echo
    "$@"
  } >"${outfile}" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo >>"${outfile}"
    echo "[WARN] exit_code=${rc}" >>"${outfile}"
  fi
  return 0
}

run_sh_block() {
  # stdin으로 받은 bash 스크립트를 파일로 저장해 실행 (개행 안전)
  local name="$1"
  local shfile="${OUTDIR}/${name}.sh"
  local outfile="${OUTDIR}/${name}.txt"

  cat > "${shfile}"
  chmod +x "${shfile}"

  {
    echo "## ${name}"
    echo "## date_utc: $(date -u -Is)"
    echo "## script_file: ${shfile}"
    echo
    echo "----- SCRIPT BEGIN -----"
    sed -n '1,240p' "${shfile}"
    if [ "$(wc -l < "${shfile}")" -gt 240 ]; then
      echo "...(truncated in header; full script is saved in the .sh file)..."
    fi
    echo "----- SCRIPT END -----"
    echo
    bash "${shfile}"
  } >"${outfile}" 2>&1

  local rc=$?
  if [ $rc -ne 0 ]; then
    echo >>"${outfile}"
    echo "[WARN] exit_code=${rc}" >>"${outfile}"
  fi
  return 0
}

run_py_block() {
  # stdin으로 받은 python 코드를 파일로 저장해 실행 (개행 안전)
  local name="$1"
  local pyfile="${OUTDIR}/${name}.py"
  local outfile="${OUTDIR}/${name}.txt"

  if [ -z "${PYBIN}" ]; then
    {
      echo "## ${name}"
      echo "## date_utc: $(date -u -Is)"
      echo "[WARN] python/python3 not found"
    } > "${outfile}"
    return 0
  fi

  cat > "${pyfile}"

  {
    echo "## ${name}"
    echo "## date_utc: $(date -u -Is)"
    echo "## python: ${PYBIN}"
    echo "## script_file: ${pyfile}"
    echo
    echo "----- PY BEGIN -----"
    sed -n '1,260p' "${pyfile}"
    if [ "$(wc -l < "${pyfile}")" -gt 260 ]; then
      echo "...(truncated in header; full code is saved in the .py file)..."
    fi
    echo "----- PY END -----"
    echo
    "${PYBIN}" "${pyfile}"
  } >"${outfile}" 2>&1

  local rc=$?
  if [ $rc -ne 0 ]; then
    echo >>"${outfile}"
    echo "[WARN] exit_code=${rc}" >>"${outfile}"
  fi
  return 0
}

log "Start audit -> OUTDIR=${OUTDIR}"
log "PYBIN=${PYBIN:-<none>}"

# -------------------------
# Basic system & container
# -------------------------
run_cmd "00_time_utc" date -u -Is
run_cmd "01_whoami" whoami
run_cmd "02_id" id
run_cmd "03_hostname" hostname
run_cmd "04_uptime" uptime

run_cmd "10_os_release" cat /etc/os-release
run_cmd "11_uname" uname -a
run_cmd "12_proc_version" cat /proc/version

if command -v lsb_release >/dev/null 2>&1; then
  run_cmd "13_lsb_release" lsb_release -a
else
  echo "lsb_release: not installed" > "${OUTDIR}/13_lsb_release.txt"
fi

run_sh_block "20_container_check" <<'BASH'
set -u
test -f /.dockerenv && echo "/.dockerenv exists (likely docker)" || echo "no /.dockerenv"
echo
echo "--- /proc/1/cgroup ---"
cat /proc/1/cgroup 2>/dev/null || true
BASH

if command -v lscpu >/dev/null 2>&1; then run_cmd "30_cpu_lscpu" lscpu; else echo "lscpu: not installed" > "${OUTDIR}/30_cpu_lscpu.txt"; fi
run_cmd "31_mem_free" free -h
run_cmd "32_disk_df" df -hT
if command -v lsblk >/dev/null 2>&1; then run_cmd "33_disk_lsblk" lsblk; else echo "lsblk: not installed" > "${OUTDIR}/33_disk_lsblk.txt"; fi
run_sh_block "34_mount_head" <<'BASH'
mount | head -n 200
BASH

run_sh_block "40_glibc_ldd" <<'BASH'
ldd --version 2>/dev/null | head -n 40 || true
BASH
run_sh_block "41_gcc" <<'BASH'
gcc --version 2>/dev/null | head -n 40 || echo "gcc: not installed"
BASH
run_sh_block "42_gpp" <<'BASH'
g++ --version 2>/dev/null | head -n 40 || echo "g++: not installed"
BASH

# -------------------------
# NVIDIA / CUDA
# -------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  run_cmd "50_nvidia_smi" nvidia-smi
  # 매우 길 수 있으니 필요하면 환경변수로 끌 수 있게
  if [ "${NVIDIA_SMI_Q:-1}" = "1" ]; then
    run_cmd "51_nvidia_smi_q" nvidia-smi -q
  else
    echo "skipped (NVIDIA_SMI_Q=0)" > "${OUTDIR}/51_nvidia_smi_q.txt"
  fi
else
  echo "nvidia-smi: not found" > "${OUTDIR}/50_nvidia_smi.txt"
fi

run_sh_block "52_nvidia_driver_proc" <<'BASH'
cat /proc/driver/nvidia/version 2>/dev/null || echo "/proc/driver/nvidia/version not readable"
BASH

if command -v nvcc >/dev/null 2>&1; then
  run_cmd "53_nvcc" nvcc --version
else
  echo "nvcc: not found (often OK in containers; torch wheels may include CUDA runtime)" > "${OUTDIR}/53_nvcc.txt"
fi

run_sh_block "54_cuda_env" <<'BASH'
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "CUDA_PATH=${CUDA_PATH:-}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo
echo "--- PATH(head) ---"
echo "${PATH:-}" | tr ':' '\n' | head -n 80
BASH

# -------------------------
# Python / pip / uv
# -------------------------
if [ -n "${PYBIN}" ]; then
  run_sh_block "60_python" <<BASH
which ${PYBIN} 2>/dev/null || true
${PYBIN} -V || true
BASH

  run_sh_block "61_pip" <<BASH
${PYBIN} -m pip -V 2>/dev/null || true
pip -V 2>/dev/null || true
BASH
else
  echo "python/python3 not found" > "${OUTDIR}/60_python.txt"
  echo "python/python3 not found" > "${OUTDIR}/61_pip.txt"
fi

run_sh_block "62_uv" <<'BASH'
command -v uv >/dev/null 2>&1 && uv --version || echo "uv: not installed"
command -v uv >/dev/null 2>&1 && uv pip --version || true
BASH

# 패키지 목록은 커질 수 있어 옵션화
if [ "${PIP_LIST:-1}" = "1" ] && [ -n "${PYBIN}" ]; then
  run_sh_block "63_pip_list" <<BASH
${PYBIN} -m pip list 2>/dev/null || true
BASH
else
  echo "skipped (PIP_LIST=0 or no python)" > "${OUTDIR}/63_pip_list.txt"
fi

# -------------------------
# Torch / deps sanity check
# -------------------------
run_py_block "70_torch_check" <<'PY'
import os, sys, platform

print("python:", sys.version.replace("\n"," "))
print("platform:", platform.platform())

def try_import(name):
    try:
        m = __import__(name)
        return True, getattr(m, "__version__", "n/a")
    except Exception as e:
        return False, repr(e)

ok, ver = try_import("torch")
print("torch:", ok, ver)

if ok:
    import torch
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.backends.cudnn.version:", torch.backends.cudnn.version())
    print("cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print("cuda.device_count:", n)
        for i in range(n):
            prop = torch.cuda.get_device_properties(i)
            cc = f"{prop.major}.{prop.minor}"
            print(f"gpu[{i}] name={prop.name}")
            print(f"gpu[{i}] capability={cc}")
            print(f"gpu[{i}] total_vram_gb={prop.total_memory/1024**3:.2f}")
            try:
                free, total = torch.cuda.mem_get_info(i)
                print(f"gpu[{i}] mem_free_gb={free/1024**3:.2f} mem_total_gb={total/1024**3:.2f}")
            except Exception as e:
                print(f"gpu[{i}] mem_get_info failed: {e!r}")

for pkg in ["xformers","triton","bitsandbytes","transformers","accelerate","peft","trl","unsloth","unsloth_zoo"]:
    ok, ver = try_import(pkg)
    print(f"{pkg}:", ok, ver)

# 민감정보 방지: 토큰/키류는 값 노출 금지(경로형만 출력)
keys = ["CUDA_VISIBLE_DEVICES","NVIDIA_VISIBLE_DEVICES","HF_HOME","TRANSFORMERS_CACHE","TORCH_HOME"]
print("env(selected):")
for k in keys:
    print(f"  {k}={os.environ.get(k,'')}")
PY

# -------------------------
# Quick summary (개행 안전)
# -------------------------
run_sh_block "99_summary" <<BASH
echo "=== QUICK SUMMARY ==="
echo
echo "[OS]"
cat /etc/os-release 2>/dev/null || true
echo
echo "[NVIDIA]"
(nvidia-smi 2>/dev/null | head -n 40) || echo "nvidia-smi not available"
echo
echo "[PYTHON]"
if [ -n "${PYBIN}" ]; then ${PYBIN} -V || true; else echo "python not found"; fi
echo
echo "[TORCH]"
if [ -n "${PYBIN}" ]; then
  ${PYBIN} - <<'PY'
try:
    import torch
    print("torch", torch.__version__)
    print("torch.version.cuda", torch.version.cuda)
    print("cuda.is_available", torch.cuda.is_available())
except Exception as e:
    print("torch not usable:", repr(e))
PY
else
  echo "python not found"
fi
BASH

# -------------------------
# Archive
# -------------------------
tar -czf "${OUTDIR}.tar.gz" "${OUTDIR}" 2>/dev/null || true

log "Done."
log "Output directory: ${OUTDIR}"
log "Tarball: ${OUTDIR}.tar.gz"
echo "${OUTDIR}"

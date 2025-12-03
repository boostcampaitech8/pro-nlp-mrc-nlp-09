import time
import sys
import subprocess

def wait_for_gpu_availability(threshold_mb=3000, check_interval=10):
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
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            )
            # 첫 번째 GPU 메모리 파싱
            used_mem = int(result.strip().split('\n')[0])
            
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
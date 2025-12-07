import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    프로젝트 전역에서 사용할 공통 로거를 생성합니다.

    특징:
    1. 중복 핸들러 방지 (로그 중복 출력 해결)
    2. 상위 전파 방지 (Root 로거 중복 출력 해결)
    3. Colorlog 적용 (레벨에만 색상 적용, 가독성 향상)
    4. 포맷 정렬 (레벨 길이 고정으로 줄맞춤)
    """
    logger = logging.getLogger(name)

    # 1. [중복 방지] 이미 핸들러가 설정된 로거라면 기존 로거 반환
    # (주피터 노트북이나 반복 실행 환경에서 로그가 쌓이는 것 방지)
    if logger.handlers:
        return logger

    # 2. [전파 방지] 상위 로거(Root Logger)로 로그를 보내지 않음
    # (이 설정이 없으면 루트 로거 설정에 따라 로그가 두 번 찍힐 수 있음)
    logger.propagate = False

    logger.setLevel(level)

    # 3. [포맷터 설정] colorlog 설치 여부에 따라 분기 처리
    try:
        import colorlog

        # 레벨(INFO, ERROR 등) 부분에만 색상을 적용하고, 나머지는 기본색 유지
        formatter = colorlog.ColoredFormatter(
            fmt="%(asctime)s | %(log_color)s%(levelname)-8s%(reset)s | %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    except ImportError:
        # 라이브러리가 없을 경우 기본 포맷 사용
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
        )

    # 4. [핸들러 설정] 콘솔(Standard Output)에 출력
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

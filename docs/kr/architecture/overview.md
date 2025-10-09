# 프로젝트 개요

이 도구 프로젝트는 다양한 기능을 지원하는 여러 파일과 디렉터리로 구성되어 있습니다.  
새로운 단계(step), 그래프 수정(surgery), 또는 컴파일 로직을 추가하기 위해 *__플러그인 기반 아키텍처__*를 사용합니다.

| 디렉터리 / 파일 | 설명 |
|------------------|------|
| [`main.py`](../../../model_to_pipeline/main.py) | 도구의 메인 진입점(Main entry point)입니다. |
| [`cli.py`](../../../model_to_pipeline/cli.py) | 명령줄 인자(Command-line arguments)를 처리하고 `argparse.Namespace` 인스턴스를 반환합니다. |
| [`utils`](../../../model_to_pipeline/utils) | `loader`, 서브프로세스 유틸리티, 모델 그래프 수정을 위한 `onnx_helper.py` 등 보조 모듈을 포함합니다. 또한 모든 단계에서 사용되는 로깅 모듈이 포함되어 있습니다. |
| [`compilers`](../../../model_to_pipeline/compilers) | 지원되는 모델의 컴파일 로직이 포함되어 있습니다. |
| [`constants`](../../../model_to_pipeline/constants) | 프로젝트 전반에서 사용되는 내부 상수 값을 저장합니다. |
| [`pipeline`](../../../model_to_pipeline/pipeline) | 컴파일된 모델 `.tar.gz`를 사용하여 파이프라인을 생성합니다. 또한 `genericboxdecode`와 `overlay` 등의 플러그인을 해당 설정 파일과 함께 삽입합니다. RTSP 및 PCIe 파이프라인용 `application.json` 파일을 생성합니다. |
| [`surgeons`](../../../model_to_pipeline/surgeons) | 컴파일 전에 모델 그래프를 수정하기 위한 서전(surgeon) 스크립트를 포함합니다. |
| [`steps`](../../../model_to_pipeline/steps) | 모델 수정부터 최종 프로젝트 생성까지의 전체 워크플로우를 정의하는 실행 단계(step)들이 포함되어 있습니다. |

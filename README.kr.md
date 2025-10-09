# SiMa.ai 모델-파이프라인 변환 툴킷

[![SDK 호환성](https://img.shields.io/badge/SDK-1.7.0-blue.svg)](#)
[![모델](https://img.shields.io/badge/지원-YOLO-green.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-orange.svg)](#)
[![라이선스](https://img.shields.io/badge/License-MIT-lightgrey.svg)](../LICENSE)
[![English Docs](https://img.shields.io/badge/Docs-English-lightblue.svg)](README.md)

**Model-to-Pipeline Tool**은 FP32 YOLO 모델을 **SiMa.ai** 플랫폼용 최적화된 GStreamer 파이프라인으로 변환합니다.

이 가이드는 **Ultralytics YOLOv8m** 모델을 예시로 하여 SiMa.ai의 **Modalix Edge AI 플랫폼**에서 AI 추론 파이프라인을 직접 구축하고 실행하는 단계별 실습 안내서입니다.  이 문서를 따라가면 SDK 환경 설정부터 모델 변환, MPK 생성, Modalix에서의 실행까지 전 과정을 한 번에 체험할 수 있습니다.

---

### 📦 설치

[설치 가이드](docs/kr/installation.md)를 참고하세요.

### 🚀 사용 방법

- [Model-to-Pipeline 실행](docs/kr/usage/model-to-pipeline.md)
- [FPS 측정 (Get-FPS)](docs/kr/usage/get-fps.md)
- [추론 실행 (Infer)](docs/kr/usage/infer.md)

```
$ sima-model-to-pipeline --help

 Usage: sima-model-to-pipeline [OPTIONS] COMMAND [ARGS]...

╭─ Options ───────────────────────────────────────────────────────────────╮
│ --install-completion   Install completion for the current shell.        │
│ --show-completion      Show completion for the current shell, to copy   │
│                        it or customize the installation.                │
│ --help                 Show this message and exit.                      │
╰─────────────────────────────────────────────────────────────────────────╯

╭─ Commands ──────────────────────────────────────────────────────────────╮
│ model-to-pipeline   Convert a model into a working pipeline             │
│ get-fps             Find the MLA-only FPS using mla-rt                  │
│ infer               Create a pipeline for inference                     │
╰─────────────────────────────────────────────────────────────────────────╯
```

### 🧠 아키텍처 및 설계

이 도구의 내부 동작 방식을 알아보세요: 

- [프로젝트 개요](docs/kr/architecture/overview.md)
- [새 모델 추가하기](docs/kr/architecture/add-model.md)
- [새 단계 추가하기](docs/kr/architecture/add-step.md)
- [새 도구 추가하기](docs/kr/architecture/add-tool.md)

### 🧾 로그

[로그 개요](logs.md)를 참고하세요.
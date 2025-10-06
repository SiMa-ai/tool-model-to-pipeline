# 설치

## 사전 준비

이 도구를 설치하거나 실행하기 전에 다음 요구사항을 충족해야 합니다:

- 네트워크에 연결된 **SiMa.ai DevKit**  
- **Ubuntu 22.04** 환경의 개발용 PC  
- [Palette SDK](https://docs.sima.ai/pages/palette/main.html) 1.7 버전이 올바르게 설치되어 있을 것  
- 초기 설정 및 디버깅을 위해 [시리얼 연결](https://docs.sima.ai/pages/overview/setup_standalone_mode/setup_serial.html)이 미리 구성되어 있을 것  

## `sima-cli` 업데이트

설치를 진행하기 전에, [sima-cli가 최신 버전](https://docs.sima.ai/pages/sima_cli/main.html)인지 확인하세요.

## 도구 설치

**Palette SDK** 환경 내에서 다음 명령어로 설치할 수 있습니다:

```bash
sima-cli install gh:sima-ai/tool-model-to-pipeline
source ~/.bashrc
```

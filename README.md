# InpaLLa: mLLM 아키텍처를 활용한 이미지 인페인팅 도구

![banner](https://github.com/your-repo/banner-image-url)

안녕하세요! **InpaLLa**에 오신 것을 환영합니다. InpaLLa는 텍스트 프롬프트를 기반으로 이미지를 창의적으로 수정할 수 있는 도구입니다. 사용자 친화적인 인터페이스와 강력한 기능을 통해 손쉽게 원하는 이미지 변환을 경험해보세요.

## 팀 정보
- **홍길동**
  - 서울대학교 컴퓨터공학과 학사
  - 서울대학교 인공지능 대학원 석사
- **김영희**
  - 서울대학교 환경공학과 학사
  - 서울대학교 통계학과 석사

## 업데이트
- **2024.04.27**: 초기 버전 릴리즈
- **2024.05.10**: FLUX, LISA, VLLM 모델 통합
- **2024.06.15**: 포스터 및 발표자료 업로드 기능 추가

## 주요 기능
- **직관적인 UI**: Gradio 기반의 간단하고 깔끔한 인터페이스
- **자동 사용자 관리**: 고유한 사용자 ID 생성으로 데이터 보안 강화
- **이미지 최적화**: 업로드된 이미지를 자동으로 최대 1024x1024 픽셀로 조정
- **고품질 인페인팅**: FLUX, LISA, VLLM 모델을 연계하여 뛰어난 결과물 제공
- **파라미터 조정**: 디노이징 강도, 추론 단계 수, 가이드 스케일 등 사용자 맞춤 설정 가능
- **결과 시각화**: 인페인팅된 이미지와 마스크 이미지를 함께 확인

## 시스템 요구 사항
- **OS**: Ubuntu 20.04
- **Python**: 3.8 이상
- **CUDA**: 11.4
- **NVIDIA 드라이버**: 470.82.01 이상
- **GPU**: NVIDIA GeForce RTX 3090 (24GB 권장)
- **랜덤 시드**: 605

## 설치 방법
1. **필요한 패키지 설치**
    ```bash
    pip install requests gradio diffusers Pillow
    ```
2. **스크립트 실행**
    원하는 디렉토리에 스크립트를 저장한 후, 다음 명령어로 실행하세요.
    ```bash
    python your_script.py
    ```

## 사용 방법
1. **웹 인터페이스 접속**: 스크립트를 실행하면 로컬 호스트에서 웹 인터페이스가 열립니다.
2. **이미지 업로드**: 수정하고 싶은 이미지를 업로드하세요.
3. **프롬프트 입력**: 원하는 이미지 변환 내용을 텍스트로 입력합니다.
4. **파라미터 설정**: 디노이징 강도, 추론 단계 수, 가이드 스케일 등을 조정합니다.
5. **인페인팅 실행**: "Inpainting" 버튼을 클릭하여 처리를 시작합니다.
6. **결과 확인**: 인페인팅된 이미지와 마스크 이미지를 확인할 수 있습니다.

## 디렉토리 구조
- **`./data/{user_id}/`**
  - **`{user_id}_image.jpg`**: 업로드된 원본 이미지
  - **`inpainted_image.jpg`**: 인페인팅된 최종 이미지
  - **`masked_img.jpg`**: 생성된 세그멘테이션 마스크 이미지

## 결과
![결과 이미지1](https://github.com/your-repo/result-image1-url)
![결과 이미지2](https://github.com/your-repo/result-image2-url)
<br/>
(Loss : BCELogitLoss / Evaluation Score : mAP)

## 레퍼런스
- FLUX: [FLUX 모델 설명](https://flux-model-link.com)
- LISA: [LISA 모델 설명](https://lisa-model-link.com)
- VLLM: [VLLM 모델 설명](https://vllm-model-link.com)
- TAN, Mingxing; LE, Quoc. Efficientnet: Rethinking model scaling for convolutional neural networks. In: International conference on machine learning. PMLR, 2019.
- Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international conference on computer vision. 2017.
- Foret, Pierre, et al. "Sharpness-aware minimization for efficiently improving generalization." arXiv preprint arXiv:2010.01412 (2020).
- Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization." arXiv preprint arXiv:1710.09412 (2017).

## 포스터 및 발표자료
포스터와 발표자료는 아래 링크에서 다운로드할 수 있습니다:

- [포스터 다운로드](https://github.com/your-repo/poster.pdf)
- [발표자료 다운로드](https://github.com/your-repo/presentation.pdf)

## 기여 방법
InpaLLa는 오픈 소스 프로젝트로, 여러분의 많은 기여를 환영합니다! 버그 리포트, 기능 제안, 코드 기여 등 다양한 방법으로 참여해 주세요.

1. **포크(Fork)**: 저장소를 포크합니다.
2. **브랜치 생성**: 새로운 기능이나 버그 수정을 위한 브랜치를 만듭니다.
3. **커밋**: 변경 사항을 커밋합니다.
4. **풀 리퀘스트**: 원본 저장소로 풀 리퀘스트를 보냅니다.

## 라이선스
이 프로젝트는 [MIT 라이선스](LICENSE)를 따릅니다. 자유롭게 사용하고 수정할 수 있습니다.

## 문의
문의 사항이나 도움이 필요하시면 [이슈 트래커](https://github.com/your-repo/issues)를 통해 알려주세요. 언제든지 도와드리겠습니다!

감사합니다. 즐거운 이미지 인페인팅 경험 되세요! 😊

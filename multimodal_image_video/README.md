# Multimodal Image to Video

Amazon Bedrock을 활용하여 이미지를 생성하고, SageMaker에 배포된 Image-to-Video 모델을 통해 생성된 이미지를 기반으로 짧은 비디오로 생성합니다.

### `1-mm-prompt-to-image.ipynb`

- 새롭게 출시 된 Stable Image Ultra, Stable Diffusion 3 Large, Stable Image Core 모델을 활용한 이미지 생성 예제를 포함합니다.
- 이미지 생성을 위한 프롬프트를 쉽게 만들기 위해, Claude 3.5 Sonnet을 활용해 multimodal 정보를 기반으로 이미지 생성 프롬프트를 제작합니다.

### `2-deploy-svd-xt.ipynb`

- Stable Video Diffusion 모델을 활용하기 위해 SageMaker Asynchronous Inference Endpoint를 구성 및 생성합니다.
- 이 작업은 약 1시간 소요됩니다.

### `3-inference.ipynb`

- 배포된 Stable Video Diffusion 모델을 통해 Image-to-Video 테스트를 수행합니다.

### `4-clean-up.ipynb`

- 배포한 SageMaker Endpoint를 삭제합니다.
- S3에 저장된 모델과 inference 결과는 별도 삭제 필요합니다.

## Reference Link

- [svdxt-sagemaker-huggingface](https://github.com/garystafford/svdxt-sagemaker-huggingface)
- [HuggingFace](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)

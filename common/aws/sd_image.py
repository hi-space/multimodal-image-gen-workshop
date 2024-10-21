import boto3
import json
import secrets
from io import BytesIO
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
from botocore.config import Config


class SDSampler(Enum):
    DDIM = "DDIM"
    DDPM = "DDPM"
    K_DPMPP_2M = "K_DPMPP_2M"
    K_DPMPP_2S_ANCESTRAL = "K_DPMPP_2S_ANCESTRAL"
    K_DPM_2 = "K_DPM_2"
    K_DPM_2_ANCESTRAL = "K_DPM_2_ANCESTRAL"
    K_EULER = "K_EULER"
    K_EULER_ANCESTRAL = "K_EULER_ANCESTRAL"
    K_HEUN = "K_HEUN"
    K_LMS = "K_LMS"

class SDClipPreset(Enum):
    FAST_BLUE = "FAST_BLUE"
    FAST_GREEN = "FAST_GREEN"
    NONE = "NONE"
    SIMPLE = "SIMPLE"
    SLOW = "SLOW"
    SLOWER = "SLOWER"
    SLOWEST = "SLOWEST"


class StylePreset(Enum):
    THREE_D_MODEL = "3d-model"
    ANALOG_FILM = "analog-film"
    ANIME = "anime"
    CINEMATIC = "cinematic"
    COMIC_BOOK = "comic-book"
    DIGITAL_ART = "digital-art"
    ENHANCE = "enhance"
    FANTASY_ART = "fantasy-art"
    ISOMETRIC = "isometric"
    LINE_ART = "line-art"
    LOW_POLY = "low-poly"
    MODELING_COMPOUND = "modeling-compound"
    NEON_PUNK = "neon-punk"
    ORIGAMI = "origami"
    PHOTOGRAPHIC = "photographic"
    PIXEL_ART = "pixel-art"
    TILE_TEXTURE = "tile-texture"

class ImageSize(Enum):
    SIZE_512x512 = (512, 512)   # 1:1
    SIZE_1024x1024 = (1024, 1024)   # 1:1
    SIZE_1152x896 = (1152, 896)     # 9:7
    SIZE_896x1152 = (896, 1152)     # 7:9
    SIZE_1216x832 = (1216, 832)
    SIZE_1344x768 = (1344, 768)
    SIZE_768x1344 = (768, 1344)
    SIZE_1536x640 = (1536, 640)

    def __init__(self, width, height):
        self.width = width
        self.height = height


@dataclass
class TextPrompt:
    text: str  # 프롬프트 텍스트 (최대 2000자)
    weight: Optional[float] = None  # 프롬프트 가중치 (음수로 설정하면 negative prompt)

    def to_jsonstr(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class SDT2IRequest:
    text_prompts: List[TextPrompt]  # 프롬프트 텍스트 배열 (비어있지 않아야 함)
    height: Optional[int] = 512  # 이미지의 높이 (64의 배수이며 최소 128)
    width: Optional[int] = 512  # 이미지의 너비 (64의 배수이며 최소 128)
    cfg_scale: Optional[float] = 7.0  # 프롬프트에 대한 이미지 정확도 (0에서 35 사이)
    clip_guidance_preset: Optional[SDClipPreset] = "NONE"  # Clip Guidance 프리셋 (FAST_BLUE, FAST_GREEN, NONE 등)
    sampler: Optional[SDSampler] = None  # 샘플러 (DDIM, DDPM, K_DPMPP_2M 등)
    samples: Optional[int] = 1  # 생성할 이미지 수 (1에서 10 사이)
    seed: Optional[int] = 0  # 랜덤 시드 값 (0을 사용하면 랜덤 시드)
    steps: Optional[int] = 30  # 디퓨전 스텝 수 (10에서 50 사이)
    style_preset: Optional[StylePreset] = None  # 스타일 프리셋 (3d-model, anime, cinematic 등)

    def to_jsonstr(self) -> str:
        return json.dumps(asdict(self), default=lambda o: o.value if isinstance(o, Enum) else o)


@dataclass
class SDI2IRequest:
    text_prompts: List[TextPrompt]  # 인덱스된 텍스트 프롬프트 배열
    init_image: Optional[str] = None  # 시작 이미지
    init_image_mode: Optional[str] = "IMAGE_STRENGTH"  # 이미지 강도 모드 (IMAGE_STRENGTH, STEP_SCHEDULE)
    image_strength: Optional[float] = 0.35  # 이미지 강도 (0에서 1 사이)
    cfg_scale: Optional[float] = 7.0  # 프롬프트에 따른 이미지 정확도 (0에서 35 사이)
    clip_guidance_preset: Optional[SDClipPreset] = SDClipPreset.NONE  # Clip Guidance 프리셋 (Enum 사용)
    sampler: Optional[SDSampler] = None  # 샘플러 (DDIM, K_EULER 등)
    samples: Optional[int] = 1  # 생성할 이미지 수 (1에서 10 사이)
    seed: Optional[int] = 0  # 랜덤 시드 값 (0을 사용하면 랜덤 시드)
    steps: Optional[int] = 30  # 디퓨전 스텝 수 (10에서 50 사이)
    style_preset: Optional[StylePreset] = None  # 스타일 프리셋 (3d-model, anime, etc.)

    def to_jsonstr(self) -> str:
        return json.dumps(asdict(self), default=lambda o: o.value if isinstance(o, Enum) else o)


class BedrockStableDiffusion():
    def __init__(self, region='us-west-2', modelId = 'stability.stable-diffusion-xl-v1'):
        self.region = region
        self.modelId = modelId
        self.bedrock = boto3.client(
            service_name = 'bedrock-runtime',
            region_name = self.region,
            config = Config(
                connect_timeout=120,
                read_timeout=120,
                retries={'max_attempts': 5}
            ),
        )

    def text_to_image(self, request: SDT2IRequest):
        response = self.bedrock.invoke_model(body=request, modelId=self.modelId)
        response_body = json.loads(response.get("body").read())
        return response_body["artifacts"][0].get("base64")

    def image_to_image(self, request: SDI2IRequest):
        response = self.bedrock.invoke_model(body=request, modelId=self.modelId)
        response_body = json.loads(response.get("body").read())
        return response_body["artifacts"][0].get("base64")


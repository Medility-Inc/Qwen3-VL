"""
vLLM FastAPI Inference Server for Qwen3-VL-32B-Thinking
포트 1119에서 실행되는 Vision-Language 모델 추론 서버
"""

import os
import sys
import base64
import logging
from io import BytesIO
from typing import List, Optional, Dict, Any, Union

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
import uvicorn

# vLLM 및 qwen-vl-utils import
from vllm import SamplingParams, LLM
from transformers import AutoProcessor

# qwen-vl-utils 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'qwen-vl-utils', 'src'))
from qwen_vl_utils import process_vision_info

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="Qwen3-VL vLLM Inference Server",
    description="Vision-Language Model Inference API",
    version="1.0.0"
)

# 글로벌 변수
model: Optional[LLM] = None
processor: Optional[AutoProcessor] = None


class ImageInput(BaseModel):
    """이미지 입력 형식"""
    type: str = Field(default="image", description="입력 타입 (image)")
    image: str = Field(..., description="이미지 URL, 로컬 경로, 또는 base64 인코딩된 이미지")
    min_pixels: Optional[int] = Field(None, description="최소 픽셀 수")
    max_pixels: Optional[int] = Field(None, description="최대 픽셀 수")


class GenerateRequest(BaseModel):
    """생성 요청 형식"""
    text: str = Field(..., description="텍스트 프롬프트")
    images: Optional[List[ImageInput]] = Field(None, description="이미지 리스트")
    max_tokens: int = Field(1024, description="생성할 최대 토큰 수")
    temperature: float = Field(0.7, description="샘플링 temperature")
    top_p: float = Field(0.8, description="Top-p 샘플링")
    system_prompt: Optional[str] = Field(None, description="시스템 프롬프트")


class GenerateResponse(BaseModel):
    """생성 응답 형식"""
    text: str = Field(..., description="생성된 텍스트")
    model: str = Field(..., description="사용된 모델 이름")


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str
    model_loaded: bool
    model_name: str


def initialize_model(
    model_name: str = "Qwen/Qwen3-VL-8B-Thinking",
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: Optional[int] = None,
    max_model_len: int = 32768
) -> tuple[LLM, AutoProcessor]:
    """
    vLLM 모델과 프로세서 초기화
    
    Args:
        model_name: HuggingFace 모델 이름
        gpu_memory_utilization: GPU 메모리 사용률
        tensor_parallel_size: Tensor parallel 크기 (None이면 자동 감지)
        max_model_len: 최대 모델 길이 (KV cache 메모리 제약 고려)
    
    Returns:
        (model, processor) 튜플
    """
    logger.info(f"모델 초기화 시작: {model_name}")
    logger.info(f"최대 모델 길이: {max_model_len}")
    
    # vLLM 멀티프로세싱 설정
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    
    # Tensor parallel 크기 자동 감지
    if tensor_parallel_size is None:
        tensor_parallel_size = torch.cuda.device_count()
        logger.info(f"GPU 자동 감지: {tensor_parallel_size}개")
    
    # vLLM 모델 초기화
    model = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        seed=0
    )
    
    # 프로세서 로드
    processor = AutoProcessor.from_pretrained(model_name)
    
    logger.info("모델 초기화 완료")
    return model, processor


def prepare_inputs_for_vllm(
    messages: List[Dict[str, Any]],
    processor: AutoProcessor
) -> Dict[str, Any]:
    """
    vLLM 추론을 위한 입력 준비
    
    Args:
        messages: 대화 메시지 리스트
        processor: AutoProcessor 인스턴스
    
    Returns:
        vLLM 입력 딕셔너리
    """
    # 채팅 템플릿 적용
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 비전 정보 처리
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    
    # 멀티모달 데이터 준비
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs
    
    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


def build_messages(
    text: str,
    images: Optional[List[ImageInput]] = None,
    system_prompt: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    API 요청을 모델 입력 메시지 형식으로 변환
    
    Args:
        text: 사용자 텍스트
        images: 이미지 리스트
        system_prompt: 시스템 프롬프트
    
    Returns:
        메시지 리스트
    """
    messages = []
    
    # 시스템 프롬프트 추가
    if system_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })
    
    # 사용자 메시지 구성
    content = []
    
    # 이미지 추가
    if images:
        for img in images:
            img_dict = {"type": "image", "image": img.image}
            if img.min_pixels is not None:
                img_dict["min_pixels"] = img.min_pixels
            if img.max_pixels is not None:
                img_dict["max_pixels"] = img.max_pixels
            content.append(img_dict)
    
    # 텍스트 추가
    content.append({"type": "text", "text": text})
    
    messages.append({
        "role": "user",
        "content": content
    })
    
    return messages


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    global model, processor
    try:
        model, processor = initialize_model()
        logger.info("서버 준비 완료")
    except Exception as e:
        logger.error(f"모델 초기화 실패: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스체크 엔드포인트"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_name="Qwen/Qwen3-VL-8B-Thinking"
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    텍스트 생성 엔드포인트
    
    Args:
        request: GenerateRequest 객체
    
    Returns:
        GenerateResponse 객체
    """
    global model, processor
    
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
    
    try:
        # 메시지 구성
        messages = build_messages(
            text=request.text,
            images=request.images,
            system_prompt=request.system_prompt
        )
        
        logger.info(f"생성 요청: {len(messages)} 메시지")
        
        # vLLM 입력 준비
        inputs = prepare_inputs_for_vllm(messages, processor)
        
        # 샘플링 파라미터 설정
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # 추론 실행
        outputs = model.generate(inputs, sampling_params=sampling_params)
        
        # 결과 추출
        generated_text = ""
        for output in outputs:
            for completion in output.outputs:
                generated_text += completion.text
        
        logger.info(f"생성 완료: {len(generated_text)} 문자")
        
        return GenerateResponse(
            text=generated_text,
            model="Qwen/Qwen3-VL-8B-Thinking"
        )
    
    except Exception as e:
        logger.error(f"생성 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Qwen3-VL vLLM Inference Server",
        "version": "1.0.0",
        "model": "Qwen/Qwen3-VL-8B-Thinking",
        "endpoints": {
            "health": "/health",
            "generate": "/generate (POST)",
            "docs": "/docs"
        }
    }


def main():
    """메인 함수"""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=1119,
        log_level="info"
    )


if __name__ == "__main__":
    main()


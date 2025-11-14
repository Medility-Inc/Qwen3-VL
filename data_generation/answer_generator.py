"""
답변 생성 모듈
검수 통과한 질문에 대해 답변을 생성합니다.
"""

import os
import logging
import requests
import re
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

import yaml
from openai import OpenAI

# 환경변수 로드
load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """답변 생성 클래스"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.api_config = self.config["api"]
        
        # OpenAI 클라이언트 초기화
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Qwen3-VL API URL
        self.qwen3vl_url = f"{self.api_config['qwen3vl']['base_url']}/generate"
        
        logger.info("답변 생성기 초기화 완료")
    
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        config_file = Path(__file__).parent / config_path
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _clean_answer(self, answer: str) -> str:
        """
        답변에서 영어 reasoning 부분을 제거하고 한국어 답변만 추출
        
        Args:
            answer: 원본 답변
            
        Returns:
            정제된 한국어 답변
        """
        if not answer:
            return ""
        
        # </think> 또는 </think> 태그 이후의 내용만 추출 (Qwen3-VL Thinking 모델의 reasoning 태그 제거)
        for tag in ["</think>", "</think>"]:
            if tag in answer:
                parts = answer.split(tag)
                if len(parts) > 1:
                    answer = parts[-1].strip()
                    break
        
        # 영어로 시작하는 reasoning 패턴 제거
        # "So, let's", "Wait,", "Got it,", "Let me", "I need to" 등으로 시작하는 문장 제거
        lines = answer.split('\n')
        cleaned_lines = []
        skip_until_korean = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 영어 reasoning 패턴 감지
            english_reasoning_patterns = [
                r'^(So,|Wait,|Got it,|Let me|I need to|First,|Then,|But|However|Therefore|Thus|In the image|Looking at|Let\'s|I can see|The image shows)',
                r'^(So|Wait|Got it|Let me|I need|First|Then|But|However|Therefore|Thus)',
            ]
            
            is_english_reasoning = False
            for pattern in english_reasoning_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_english_reasoning = True
                    skip_until_korean = True
                    break
            
            # 한국어가 포함되어 있는지 확인
            has_korean = bool(re.search(r'[가-힣]', line))
            
            # 영어만 있고 한국어가 없는 경우 제외 (reasoning일 가능성)
            if not has_korean and re.match(r'^[A-Za-z0-9\s\.,:;!?\-\(\)\[\]\"\']+$', line):
                if skip_until_korean:
                    continue
            
            # 한국어가 포함된 경우 reasoning 모드 종료
            if has_korean:
                skip_until_korean = False
                cleaned_lines.append(line)
            elif not skip_until_korean and line:
                # 한국어가 없지만 reasoning 모드가 아닌 경우 (숫자, 기호 등)
                cleaned_lines.append(line)
        
        cleaned_answer = '\n'.join(cleaned_lines).strip()
        
        # 최종 검증: 한국어가 전혀 없는 경우 원본 반환 (숫자만 있는 경우 등)
        if not bool(re.search(r'[가-힣]', cleaned_answer)) and bool(re.search(r'[가-힣]', answer)):
            # 원본에서 한국어 부분만 추출
            korean_parts = []
            for line in answer.split('\n'):
                if re.search(r'[가-힣]', line):
                    korean_parts.append(line.strip())
            if korean_parts:
                cleaned_answer = '\n'.join(korean_parts).strip()
        
        return cleaned_answer if cleaned_answer else answer
    
    def generate_with_qwen3vl(self, question: str, image_path: str) -> str:
        """
        Qwen3-VL-8B-Thinking을 사용하여 답변 생성
        
        Args:
            question: 질문
            image_path: 이미지 경로
        
        Returns:
            생성된 답변
        """
        logger.debug(f"Qwen3-VL로 답변 생성: {question[:50]}...")
        
        # 이미지 절대 경로 변환
        if Path(image_path).is_absolute():
            abs_image_path = Path(image_path)
        else:
            abs_image_path = Path(__file__).parent / image_path
        
        if not abs_image_path.exists():
            logger.error(f"이미지 파일을 찾을 수 없습니다: {abs_image_path}")
            return ""
        
        prompt = f"""다음 질문에 대해 이미지를 보고 정확하고 명확하게 답변해주세요.

질문: {question}

답변 지침:
- **반드시 한국어로만 답변하세요. 영어로 생각 과정이나 reasoning을 작성하지 마세요.**
- 이미지만으로 확인할 수 있는 내용만 답변하세요
- Yes/No 질문에는 명확하게 "예" 또는 "아니요"로 답변하세요
- 개수 질문에는 정확한 숫자로 답변하세요
- 설명 질문에는 상세하고 구체적으로 답변하세요
- 모르는 내용은 추측하지 말고 "이미지에서 확인할 수 없습니다"라고 답변하세요
- 답변은 간결하고 명확하게 작성하세요. 불필요한 설명이나 생각 과정은 포함하지 마세요."""

        try:
            payload = {
                "text": prompt,
                "images": [
                    {
                        "type": "image",
                        "image": str(abs_image_path)
                    }
                ],
                "max_tokens": self.api_config["qwen3vl"]["max_tokens"],
                "temperature": 0.3,  # 답변은 낮은 temperature 사용
                "top_p": 0.8
            }
            
            response = requests.post(
                self.qwen3vl_url,
                json=payload,
                timeout=120
            )
            
            if response.status_code != 200:
                logger.error(f"Qwen3-VL API 호출 실패: {response.status_code}")
                return ""
            
            result = response.json()
            answer = result.get("text", "").strip()
            
            # 답변 정제: 영어 reasoning 제거
            cleaned_answer = self._clean_answer(answer)
            
            return cleaned_answer
            
        except Exception as e:
            logger.error(f"Qwen3-VL 답변 생성 중 오류 발생: {str(e)}")
            return ""
    
    def generate_with_gpt(self, question: str, image_path: str) -> str:
        """
        GPT-5.1을 사용하여 답변 생성
        
        Args:
            question: 질문
            image_path: 이미지 경로
        
        Returns:
            생성된 답변
        """
        logger.debug(f"GPT-5.1로 답변 생성: {question[:50]}...")
        
        prompt = f"""다음 질문에 대해 약포 이미지를 보고 정확하고 명확하게 답변해주세요.

질문: {question}
이미지 경로: {image_path}

답변 지침:
- **반드시 한국어로만 답변하세요. 영어로 생각 과정이나 reasoning을 작성하지 마세요.**
- 이미지만으로 확인할 수 있는 내용만 답변하세요
- Yes/No 질문에는 명확하게 "예" 또는 "아니요"로 답변하세요
- 개수 질문에는 정확한 숫자로 답변하세요
- 설명 질문에는 상세하고 구체적으로 답변하세요
- 모르는 내용은 추측하지 말고 "이미지에서 확인할 수 없습니다"라고 답변하세요
- 답변은 간결하고 명확하게 작성하세요. 불필요한 설명이나 생각 과정은 포함하지 마세요.

참고: 실제 이미지를 볼 수 없으므로, 일반적인 약포 이미지의 특성을 고려하여 답변해주세요."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.api_config["openai"]["model"],
                messages=[
                    {"role": "system", "content": "당신은 약포 이미지를 분석하여 질문에 답변하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=self.api_config["openai"]["max_tokens"],
                temperature=0.3  # 답변은 낮은 temperature 사용
            )
            
            answer = response.choices[0].message.content.strip()
            
            # 답변 정제: 영어 reasoning 제거
            cleaned_answer = self._clean_answer(answer)
            
            return cleaned_answer
            
        except Exception as e:
            logger.error(f"GPT-5.1 답변 생성 중 오류 발생: {str(e)}")
            return ""
    
    def generate_answers(self, questions: List[str], image_path: str, use_qwen: bool = True) -> List[Dict]:
        """
        질문 리스트에 대해 답변 생성
        
        Args:
            questions: 질문 리스트
            image_path: 이미지 경로
            use_qwen: Qwen3-VL 사용 여부 (True면 Qwen3-VL, False면 GPT)
        
        Returns:
            [{"question": str, "answer": str}, ...] 형식의 리스트
        """
        logger.info(f"{len(questions)}개의 질문에 대해 답변 생성 시작...")
        
        qa_pairs = []
        
        for question in questions:
            if use_qwen:
                answer = self.generate_with_qwen3vl(question, image_path)
            else:
                answer = self.generate_with_gpt(question, image_path)
            
            if answer:
                qa_pairs.append({
                    "question": question,
                    "answer": answer
                })
            else:
                logger.warning(f"답변 생성 실패: {question[:50]}...")
        
        logger.info(f"{len(qa_pairs)}개의 답변 생성 완료")
        return qa_pairs


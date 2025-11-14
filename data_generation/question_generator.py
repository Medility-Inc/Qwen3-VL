"""
질문 생성 모듈
GPT-5.1과 Qwen3-VL-8B-Thinking을 사용하여 VQA 질문을 생성합니다.
"""

import os
import logging
import requests
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

import yaml
from openai import OpenAI

# 환경변수 로드
load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """질문 생성 클래스"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.api_config = self.config["api"]
        self.question_config = self.config["question_generation"]
        
        # OpenAI 클라이언트 초기화
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Qwen3-VL API URL
        self.qwen3vl_url = f"{self.api_config['qwen3vl']['base_url']}/generate"
        
        logger.info("질문 생성기 초기화 완료")
    
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        config_file = Path(__file__).parent / config_path
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _build_medicine_context(self, medicine_info: List[Dict]) -> str:
        """의약품 정보를 컨텍스트 문자열로 변환"""
        if not medicine_info:
            return "의약품 정보가 없습니다."
        
        context_parts = []
        for med in medicine_info:
            parts = []
            if med.get("item_name"):
                parts.append(f"품명: {med['item_name']}")
            if med.get("drug_shape"):
                parts.append(f"형태: {med['drug_shape']}")
            if med.get("print_front"):
                parts.append(f"앞면 인쇄: {med['print_front']}")
            if med.get("print_back"):
                parts.append(f"뒷면 인쇄: {med['print_back']}")
            if med.get("onesglobal_ingredient_ko"):
                parts.append(f"성분: {med['onesglobal_ingredient_ko']}")
            
            if parts:
                context_parts.append(" | ".join(parts))
        
        return "\n".join(context_parts) if context_parts else "의약품 정보가 없습니다."
    
    def generate_with_gpt(self, image_path: str, medicine_info: List[Dict]) -> List[str]:
        """
        GPT-5.1을 사용하여 질문 생성
        
        Args:
            image_path: 이미지 경로
            medicine_info: 의약품 정보 리스트
        
        Returns:
            생성된 질문 리스트
        """
        logger.info(f"GPT-5.1로 질문 생성 시작: {image_path}")
        
        # 의약품 정보 컨텍스트
        medicine_context = self._build_medicine_context(medicine_info)
        
        # 최대 질문 수 가져오기
        max_questions = self.question_config.get("max_questions_per_model", 10)
        negative_ratio = int(self.question_config.get("negative_sample_ratio", 0.25) * 100)
        
        # 질문 생성 프롬프트
        system_prompt = f"""당신은 의약품 이미지를 분석하여 VQA(Visual Question Answering) 질문을 생성하는 전문가입니다.
다음 유형의 질문을 생성해주세요:
1. 존재 확인 (Yes/No): "약포에 파란색 약이 있나요?", "원형 알약이 포함되어 있나요?"
2. 개수 세기 (Counting): "약포에 몇 개의 알약이 있나요?", "파란색 약은 몇 개인가요?"
3. 색상 식별: "가장 큰 알약의 색깔은 무엇인가요?", "캡슐의 색상을 설명해주세요."
4. 형태 설명: "이 알약의 형태는 무엇인가요?"
5. 위치 확인: "파란색 약은 어디에 위치하나요?"

**중요: 부정 샘플(negative samples)도 포함해주세요.**
부정 샘플은 이미지에 없는 것에 대해 질문하여 "아니요" 답변을 유도하는 질문입니다.
예시:
- "파란색 원형 약이 있나요?" (이미지에 파란색 약이 없는 경우)
- "5개 이상의 알약이 있나요?" (실제로는 3개만 있는 경우)
- "빨간색 알약이 있나요?" (빨간색 알약이 없는 경우)

부정 샘플은 전체 질문의 약 {negative_ratio}%를 차지하도록 생성해주세요.

질문은 이미지만으로 답변할 수 있어야 하며, 모호하거나 애매한 질문은 피해주세요.
난이도는 쉬움(50%), 중간(30%), 어려움(20%)으로 분포시켜주세요.
각 질문은 한 줄로 작성하고, 정확히 {max_questions}개 이하의 질문만 생성해주세요."""

        user_prompt = f"""의약품 정보:
{medicine_context}

위 의약품 정보를 참고하여, 약포 이미지에 대한 다양한 VQA 질문을 생성해주세요.
이미지 경로: {image_path}

질문만 나열해주세요 (번호나 기호 없이, 각 질문은 한 줄씩)."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.api_config["openai"]["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=self.api_config["openai"]["max_tokens"],
                temperature=self.api_config["openai"]["temperature"]
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            # 질문 파싱 (줄바꿈으로 구분)
            raw_lines = [q.strip() for q in generated_text.split('\n') if q.strip()]
            
            # 질문 필터링: 실제 질문만 추출
            questions = []
            exclude_patterns = [
                r'^\d+\.',  # 번호로 시작 (예: "1. Existence")
                r'For example',  # 영어 예시
                r'Negative examples',  # 영어 부정 예시
                r'Check if',  # 영어 설명
                r'Wait,',  # 모델의 생각 과정
                r'So,',  # 모델의 생각 과정
                r'Got it,',  # 모델의 생각 과정
                r'as per info',  # 정보 참조
                r'품명 is',  # 의약품 정보
                r'형태:',  # 의약품 정보
                r'앞면 인쇄:',  # 의약품 정보
            ]
            
            import re
            for line in raw_lines:
                # 제외 패턴 체크
                should_exclude = False
                for pattern in exclude_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        should_exclude = True
                        break
                
                if should_exclude:
                    continue
                
                # 실제 질문인지 확인
                # 한국어가 포함되어 있고, '?'가 있고, 최소 길이 체크
                has_korean = bool(re.search(r'[가-힣]', line))
                has_question_mark = '?' in line
                is_long_enough = len(line) > 5
                
                # 영어만 있는 경우 제외 (예시나 설명일 가능성)
                is_english_only = bool(re.match(r'^[A-Za-z0-9\s\.,:;!?\-\(\)\[\]\"\']+$', line))
                
                if has_korean and has_question_mark and is_long_enough and not is_english_only:
                    # 앞뒤 따옴표 제거
                    line = line.strip('"\'')
                    questions.append(line)
            
            # 최대 개수 제한
            max_questions = self.question_config.get("max_questions_per_model", 10)
            questions = questions[:max_questions]
            
            logger.info(f"GPT-5.1로 {len(questions)}개의 질문 생성 완료")
            return questions
            
        except Exception as e:
            logger.error(f"GPT-5.1 질문 생성 중 오류 발생: {str(e)}")
            return []
    
    def generate_with_qwen3vl(self, image_path: str, medicine_info: List[Dict]) -> List[str]:
        """
        Qwen3-VL-8B-Thinking을 사용하여 질문 생성
        
        Args:
            image_path: 이미지 경로 (절대 경로로 변환 필요)
            medicine_info: 의약품 정보 리스트
        
        Returns:
            생성된 질문 리스트
        """
        logger.info(f"Qwen3-VL-8B-Thinking으로 질문 생성 시작: {image_path}")
        
        # 이미지 절대 경로 변환 (상대 경로를 절대 경로로)
        if Path(image_path).is_absolute():
            abs_image_path = Path(image_path)
        else:
            # data_generation 폴더 기준으로 상대 경로 해석
            abs_image_path = Path(__file__).parent / image_path
        
        if not abs_image_path.exists():
            logger.error(f"이미지 파일을 찾을 수 없습니다: {abs_image_path}")
            return []
        
        # 최대 질문 수 가져오기
        max_questions = self.question_config.get("max_questions_per_model", 10)
        negative_ratio = int(self.question_config.get("negative_sample_ratio", 0.25) * 100)
        
        # 의약품 정보 컨텍스트
        medicine_context = self._build_medicine_context(medicine_info)
        
        # 질문 생성 프롬프트
        prompt = f"""의약품 정보:
{medicine_context}

위 의약품 정보를 참고하여, 이 약포 이미지를 보고 다양한 VQA 질문을 생성해주세요.

다음 유형의 질문을 생성해주세요:
1. 존재 확인 (Yes/No): "약포에 파란색 약이 있나요?", "원형 알약이 포함되어 있나요?"
2. 개수 세기 (Counting): "약포에 몇 개의 알약이 있나요?", "파란색 약은 몇 개인가요?"
3. 색상 식별: "가장 큰 알약의 색깔은 무엇인가요?", "캡슐의 색상을 설명해주세요."
4. 형태 설명: "이 알약의 형태는 무엇인가요?"
5. 위치 확인: "파란색 약은 어디에 위치하나요?"

**중요: 부정 샘플(negative samples)도 포함해주세요.**
부정 샘플은 이미지에 없는 것에 대해 질문하여 "아니요" 답변을 유도하는 질문입니다.
예시:
- "파란색 원형 약이 있나요?" (이미지에 파란색 약이 없는 경우)
- "5개 이상의 알약이 있나요?" (실제로는 3개만 있는 경우)
- "빨간색 알약이 있나요?" (빨간색 알약이 없는 경우)

부정 샘플은 전체 질문의 약 {negative_ratio}%를 차지하도록 생성해주세요.

질문은 이미지만으로 답변할 수 있어야 하며, 모호하거나 애매한 질문은 피해주세요.
난이도는 쉬움(50%), 중간(30%), 어려움(20%)으로 분포시켜주세요.
각 질문은 한 줄로 작성하고, 정확히 {max_questions}개 이하의 질문만 생성해주세요.

질문만 나열해주세요 (번호나 기호 없이, 각 질문은 한 줄씩)."""

        try:
            # Qwen3-VL API 호출
            payload = {
                "text": prompt,
                "images": [
                    {
                        "type": "image",
                        "image": str(abs_image_path)
                    }
                ],
                "max_tokens": self.api_config["qwen3vl"]["max_tokens"],
                "temperature": self.api_config["qwen3vl"]["temperature"],
                "top_p": self.api_config["qwen3vl"]["top_p"]
            }
            
            response = requests.post(
                self.qwen3vl_url,
                json=payload,
                timeout=120
            )
            
            if response.status_code != 200:
                logger.error(f"Qwen3-VL API 호출 실패: {response.status_code}, {response.text}")
                return []
            
            result = response.json()
            generated_text = result.get("text", "").strip()
            
            # 질문 파싱 (줄바꿈으로 구분)
            raw_lines = [q.strip() for q in generated_text.split('\n') if q.strip()]
            
            # 질문 필터링: 실제 질문만 추출
            questions = []
            exclude_patterns = [
                r'^\d+\.',  # 번호로 시작 (예: "1. Existence")
                r'For example',  # 영어 예시
                r'Negative examples',  # 영어 부정 예시
                r'Check if',  # 영어 설명
                r'Wait,',  # 모델의 생각 과정
                r'So,',  # 모델의 생각 과정
                r'Got it,',  # 모델의 생각 과정
                r'as per info',  # 정보 참조
                r'품명 is',  # 의약품 정보
                r'형태:',  # 의약품 정보
                r'앞면 인쇄:',  # 의약품 정보
            ]
            
            import re
            for line in raw_lines:
                # 제외 패턴 체크
                should_exclude = False
                for pattern in exclude_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        should_exclude = True
                        break
                
                if should_exclude:
                    continue
                
                # 실제 질문인지 확인
                # 한국어가 포함되어 있고, '?'가 있고, 최소 길이 체크
                has_korean = bool(re.search(r'[가-힣]', line))
                has_question_mark = '?' in line
                is_long_enough = len(line) > 5
                
                # 영어만 있는 경우 제외 (예시나 설명일 가능성)
                is_english_only = bool(re.match(r'^[A-Za-z0-9\s\.,:;!?\-\(\)\[\]\"\']+$', line))
                
                if has_korean and has_question_mark and is_long_enough and not is_english_only:
                    # 앞뒤 따옴표 제거
                    line = line.strip('"\'')
                    questions.append(line)
            
            # 최대 개수 제한
            max_questions = self.question_config.get("max_questions_per_model", 10)
            questions = questions[:max_questions]
            
            logger.info(f"Qwen3-VL-8B-Thinking으로 {len(questions)}개의 질문 생성 완료")
            return questions
            
        except Exception as e:
            logger.error(f"Qwen3-VL 질문 생성 중 오류 발생: {str(e)}")
            return []
    
    def generate_questions(self, data_item: Dict, use_gpt: bool = True, use_qwen: bool = True) -> Dict[str, List[str]]:
        """
        두 모델을 사용하여 질문 생성
        
        Args:
            data_item: 데이터 아이템 (data_collector에서 생성된 형식)
            use_gpt: GPT 사용 여부
            use_qwen: Qwen3-VL 사용 여부
        
        Returns:
            {"gpt": [...], "qwen3vl": [...]} 형식의 딕셔너리
        """
        image_path = data_item["image_path"]
        medicine_info = data_item.get("medicine_info", [])
        
        results = {}
        
        if use_gpt:
            results["gpt"] = self.generate_with_gpt(image_path, medicine_info)
        
        if use_qwen:
            results["qwen3vl"] = self.generate_with_qwen3vl(image_path, medicine_info)
        
        return results


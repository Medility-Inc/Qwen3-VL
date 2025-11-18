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

    def _build_medicine_context(self, medicine_info: List[Dict]) -> str:
        """의약품 정보를 컨텍스트 문자열로 변환 (질문 생성용 간결 버전)"""
        if not medicine_info:
            return "의약품 정보가 없습니다."
        
        context_parts = []
        max_context_length = 5000  # 최대 컨텍스트 길이 제한 (문자 수)
        current_length = 0
        
        for med in medicine_info:
            parts = []
            # 기본 정보 (간결하게)
            if med.get("item_name"):
                parts.append(f"품명: {med['item_name']}")
            if med.get("manufacturer_name"):
                parts.append(f"제조사: {med['manufacturer_name']}")
            if med.get("class_name"):
                parts.append(f"분류: {med['class_name']}")
            if med.get("etc_otc_name"):
                parts.append(f"구분: {med['etc_otc_name']}")

            # 형태 정보 (시각적으로 확인 가능)
            shape_parts = []
            if med.get("drug_shape"):
                shape_parts.append(med["drug_shape"])
            if med.get("form_code_name"):
                shape_parts.append(med["form_code_name"])
            if shape_parts:
                parts.append(f"형태: {', '.join(shape_parts)}")
            
            # 크기 정보 (시각적으로 확인 가능)
            size_parts = []
            if med.get("thick"):
                size_parts.append(f"두께:{med['thick']}")
            if med.get("length_long"):
                size_parts.append(f"장축:{med['length_long']}")
            if med.get("length_short"):
                size_parts.append(f"단축:{med['length_short']}")
            if size_parts:
                parts.append("크기:" + "|".join(size_parts))
            
            # 색상 정보 (시각적으로 확인 가능)
            color_parts = []
            if med.get("color_front"):
                color_parts.append(med["color_front"])
            if med.get("color_back"):
                color_parts.append(med["color_back"])
            if color_parts:
                parts.append(f"색상:{','.join(color_parts)}")
            
            # 인쇄 정보 (시각적으로 확인 가능, 짧게)
            print_parts = []
            if med.get("print_front"):
                print_val = str(med['print_front'])[:50]  # 최대 50자로 제한
                print_parts.append(f"앞면:{print_val}")
            if med.get("print_back"):
                print_val = str(med['print_back'])[:50]  # 최대 50자로 제한
                print_parts.append(f"뒷면:{print_val}")
            if print_parts:
                parts.append("|".join(print_parts))
            
            # 시각적 설명 (있는 경우)
            if med.get("visual_description"):
                desc_val = str(med['visual_description'])[:100]  # 최대 100자로 제한
                parts.append(f"설명:{desc_val}")

            # 질문 생성에 필요한 핵심 정보만 포함 (효능/효과, 용법/용량, 주의사항 등 긴 텍스트 제외)
            med_context = " | ".join(parts)
            
            # 길이 체크
            if current_length + len(med_context) > max_context_length:
                # 남은 공간만큼만 추가
                remaining = max_context_length - current_length
                if remaining > 100:  # 최소 100자 이상 남았을 때만 추가
                    med_context = med_context[:remaining] + "..."
                    context_parts.append(med_context)
                break
            
            context_parts.append(med_context)
            current_length += len(med_context) + 1  # +1 for newline

        result = "\n".join(context_parts) if context_parts else "의약품 정보가 없습니다."
        
        # 최종 길이 제한 (안전장치)
        if len(result) > max_context_length:
            result = result[:max_context_length] + "\n...(의약품 정보가 길어 일부 생략됨)"
        
        return result
    
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
        
        # 영어 reasoning이 대부분인 경우: 한국어 부분만 추출
        lines = answer.split('\n')
        korean_lines = []
        english_line_count = 0
        total_line_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            total_line_count += 1
            has_korean = bool(re.search(r'[가-힣]', line))
            
            # 영어로만 구성된 라인 (reasoning일 가능성)
            is_english_only = bool(re.match(r'^[A-Za-z0-9\s\.,:;!?\-\(\)\[\]\"\'/]+$', line)) and not has_korean
            
            # 영어 reasoning 패턴 감지 (더 강력하게)
            english_reasoning_patterns = [
                r'^(Okay|So|Wait|Got it|Let me|I need|First|Then|But|However|Therefore|Thus|In the image|Looking at|Let\'s|I can see|The image shows|The user|Alternatively|Another|Check if|Wait,|So,)',
                r'^(tackle|asking about|says|translates to|appears to be|should be|would be)',
                r'which translates to',
                r'Wait, but',
                r'So the answer',
            ]
            
            is_reasoning = False
            for pattern in english_reasoning_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    is_reasoning = True
                    break
            
            if is_reasoning or (is_english_only and len(line) > 20):  # 긴 영어 라인은 reasoning으로 간주
                english_line_count += 1
                continue
            
            # 한국어가 포함된 라인만 추가
            if has_korean:
                korean_lines.append(line)
        
        # 영어 라인이 50% 이상이면 한국어 부분만 추출
        if total_line_count > 0 and english_line_count / total_line_count > 0.5:
            # 한국어가 포함된 라인만 사용
            cleaned_answer = '\n'.join(korean_lines).strip()
        else:
            # 일반적인 경우: 영어 reasoning 패턴 제거
            cleaned_lines = []
            skip_until_korean = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 영어 reasoning 패턴 감지
                is_reasoning = False
                for pattern in english_reasoning_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        is_reasoning = True
                        skip_until_korean = True
                        break
                
                # 영어만 있고 한국어가 없는 긴 라인은 제외
                has_korean = bool(re.search(r'[가-힣]', line))
                is_english_only = bool(re.match(r'^[A-Za-z0-9\s\.,:;!?\-\(\)\[\]\"\'/]+$', line)) and not has_korean
                
                if is_reasoning or (is_english_only and len(line) > 20):
                    skip_until_korean = True
                    continue
                
                # 한국어가 포함된 경우 reasoning 모드 종료
                if has_korean:
                    skip_until_korean = False
                    cleaned_lines.append(line)
                elif not skip_until_korean and line:
                    # 한국어가 없지만 reasoning 모드가 아닌 경우 (숫자, 기호 등)
                    cleaned_lines.append(line)
            
            cleaned_answer = '\n'.join(cleaned_lines).strip()
        
        # 최종 검증: 한국어가 전혀 없는 경우 원본에서 한국어 부분만 추출
        if not bool(re.search(r'[가-힣]', cleaned_answer)) and bool(re.search(r'[가-힣]', answer)):
            # 원본에서 한국어 부분만 추출
            korean_parts = []
            for line in answer.split('\n'):
                if re.search(r'[가-힣]', line):
                    # 영어 reasoning 패턴이 포함된 라인은 제외
                    is_reasoning = False
                    for pattern in english_reasoning_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            is_reasoning = True
                            break
                    if not is_reasoning:
                        korean_parts.append(line.strip())
            if korean_parts:
                cleaned_answer = '\n'.join(korean_parts).strip()
        
        return cleaned_answer if cleaned_answer else answer
    
    def generate_with_qwen3vl(self, question: str, image_path: str, medicine_info: List[Dict]) -> str:
        """
        Qwen3-VL-8B-Thinking을 사용하여 답변 생성
        
        Args:
            question: 질문
            image_path: 이미지 경로
        
        Returns:
            생성된 답변
        """
        medicine_context = self._build_medicine_context(medicine_info)
        logger.debug(f"Qwen3-VL로 답변 생성: {question[:50]}...")
        
        # 이미지 절대 경로 변환
        if Path(image_path).is_absolute():
            abs_image_path = Path(image_path)
        else:
            abs_image_path = Path(__file__).parent / image_path
        
        if not abs_image_path.exists():
            logger.error(f"이미지 파일을 찾을 수 없습니다: {abs_image_path}")
            return ""
        
        prompt = f"""다음 질문에 대해 약포 이미지를 중심으로 분석하여 2~4문장 분량의 서술형으로 답변해주세요.

의약품 정보 (참고용, 필요할 때만 사용):
{medicine_context}

이미지 경로: {image_path}

질문: {question}

답변 지침:
- 반드시 한국어로만 작성하고, 영어 단어나 영문 약어를 사용하지 마세요. 문장 흐름으로 2~4문장을 유지하세요.
- **중요: 답변은 반드시 이미지 관찰(색상, 형태, 위치, 개수 등)을 중심으로 작성하고, 의약품 메타데이터는 특정 의약품을 식별하거나 그 정보를 확인해야 할 때만 사용하세요.**
- Yes/No 질문은 "예" 또는 "아니요"로 시작하며, 그 이후에 이미지에서 관찰한 내용을 먼저 설명하고, 필요시 의약품 정보를 보조적으로 언급하세요.
- 개수 질문은 이미지에서 관찰한 정확한 수치와 관찰 과정을 먼저 기술하세요.
- 의약품 정보가 필요한 경우(예: 주성분, 효능 등)에만 메타데이터를 사용하세요.
- 확인할 수 없는 내용은 "이미지에서 확인할 수 없습니다"라고 명시하세요.
- 추측이나 근거 없는 확장은 피하고, 이미지 관찰을 우선시하세요."""

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
    
    def generate_with_gpt(self, question: str, image_path: str, medicine_info: List[Dict]) -> str:
        """
        GPT-5.1을 사용하여 답변 생성
        
        Args:
            question: 질문
            image_path: 이미지 경로
        
        Returns:
            생성된 답변
        """
        medicine_context = self._build_medicine_context(medicine_info)
        logger.debug(f"GPT-5.1로 답변 생성: {question[:50]}...")
        
        prompt = f"""다음 질문에 대해 약포 이미지를 중심으로 분석하여 2~4문장 분량의 서술형으로 답변해주세요.

의약품 정보 (참고용, 필요할 때만 사용):
{medicine_context}

이미지 경로: {image_path}

질문: {question}

답변 지침:
- 반드시 한국어로만 작성하고, 영어 단어나 영문 약어를 사용하지 마세요. 문장 흐름을 유지하며 2~4문장 분량의 서술형으로 답변하세요.
- **중요: 답변은 반드시 이미지 관찰(색상, 형태, 위치, 개수 등)을 중심으로 작성하고, 의약품 메타데이터는 특정 의약품을 식별하거나 그 정보를 확인해야 할 때만 사용하세요.**
- Yes/No 질문은 "예" 또는 "아니요"로 시작하고, 그 뒤에 이미지에서 관찰한 내용을 먼저 설명하고, 필요시 의약품 정보를 보조적으로 언급하세요.
- 개수/숫자 질문은 이미지에서 관찰한 정확한 수치와 관찰 과정을 먼저 기술하세요.
- 의약품 정보가 필요한 경우(예: 주성분, 효능 등)에만 메타데이터를 사용하세요.
- 확인할 수 없는 내용은 "이미지에서 확인할 수 없습니다"라고 명시하세요.
- 추측과 모순을 피하고, 이미지 관찰을 우선시하세요."""

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
    
    def generate_answers(self, questions: List[str], image_path: str, medicine_info: List[Dict], use_qwen: bool = True) -> List[Dict]:
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
                answer = self.generate_with_qwen3vl(question, image_path, medicine_info)
            else:
                answer = self.generate_with_gpt(question, image_path, medicine_info)
            
            if answer:
                qa_pairs.append({
                    "question": question,
                    "answer": answer
                })
            else:
                logger.warning(f"답변 생성 실패: {question[:50]}...")
        
        logger.info(f"{len(qa_pairs)}개의 답변 생성 완료")
        return qa_pairs


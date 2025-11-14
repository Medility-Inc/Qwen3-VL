"""
검수 모듈
GPT-5.1과 Qwen3-VL-8B-Thinking이 생성한 질문을 상호 검수하고, 중복을 제거합니다.
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


class Reviewer:
    """검수 클래스"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.api_config = self.config["api"]
        self.review_config = self.config["review"]
        
        # OpenAI 클라이언트 초기화
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Qwen3-VL API URL
        self.qwen3vl_url = f"{self.api_config['qwen3vl']['base_url']}/generate"
        
        logger.info("검수 모듈 초기화 완료")
    
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        config_file = Path(__file__).parent / config_path
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def review_with_qwen3vl(self, question: str, image_path: str) -> Dict[str, any]:
        """
        Qwen3-VL-8B-Thinking으로 질문 검수
        
        Args:
            question: 검수할 질문
            image_path: 이미지 경로
        
        Returns:
            {"approved": bool, "reason": str, "score": float} 형식
        """
        logger.debug(f"Qwen3-VL로 질문 검수: {question[:50]}...")
        
        # 이미지 절대 경로 변환
        if Path(image_path).is_absolute():
            abs_image_path = Path(image_path)
        else:
            abs_image_path = Path(__file__).parent / image_path
        
        if not abs_image_path.exists():
            logger.warning(f"이미지 파일을 찾을 수 없습니다: {abs_image_path}")
            return {"approved": False, "reason": "이미지 파일을 찾을 수 없습니다", "score": 0.0}
        
        prompt = f"""다음 질문이 약포 이미지에 대한 적절한 VQA 질문인지 검수해주세요.

질문: {question}

검수 기준:
1. 이미지만으로 답변할 수 있는 질문인가요?
2. 질문이 명확하고 모호하지 않은가요?
3. 의학적으로 부적절한 내용이 없는가요?
4. 질문이 이미지의 내용과 관련이 있는가요?

다음 형식으로 응답해주세요:
APPROVED: [YES/NO]
SCORE: [0.0-1.0 사이의 점수]
REASON: [검수 이유]"""

        try:
            payload = {
                "text": prompt,
                "images": [
                    {
                        "type": "image",
                        "image": str(abs_image_path)
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.3,
                "top_p": 0.8
            }
            
            response = requests.post(
                self.qwen3vl_url,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"Qwen3-VL API 호출 실패: {response.status_code}")
                return {"approved": False, "reason": "API 호출 실패", "score": 0.0}
            
            result = response.json()
            response_text = result.get("text", "").strip()
            
            # 응답 파싱
            approved = "APPROVED: YES" in response_text.upper() or "YES" in response_text.upper()[:50]
            score = 0.5  # 기본값
            
            # 점수 추출 시도
            import re
            score_match = re.search(r'SCORE:\s*([0-9.]+)', response_text, re.IGNORECASE)
            if score_match:
                try:
                    score = float(score_match.group(1))
                except:
                    pass
            
            reason = response_text.split("REASON:")[-1].strip() if "REASON:" in response_text else response_text
            
            return {
                "approved": approved and score >= self.review_config["min_quality_score"],
                "reason": reason,
                "score": score
            }
            
        except Exception as e:
            logger.error(f"Qwen3-VL 검수 중 오류 발생: {str(e)}")
            return {"approved": False, "reason": f"오류: {str(e)}", "score": 0.0}
    
    def review_with_gpt(self, question: str, image_path: str) -> Dict[str, any]:
        """
        GPT-5.1으로 질문 검수
        
        Args:
            question: 검수할 질문
            image_path: 이미지 경로
        
        Returns:
            {"approved": bool, "reason": str, "score": float} 형식
        """
        logger.debug(f"GPT-5.1로 질문 검수: {question[:50]}...")
        
        prompt = f"""다음 질문이 약포 이미지에 대한 적절한 VQA 질문인지 검수해주세요.

질문: {question}
이미지 경로: {image_path}

검수 기준:
1. 이미지만으로 답변할 수 있는 질문인가요?
2. 질문이 명확하고 모호하지 않은가요?
3. 의학적으로 부적절한 내용이 없는가요?
4. 질문이 이미지의 내용과 관련이 있는가요?

다음 형식으로 응답해주세요:
APPROVED: [YES/NO]
SCORE: [0.0-1.0 사이의 점수]
REASON: [검수 이유]"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.api_config["openai"]["model"],
                messages=[
                    {"role": "system", "content": "당신은 VQA 질문을 검수하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=512,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # 응답 파싱
            approved = "APPROVED: YES" in response_text.upper() or "YES" in response_text.upper()[:50]
            score = 0.5  # 기본값
            
            # 점수 추출 시도
            import re
            score_match = re.search(r'SCORE:\s*([0-9.]+)', response_text, re.IGNORECASE)
            if score_match:
                try:
                    score = float(score_match.group(1))
                except:
                    pass
            
            reason = response_text.split("REASON:")[-1].strip() if "REASON:" in response_text else response_text
            
            return {
                "approved": approved and score >= self.review_config["min_quality_score"],
                "reason": reason,
                "score": score
            }
            
        except Exception as e:
            logger.error(f"GPT-5.1 검수 중 오류 발생: {str(e)}")
            return {"approved": False, "reason": f"오류: {str(e)}", "score": 0.0}
    
    def review_questions(self, questions: List[str], image_path: str, source: str) -> List[Dict]:
        """
        질문 리스트 검수
        
        Args:
            questions: 검수할 질문 리스트
            image_path: 이미지 경로
            source: 질문 출처 ("gpt" 또는 "qwen3vl")
        
        Returns:
            검수 결과 리스트 [{"question": str, "approved": bool, "reason": str, "score": float}, ...]
        """
        logger.info(f"{source}에서 생성된 {len(questions)}개의 질문 검수 시작...")
        
        # Rate limiting 설정
        rate_limit_delay = self.review_config.get("rate_limit_delay", 0.5)
        
        reviewed_questions = []
        
        import time
        for idx, question in enumerate(questions, 1):
            if idx > 1:
                # Rate limiting: 첫 번째 요청 이후 대기
                time.sleep(rate_limit_delay)
            
            if source == "gpt":
                # GPT가 만든 질문은 Qwen3-VL이 검수
                review_result = self.review_with_qwen3vl(question, image_path)
            else:
                # Qwen3-VL이 만든 질문은 GPT가 검수
                review_result = self.review_with_gpt(question, image_path)
            
            reviewed_questions.append({
                "question": question,
                "approved": review_result["approved"],
                "reason": review_result["reason"],
                "score": review_result["score"]
            })
            
            if idx % 5 == 0:
                logger.info(f"검수 진행 중: {idx}/{len(questions)}")
        
        approved_count = sum(1 for q in reviewed_questions if q["approved"])
        logger.info(f"검수 완료: {approved_count}/{len(questions)}개 통과")
        
        return reviewed_questions
    
    def deduplicate_questions(self, questions: List[str], image_path: str) -> List[str]:
        """
        Qwen3-VL-8B-Thinking을 사용하여 중복 질문 제거
        
        Args:
            questions: 질문 리스트
            image_path: 이미지 경로
        
        Returns:
            중복 제거된 질문 리스트
        """
        if len(questions) <= 1:
            return questions
        
        logger.info(f"{len(questions)}개의 질문에서 중복 제거 시작...")
        
        # 이미지 절대 경로 변환
        if Path(image_path).is_absolute():
            abs_image_path = Path(image_path)
        else:
            abs_image_path = Path(__file__).parent / image_path
        
        if not abs_image_path.exists():
            logger.warning(f"이미지 파일을 찾을 수 없습니다: {abs_image_path}")
            return questions
        
        # 질문들을 하나의 프롬프트로 구성
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        
        prompt = f"""다음 질문들 중에서 의미적으로 중복되거나 유사한 질문들을 제거해주세요.

질문 목록:
{questions_text}

중복 제거 기준:
- 의미가 동일하거나 매우 유사한 질문은 하나만 남기기
- 유사도가 85% 이상인 질문들은 중복으로 간주
- 각 질문의 고유성을 유지하면서 최대한 다양한 질문 유지

중복 제거 후 남은 질문들의 번호만 나열해주세요 (예: 1, 3, 5, 7)."""

        try:
            payload = {
                "text": prompt,
                "images": [
                    {
                        "type": "image",
                        "image": str(abs_image_path)
                    }
                ],
                "max_tokens": 512,
                "temperature": 0.3,
                "top_p": 0.8
            }
            
            response = requests.post(
                self.qwen3vl_url,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"중복 제거 API 호출 실패: {response.status_code}")
                return questions  # 실패 시 원본 반환
            
            result = response.json()
            response_text = result.get("text", "").strip()
            
            # 선택된 질문 번호 추출 (더 엄격한 파싱)
            import re
            # 쉼표나 공백으로 구분된 숫자 패턴 찾기
            numbers = re.findall(r'\b(\d+)\b', response_text)
            selected_indices = []
            
            for n in numbers:
                try:
                    num = int(n)
                    # 유효한 범위 내의 번호만 선택 (1부터 시작하므로 -1)
                    if 1 <= num <= len(questions):
                        idx = num - 1
                        if idx not in selected_indices:  # 중복 제거
                            selected_indices.append(idx)
                except ValueError:
                    continue
            
            # 유효한 인덱스가 없거나 원본보다 많으면 원본 반환
            if not selected_indices:
                logger.warning("중복 제거 결과 파싱 실패, 모든 질문 유지")
                return questions
            
            if len(selected_indices) > len(questions):
                logger.warning(f"중복 제거 결과가 원본보다 많음 ({len(selected_indices)} > {len(questions)}), 원본 반환")
                return questions
            
            deduplicated = [questions[i] for i in sorted(selected_indices)]
            
            logger.info(f"중복 제거 완료: {len(questions)}개 -> {len(deduplicated)}개")
            return deduplicated
            
        except Exception as e:
            logger.error(f"중복 제거 중 오류 발생: {str(e)}")
            return questions  # 오류 시 원본 반환


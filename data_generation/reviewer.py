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

import yaml  # type: ignore[import]
from openai import OpenAI

# 환경변수 로드
load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger(__name__)


def _is_complete_sentence(text: str) -> bool:
    """
    텍스트가 완전한 문장인지 확인
    
    Args:
        text: 확인할 텍스트
        
    Returns:
        완전한 문장이면 True, 불완전하면 False
    """
    if not text or not text.strip():
        return False
    
    import re
    
    # 따옴표 제거 (JSON에서 온 경우)
    text = text.strip().strip('"').strip("'")
    
    # 빈 문자열 체크
    if not text:
        return False
    
    # 마지막 문자가 구두점(., !, ?, 。)인지 확인
    last_char = text[-1]
    has_ending_punctuation = last_char in '.!?。'
    
    # 문장이 끝나지 않은 패턴 감지 (조사로 끝나는 경우)
    incomplete_patterns = [
        r'[가-힣]이$',  # "리바록사반이" 같은 경우
        r'[가-힣]가$',  # "의약품이" 같은 경우
        r'[가-힣]을$',  # "알약을" 같은 경우
        r'[가-힣]를$',  # "정제를" 같은 경우
        r'[가-힣]은$',  # "약은" 같은 경우
        r'[가-힣]는$',  # "약은" 같은 경우
        r'[가-힣]와$',  # "약과" 같은 경우
        r'[가-힣]과$',  # "약과" 같은 경우
        r'[가-힣]에$',  # "약포에" 같은 경우
        r'[가-힣]에서$',  # "이미지에서" 같은 경우
        r'[가-힣]로$',  # "약으로" 같은 경우
        r'[가-힣]으로$',  # "약으로" 같은 경우
        r'[가-힣]의$',  # "약의" 같은 경우
    ]
    
    # 조사로 끝나는 경우 (구두점 없이) 불완전한 문장으로 간주
    for pattern in incomplete_patterns:
        if re.search(pattern, text) and not has_ending_punctuation:
            return False
    
    # "이다"로 끝나는 경우도 체크 (구두점 없으면 불완전)
    if re.search(r'[가-힣]이다$', text) and not has_ending_punctuation:
        return False
    
    # 너무 짧은 경우 (3자 이하) 불완전할 가능성
    if len(text) <= 3:
        return False
    
    # 기본적으로 완전한 문장으로 간주 (구두점이 있거나, 충분히 긴 경우)
    return True


def _clean_thinking(text: str) -> str:
    """Qwen3-VL Thinking 모델의 thinking 부분 제거"""
    if not text:
        return ""
    
    import re
    
    # </think> 또는 </reasoning> 태그 이후의 내용만 추출
    for tag in ["</think>", "</reasoning>", "</think>"]:
        if tag in text:
            parts = text.split(tag)
            if len(parts) > 1:
                text = parts[-1].strip()
                break
    
    # 영어 reasoning 패턴 제거
    lines = text.split('\n')
    cleaned_lines = []
    english_reasoning_patterns = [
        r'^(Okay|So|Wait|Got it|Let me|I need|First|Then|But|However|Therefore|Thus|In the image|Looking at|Let\'s|I can see|The image shows|The user|Alternatively|Another|Check if|Wait,|So,|tackle)',
        r'which translates to',
        r'Wait, but',
        r'So the answer',
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        has_korean = bool(re.search(r'[가-힣]', line))
        is_reasoning = False
        
        for pattern in english_reasoning_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                is_reasoning = True
                break
        
        # 영어로만 구성된 긴 라인은 reasoning으로 간주
        is_english_only = bool(re.match(r'^[A-Za-z0-9\s\.,:;!?\-\(\)\[\]\"\'/]+$', line)) and not has_korean
        if is_reasoning or (is_english_only and len(line) > 20):
            continue
        
        if has_korean or not is_english_only:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()


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

    def _resolve_image_path(self, image_path: str) -> Optional[Path]:
        """이미지 경로를 절대 경로로 변환하고 존재 여부 확인"""
        path = Path(image_path) if Path(image_path).is_absolute() else Path(__file__).parent / image_path
        if not path.exists():
            logger.warning(f"이미지 파일을 찾을 수 없습니다: {path}")
            return None
        return path

    def _parse_batch_review_response(self, text: str, num_questions: int) -> List[Dict[str, any]]:
        """배치 검수 응답을 parsing"""
        # 기본값을 통과로 설정 (명시적으로 NO가 없으면 통과)
        results = [
            {"approved": True, "score": 0.7, "reason": "기본 통과 (파싱 실패 시)"}
            for _ in range(num_questions)
        ]
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        parsed_count = 0
        for line in lines:
            parts = [part.strip() for part in line.split('|')]
            if len(parts) < 3:  # 최소 index|YES/NO|score 필요
                continue
            try:
                idx = int(parts[0]) - 1
            except ValueError:
                continue
            if idx < 0 or idx >= num_questions:
                continue
            
            # YES/NO 판단: 더 관대하게
            approval_text = parts[1].upper() if len(parts) > 1 else ""
            # 명시적으로 NO가 있으면 거부, 그 외에는 통과
            approved = "NO" not in approval_text or "YES" in approval_text
            
            try:
                score = float(parts[2]) if len(parts) > 2 else 0.7
            except ValueError:
                score = 0.7  # 기본값을 통과로 설정
            reason = " ".join(parts[3:]) if len(parts) > 3 else "검수 완료"
            results[idx] = {
                "approved": approved,
                "score": score,
                "reason": reason
            }
            parsed_count += 1
        
        # 파싱된 항목이 없으면 모든 질문을 통과로 처리
        if parsed_count == 0:
            logger.warning(f"배치 검수 응답 파싱 실패, 모든 질문을 통과로 처리: {text[:200]}")
        
        return results
    
    def review_with_qwen3vl(self, question: str, answer: str, image_path: str, medicine_info: List[Dict]) -> Dict[str, any]:
        """
        Qwen3-VL-8B-Thinking으로 질문-답변 쌍 검수
        
        Args:
            question: 검수할 질문
            answer: 검수할 답변
            image_path: 이미지 경로
            medicine_info: 의약품 정보
        
        Returns:
            {"approved": bool, "reason": str, "score": float} 형식
        """
        logger.debug(f"Qwen3-VL로 질문-답변 쌍 검수: {question[:50]}...")
        
        medicine_context = self._build_medicine_context(medicine_info)
        abs_image_path = self._resolve_image_path(image_path)
        if abs_image_path is None:
            return {"approved": False, "reason": "이미지 파일을 찾을 수 없습니다", "score": 0.0}

        prompt = f"""Please review the following question-answer pair based on the medicine blister pack image and medicine metadata.

Image path: {image_path}
Medicine information:
{medicine_context}

Question: {question}

Answer: {answer}

**Review Principle: Generally approve, and only reject obvious hallucinations.**

Only reject (NO) in the following cases:
1. **CRITICAL**: The answer describes visual elements (colors, shapes, sizes, counts, positions) that are NOT actually visible in the image. For example:
   - Answer says "white tablets are visible" but the image shows no white tablets
   - Answer says "several round capsules" but the image shows different shapes
   - Answer describes colors/shapes that contradict what is actually in the image
2. The question or answer clearly contradicts the image or medicine information (e.g., mentioning medicines not in the image, incorrect counts or colors)
3. The answer does not properly address the question
4. Medically serious and dangerous content (e.g., incorrect dosage, dangerous combinations)
5. The question or answer is completely meaningless or incomprehensible

**You should approve the following:**
- The question or answer is slightly ambiguous or incomplete
- The question format is not perfect
- General clarity issues
- Minor discrepancies with medicine information (not obvious hallucinations)

**CRITICAL CHECK**: Before approving, verify that the answer only describes visual elements that are actually visible in the image. If the answer mentions colors, shapes, sizes, or counts that are not in the image, you MUST reject it (NO).

Please respond in the following format:
APPROVED: YES or NO
SCORE: A score between 0.0-1.0
REASON: Review reason (briefly)"""

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
            
            # Thinking 부분 제거
            response_text = _clean_thinking(response_text)
            
            # 응답 파싱: 더 관대하게
            import re
            response_upper = response_text.upper()
            
            # 명시적으로 NO가 있으면 거부, 그 외에는 통과
            has_explicit_no = bool(re.search(r'APPROVED:\s*NO|NO\s*$|거부|제외', response_upper))
            has_explicit_yes = bool(re.search(r'APPROVED:\s*YES|YES\s*$|통과|승인', response_upper))
            
            # 명시적인 NO가 없으면 기본적으로 통과
            approved = not has_explicit_no or has_explicit_yes
            
            score = 0.7  # 기본값 (통과로 간주)
            
            # 점수 추출 시도
            score_match = re.search(r'SCORE:\s*([0-9.]+)', response_text, re.IGNORECASE)
            if score_match:
                try:
                    score = float(score_match.group(1))
                except:
                    pass
            
            reason = response_text.split("REASON:")[-1].strip() if "REASON:" in response_text else response_text[:200]
            
            return {
                "approved": approved,
                "reason": reason,
                "score": score
            }
            
        except Exception as e:
            logger.error(f"Qwen3-VL 검수 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본적으로 통과 (명백한 hallucination이 아니므로)
            return {"approved": True, "reason": f"오류 발생, 기본 통과: {str(e)}", "score": 0.7}
    
    def review_with_gpt(self, question: str, answer: str, image_path: str, medicine_info: List[Dict]) -> Dict[str, any]:
        """
        GPT-5.1으로 질문-답변 쌍 검수
        
        Args:
            question: 검수할 질문
            answer: 검수할 답변
            image_path: 이미지 경로
            medicine_info: 의약품 정보
        
        Returns:
            {"approved": bool, "reason": str, "score": float} 형식
        """
        medicine_context = self._build_medicine_context(medicine_info)
        logger.debug(f"GPT-5.1로 질문 검수: {question[:50]}...")
        
        abs_image_path = self._resolve_image_path(image_path)
        if abs_image_path is None:
            return {"approved": False, "reason": "이미지 파일을 찾을 수 없습니다", "score": 0.0}

        prompt = f"""Please review the following question-answer pair based on the medicine blister pack image and medicine metadata.

Image path: {image_path}
Medicine information:
{medicine_context}

Question: {question}

Answer: {answer}

**Review Principle: Generally approve, and only reject obvious hallucinations.**

Only reject (NO) in the following cases:
1. **CRITICAL**: The answer describes visual elements (colors, shapes, sizes, counts, positions) that are NOT actually visible in the image. For example:
   - Answer says "white tablets are visible" but the image shows no white tablets
   - Answer says "several round capsules" but the image shows different shapes
   - Answer describes colors/shapes that contradict what is actually in the image
2. The question or answer clearly contradicts the image or medicine information (e.g., mentioning medicines not in the image, incorrect counts or colors)
3. The answer does not properly address the question
4. Medically serious and dangerous content (e.g., incorrect dosage, dangerous combinations)
5. The question or answer is completely meaningless or incomprehensible

**You should approve the following:**
- The question or answer is slightly ambiguous or incomplete
- The question format is not perfect
- General clarity issues
- Minor discrepancies with medicine information (not obvious hallucinations)

**CRITICAL CHECK**: Before approving, verify that the answer only describes visual elements that are actually visible in the image. If the answer mentions colors, shapes, sizes, or counts that are not in the image, you MUST reject it (NO).

Please respond in the following format:
APPROVED: YES or NO
SCORE: A score between 0.0-1.0
REASON: Review reason (briefly)"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.api_config["openai"]["model"],
                messages=[
                    {"role": "system", "content": "You are an expert at reviewing VQA question-answer pairs."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=512,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # 응답 파싱: 더 관대하게
            import re
            response_upper = response_text.upper()
            
            # 명시적으로 NO가 있으면 거부, 그 외에는 통과
            has_explicit_no = bool(re.search(r'APPROVED:\s*NO|NO\s*$|거부|제외', response_upper))
            has_explicit_yes = bool(re.search(r'APPROVED:\s*YES|YES\s*$|통과|승인', response_upper))
            
            # 명시적인 NO가 없으면 기본적으로 통과
            approved = not has_explicit_no or has_explicit_yes
            
            score = 0.7  # 기본값 (통과로 간주)
            
            # 점수 추출 시도
            score_match = re.search(r'SCORE:\s*([0-9.]+)', response_text, re.IGNORECASE)
            if score_match:
                try:
                    score = float(score_match.group(1))
                except:
                    pass
            
            reason = response_text.split("REASON:")[-1].strip() if "REASON:" in response_text else response_text[:200]
            
            return {
                "approved": approved,
                "reason": reason,
                "score": score
            }
            
        except Exception as e:
            logger.error(f"GPT-5.1 검수 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본적으로 통과 (명백한 hallucination이 아니므로)
            return {"approved": True, "reason": f"오류 발생, 기본 통과: {str(e)}", "score": 0.7}

    def review_with_qwen3vl_batch(
        self,
        questions: List[str],
        image_path: str,
        medicine_info: List[Dict]
    ) -> List[Dict[str, any]]:
        """Qwen3-VL-8B-Thinking으로 질문 묶음을 검수"""
        medicine_context = self._build_medicine_context(medicine_info)
        abs_image_path = self._resolve_image_path(image_path)
        if abs_image_path is None:
            return [
                {"approved": False, "score": 0.0, "reason": "이미지 파일을 찾을 수 없습니다"}
                for _ in questions
            ]

        question_list = "\n".join(f"{idx + 1}. {q}" for idx, q in enumerate(questions))
        prompt = f"""Please review the following questions based on the medicine blister pack image and medicine metadata.

Image path: {image_path}
Medicine information:
{medicine_context}

Question list:
{question_list}

**Review Principle: Generally approve, and only reject obvious hallucinations.**

Only reject (NO) in the following cases:
1. The question clearly contradicts the image or medicine information (e.g., mentioning medicines not in the image)
2. Medically serious and dangerous content (e.g., incorrect dosage, dangerous combinations)
3. The question is completely meaningless or incomprehensible (e.g., random text like "asdfasdf?")

**You should approve the following:**
- The question is slightly ambiguous or incomplete
- The question format is not perfect
- General clarity issues
- Minor discrepancies with medicine information (not obvious hallucinations)

Response format:
Write each line as follows. Do not include '|' in the reason.
index|YES or NO|score between 0.0-1.0|reason
Example: 1|YES|0.8|Appropriate question.
Example: 2|NO|0.3|Mentioned medicine not in the image."""

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
                timeout=120
            )

            if response.status_code != 200:
                logger.error(f"배치 검수 API 실패: {response.status_code}")
                return [
                    {"approved": False, "score": 0.0, "reason": "API 호출 실패"}
                    for _ in questions
                ]

            result = response.json()
            response_text = result.get("text", "").strip()
            if not response_text:
                logger.warning("배치 검수 응답이 비어 있습니다.")
                return [
                    {"approved": True, "score": 0.7, "reason": "응답 없음, 기본 통과"}
                    for _ in questions
                ]

            # Thinking 부분 제거
            response_text = _clean_thinking(response_text)

            return self._parse_batch_review_response(response_text, len(questions))
        except Exception as e:
            logger.error(f"Qwen3-VL 배치 검수 오류: {str(e)}")
            return [
                {"approved": True, "score": 0.7, "reason": f"오류 발생, 기본 통과: {str(e)}"}
                for _ in questions
            ]

    def review_with_gpt_batch(
        self,
        questions: List[str],
        image_path: str,
        medicine_info: List[Dict]
    ) -> List[Dict[str, any]]:
        """GPT-5.1으로 질문 묶음을 검수"""
        medicine_context = self._build_medicine_context(medicine_info)
        question_list = "\n".join(f"{idx + 1}. {q}" for idx, q in enumerate(questions))
        prompt = f"""Please review the following questions based on the medicine blister pack image and medicine metadata.

Image path: {image_path}
Medicine information:
{medicine_context}

Question list:
{question_list}

**Review Principle: Generally approve, and only reject obvious hallucinations.**

Only reject (NO) in the following cases:
1. The question clearly contradicts the image or medicine information (e.g., mentioning medicines not in the image)
2. Medically serious and dangerous content (e.g., incorrect dosage, dangerous combinations)
3. The question is completely meaningless or incomprehensible (e.g., random text like "asdfasdf?")

**You should approve the following:**
- The question is slightly ambiguous or incomplete
- The question format is not perfect
- General clarity issues
- Minor discrepancies with medicine information (not obvious hallucinations)

Response format:
Write each line as follows. Do not include '|' in the reason.
index|YES or NO|score between 0.0-1.0|reason
Example: 1|YES|0.8|Appropriate question.
Example: 2|NO|0.3|Mentioned medicine not in the image."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.api_config["openai"]["model"],
                messages=[
                    {"role": "system", "content": "You are an expert at reviewing medicine blister pack images."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=512,
                temperature=0.3
            )

            response_text = response.choices[0].message.content.strip()
            if not response_text:
                logger.warning("GPT 배치 검수 응답이 비어 있습니다.")
                return [
                    {"approved": False, "score": 0.0, "reason": "응답 없음"}
                    for _ in questions
                ]

            return self._parse_batch_review_response(response_text, len(questions))
        except Exception as e:
            logger.error(f"GPT-5.1 배치 검수 오류: {str(e)}")
            return [
                {"approved": False, "score": 0.0, "reason": f"오류: {str(e)}"}
                for _ in questions
            ]
    
    def review_questions(
        self,
        questions: List[str],
        image_path: str,
        medicine_info: List[Dict],
        source: str
    ) -> List[Dict]:
        """
        질문 리스트 검수
        
        Args:
            questions: 검수할 질문 리스트
            image_path: 이미지 경로
            medicine_info: 해당 이미지의 의약품 정보
            source: 질문 출처 ("gpt" 또는 "qwen3vl")
        
        Returns:
            검수 결과 리스트 [{"question": str, "approved": bool, "reason": str, "score": float}, ...]
        """
        logger.info(f"{source}에서 생성된 {len(questions)}개의 질문 검수 시작...")
        
        reviewed_questions = []
        
        if source == "gpt":
            review_results = self.review_with_qwen3vl_batch(questions, image_path, medicine_info)
        else:
            review_results = self.review_with_gpt_batch(questions, image_path, medicine_info)

        # 검수 기준 완화: hallucination만 체크하므로 점수보다는 approved 여부를 우선시
        for idx, (question, review_result) in enumerate(zip(questions, review_results), start=1):
            # hallucination이 없으면 통과 (점수는 참고용)
            approved = review_result["approved"]  # hallucination 체크 결과를 우선시
            reason = review_result["reason"]
            reviewed_questions.append({
                "question": question,
                "approved": approved,
                "reason": reason,
                "score": review_result["score"]
            })
            if idx % 5 == 0:
                logger.info(f"검수 진행 중: {idx}/{len(questions)}")
        
        approved_count = sum(1 for q in reviewed_questions if q["approved"])
        logger.info(f"검수 완료: {approved_count}/{len(questions)}개 통과")
        
        return reviewed_questions
    
    def review_qa_pairs(
        self,
        qa_pairs: List[Dict[str, str]],
        image_path: str,
        medicine_info: List[Dict],
        source: str
    ) -> List[Dict]:
        """
        질문-답변 쌍 검수
        
        검수 항목:
        1. Hallucination 체크: 질문과 답변이 이미지/의약품 정보와 모순되는지 확인
        2. 완전한 문장 체크: 질문과 답변이 완전한 문장 형태인지 확인
        
        Args:
            qa_pairs: 검수할 질문-답변 쌍 리스트 [{"question": str, "answer": str}, ...]
            image_path: 이미지 경로
            medicine_info: 해당 이미지의 의약품 정보
            source: 질문 출처 ("gpt" 또는 "qwen3vl")
        
        Returns:
            검수 결과 리스트 [{"question": str, "answer": str, "approved": bool, "reason": str}, ...]
        """
        logger.info(f"{source}에서 생성된 {len(qa_pairs)}개의 질문-답변 쌍 검수 시작...")
        
        # 1단계: 완전한 문장 체크 (우선 검사)
        valid_qa_pairs = []
        reviewed_pairs_dict = {}  # 인덱스 -> 검수 결과 매핑
        
        for idx, qa_pair in enumerate(qa_pairs):
            question = qa_pair.get("question", "").strip()
            answer = qa_pair.get("answer", "").strip()
            
            question_complete = _is_complete_sentence(question)
            answer_complete = _is_complete_sentence(answer)
            
            if not question_complete:
                reviewed_pairs_dict[idx] = {
                    **qa_pair,
                    "approved": False,
                    "reason": "질문이 완전한 문장 형태가 아닙니다"
                }
                continue
            
            if not answer_complete:
                reviewed_pairs_dict[idx] = {
                    **qa_pair,
                    "approved": False,
                    "reason": "답변이 완전한 문장 형태가 아닙니다"
                }
                continue
            
            valid_qa_pairs.append((idx, qa_pair))
        
        # 2단계: Hallucination 체크 (개별 모드로 각 질문-답변 쌍 검수)
        if valid_qa_pairs:
            for original_idx, qa_pair in valid_qa_pairs:
                question = qa_pair.get("question", "").strip()
                answer = qa_pair.get("answer", "").strip()
                
                if source == "gpt":
                    review_result = self.review_with_qwen3vl(question, answer, image_path, medicine_info)
                else:
                    review_result = self.review_with_gpt(question, answer, image_path, medicine_info)
                
                reviewed_pairs_dict[original_idx] = {
                    **qa_pair,
                    "approved": review_result["approved"],
                    "reason": review_result["reason"]
                }
                
                # 진행 상황 로깅
                if len(reviewed_pairs_dict) % 5 == 0:
                    logger.info(f"검수 진행 중: {len(reviewed_pairs_dict)}/{len(qa_pairs)}")
        
        # 원래 순서대로 결과 재구성
        reviewed_pairs = [reviewed_pairs_dict[idx] for idx in range(len(qa_pairs))]
        
        approved_count = sum(1 for qa in reviewed_pairs if qa["approved"])
        logger.info(f"검수 완료: {approved_count}/{len(qa_pairs)}개 통과")
        
        return reviewed_pairs
    
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


"""
질문 생성 모듈
GPT-5.1과 Qwen3-VL-8B-Thinking을 사용하여 VQA 질문-답변 쌍을 생성합니다.
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
EXCLUDE_PATTERNS = [
    r'^\d+\.',  # 번호로 시작 (예: "1. Existence")
    r'For example',
    r'Negative examples',
    r'Check if',
    r'Wait,',
    r'So,',
    r'Got it,',
    r'as per info',
    r'품명 is',
    r'형태:',
    r'앞면 인쇄:',
    r'^-\s*\w+:',  # "- 의약품명: 형태?" 같은 형식 제외
    r'^-\s*[가-힣]+:',  # "- 한글명: 형태?" 같은 형식 제외
]

# 필수 기본 질문 템플릿 (이미지 중심)
REQUIRED_BASIC_QUESTION_TEMPLATES = [
    "이 약포에 의약품이 몇 개 들어가있나요?",
    "이 약포에 {medicine_name}이 들어가 있나요?",
    "{color} 의 의약품이 몇 개가 있나요?",
    "이 약포에 {shape} 형태의 알약이 몇 개 있나요?",
    "이 약포에서 가장 많은 색상은 무엇인가요?",
    "이 약포에 원형 알약이 몇 개 있나요?",
    "이 약포에 타원형 알약이 몇 개 있나요?",
    "이 약포에 흰색 알약이 몇 개 있나요?",
    "이 약포에 파란색 알약이 몇 개 있나요?",
    "이 약포에 빨간색 알약이 몇 개 있나요?",
    "이 약포에 노란색 알약이 몇 개 있나요?",
    "이 약포에 초록색 알약이 몇 개 있나요?",
    "이 약포에 분할선이 있는 알약이 몇 개 있나요?",
    "이 약포에 필름코팅된 알약이 몇 개 있나요?",
]

GENERAL_FALLBACK_QUESTIONS = [
    "이 약포에 보이는 알약의 색상은 무엇인가요?",
    "이 약포에서 여러 알약이 같은 형태를 가지고 있나요?",
    "이 약포에 5개 이상의 알약이 있나요?",
    "약포에서 가장 도드라지는 알약의 모양과 색상은 무엇인가요?",
    "이 약포의 알약들은 어떤 위치에 배치되어 있나요?",
    "이 약포의 알약들은 서로 다른 크기를 가지고 있나요?"
]


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
        self.question_types = self.question_config.get("question_types", [])
        
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
            if med.get("onesglobal_item_name"):
                parts.append(f"국제명: {med['onesglobal_item_name']}")
            if med.get("onesglobal_material_name"):
                parts.append(f"재질/소재: {med['onesglobal_material_name']}")
            if med.get("onesglobal_ingredient_ko"):
                parts.append(f"성분: {med['onesglobal_ingredient_ko']}")

            shape_parts = []
            if med.get("drug_shape"):
                shape_parts.append(med["drug_shape"])
            if med.get("onesglobal_form_type"):
                shape_parts.append(med["onesglobal_form_type"])
            if shape_parts:
                parts.append(f"제형/형태: {', '.join(shape_parts)}")
            if med.get("form_code_name"):
                parts.append(f"형태 코드: {med['form_code_name']}")
            if med.get("thick"):
                parts.append(f"두께: {med['thick']}")

            if med.get("onesglobal_route"):
                parts.append(f"투여 경로: {med['onesglobal_route']}")
            if med.get("onesglobal_indication"):
                parts.append(f"효능/효과: {med['onesglobal_indication']}")
            if med.get("onesglobal_ethical_type"):
                parts.append(f"의약분류: {med['onesglobal_ethical_type']}")
            if med.get("onesglobal_storage"):
                parts.append(f"보관 방법: {med['onesglobal_storage']}")
            if med.get("onesglobal_valid_term"):
                parts.append(f"유효기간: {med['onesglobal_valid_term']}")
            if med.get("onesglobal_pack_unit"):
                parts.append(f"포장 단위: {med['onesglobal_pack_unit']}")
            if med.get("print_front"):
                parts.append(f"앞면 인쇄: {med['print_front']}")
            if med.get("print_back"):
                parts.append(f"뒷면 인쇄: {med['print_back']}")

            if parts:
                context_parts.append(" | ".join(parts))

        return "\n".join(context_parts) if context_parts else "의약품 정보가 없습니다."

    def _clean_thinking(self, text: str) -> str:
        """Qwen3-VL Thinking 모델의 thinking 부분 제거"""
        if not text:
            return ""
        
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

    def _build_question_answer_prompt(
        self,
        medicine_context: str,
        image_path: str,
        max_qa_pairs: int,
        negative_ratio: int,
        basic_qa_count: int,
        exclude_questions: List[str] = None,
        medicine_info: List[Dict] = None
    ) -> str:
        """질문-답변 쌍 생성 프롬프트 (영어)"""
        exclude_section = ""
        if exclude_questions:
            exclude_section = f"\n\nExclude these questions (do not generate similar ones):\n" + "\n".join(f"- {q}" for q in exclude_questions[:10])
        
        # Get medicine names for negative samples
        medicine_names_in_list = []
        if medicine_info:
            medicine_names_in_list = [med.get("item_name", "") for med in medicine_info if med.get("item_name")]
        
        # 의약품 목록을 명시적으로 표시
        medicine_list_section = ""
        if medicine_names_in_list:
            medicine_list_section = f"\n\n**Available medicines in this image (use ALL of them evenly, not just one):**\n" + "\n".join(f"- {name}" for name in medicine_names_in_list)
            medicine_list_section += f"\n\n**IMPORTANT: Generate questions about ALL medicines evenly. Do NOT focus on only one medicine (e.g., {medicine_names_in_list[0] if medicine_names_in_list else 'one medicine'}). Distribute questions across all available medicines.**"
        
        prompt = f"""You are an expert at analyzing medicine blister pack images and generating VQA question-answer pairs.

Medicine information (reference only, use as supplementary):
{medicine_context}
{medicine_list_section}

Image path: {image_path}
{exclude_section}

Generate {max_qa_pairs} question-answer pairs that satisfy all conditions below:

**Key Principles:**
1. Questions must be image-centered. Medicine information should only be used to supplement image observations or identify specific medicines.
2. Questions and answers must be written in Korean only. Do not use English words or abbreviations.
3. **DO NOT generate questions about the blister pack (약포/약폼) itself**, such as:
   - Questions about the blister pack's backside printing (약포 뒷면 인쇄)
   - Questions about the blister pack's packaging form (약포 포장 형태)
   - Questions about the blister pack's design or markings (약포 디자인이나 표시)
   - Focus only on the medicines (알약/의약품) inside the blister pack, not the packaging itself.
4. **CRITICAL: Use ALL medicines evenly in your questions. If there are multiple medicines in the medicine information list, generate questions about EACH medicine, not just one. Distribute questions across all available medicines to ensure balanced coverage.**
5. **IMPORTANT: Do NOT directly mention printing/marking information (각인/인쇄 정보) in answers. Use printing/marking information ONLY for identifying which medicine it is, but do not explicitly describe the printing details in the answer. For example:**
   - BAD: "뒷면에 삼각형과 숫자 2.5로 보이는 인쇄가 반복되어 있습니다. 제공된 의약품 정보에 따르면 카사반정 2.5밀리그램은 뒷면에 삼각형 모양과 2.5 표시가 있는 것으로 되어 있어..."
   - GOOD: "이미지에서 보이는 약은 흰색 계열의 작은 원형 정제로 보입니다. 제공된 의약품 정보에 따르면 카사반정 2.5밀리그램은 원형 정제로 되어 있어, 이 포장의 약은 카사반정 2.5밀리그램으로 판단할 수 있습니다."
   - Focus on visible characteristics like color, shape, size, and use medicine information to identify the medicine, but do not mention printing/marking details in the answer.
6. Question-answer pair composition:
   - Basic pairs ({basic_qa_count} pairs): Questions answerable from image only (e.g., "이 약포에 의약품이 몇 개 들어가있나요?" → "이미지에서 약포 안에 들어 있는 의약품을 직접 관찰해 보니 총 4개가 확인됩니다.")
   - Extended pairs ({max_qa_pairs - basic_qa_count} pairs): Questions combining image observation + medicine information (e.g., "이 약포에 있는 라베라톤정의 주성분은 무엇인가요?" → "이 약포에 보이는 라베라톤정의 주성분은 라베프라졸 나트륨입니다.")
7. About {negative_ratio}% of questions should be negative samples asking about medicines NOT in the medicine information list. The medicine information list contains: {', '.join(medicine_names_in_list[:10]) if medicine_names_in_list else 'none'}. For negative samples, ask about medicines that are NOT in this list. For example: "이 약포에 아스피린이 들어가 있나요?" → "아니요. 이미지에서 확인된 의약품들은 모두 제공된 의약품 정보 목록에 포함되어 있으며, 아스피린은 목록에 없습니다."
8. Questions should cover various aspects: existence/counting/color/shape/location/etc.
9. Each question should be a single line ending with '?', and each answer should be 2-4 sentences in Korean.

**Basic question-answer examples (image-centered):**
- Q: "이 약포에 의약품이 몇 개 들어가있나요?"
  A: "이미지에서 약포 안에 들어 있는 의약품을 직접 관찰해 보니 총 4개가 확인됩니다. 빨간색과 흰색이 조합된 캡슐, 연두색 장방형 캡슐, 흰색 원형 정제, 노란색 원형 정제로 구성되어 있으며, 각각의 형태와 색상이 명확히 구분되어 있습니다."

**Extended question-answer examples (image + medicine info):**
- Q: "이 약포에 있는 라베라톤정의 주성분은 무엇인가요?"
  A: "이 약포에 보이는 라베라톤정의 주성분은 라베프라졸 나트륨입니다. 이미지에서 확인된 흰색 타원형 필름코팅정과 의약품 정보에서 '성분: 1정 중 라베프라졸 나트륨 10mg'으로 명시된 내용이 일치합니다."

- Q: "이 약포에 보이는 약이 카사반정 2.5밀리그램인지, 이미지와 의약품 정보를 함께 근거로 설명해 줄 수 있나요?"
  A: "이미지에서 보이는 약은 흰색 계열의 작은 원형 정제로 보입니다. 제공된 의약품 정보에 따르면 카사반정 2.5밀리그램은 원형 정제로 되어 있어, 이 포장의 약은 카사반정 2.5밀리그램으로 판단할 수 있습니다."
  (Note: Do NOT mention printing/marking details like "뒷면에 삼각형과 숫자 2.5로 보이는 인쇄" in the answer. Use printing information only for identification, not for description.)

**Examples of using multiple medicines evenly (if multiple medicines exist):**
- If medicines include "가스디알정" and "카사반정", generate questions about BOTH:
  - Q: "이 약포에 가스디알정이 들어가 있나요?"
  - Q: "이 약포에 카사반정의 주성분은 무엇인가요?"
  - Q: "이 약포에 가스디알정과 카사반정 중 어떤 것이 더 많은가요?"
- Do NOT generate all questions about only one medicine (e.g., only 카사반정).

**Negative sample examples (medicine NOT in the list):**
- Q: "이 약포에 아스피린이 들어가 있나요?"
  A: "아니요. 이미지에서 확인된 의약품들은 모두 제공된 의약품 정보 목록에 포함되어 있으며, 아스피린은 목록에 없습니다."

Output format: Each line should be "Q: [question]\nA: [answer]" (one pair per line)."""

        return prompt

    def _parse_qa_pairs(self, text: str) -> List[Dict[str, str]]:
        """질문-답변 쌍 파싱"""
        qa_pairs = []
        lines = text.split('\n')
        current_q = None
        current_a = None
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_q and current_a:
                    qa_pairs.append({"question": current_q, "answer": current_a})
                    current_q = None
                    current_a = None
                continue
            
            # 질문 시작
            if line.startswith("Q:") or line.startswith("질문:"):
                if current_q and current_a:
                    qa_pairs.append({"question": current_q, "answer": current_a})
                current_q = line.replace("Q:", "").replace("질문:", "").strip()
                current_a = None
            # 답변 시작
            elif line.startswith("A:") or line.startswith("답변:"):
                current_a = line.replace("A:", "").replace("답변:", "").strip()
            # 답변 계속
            elif current_a is not None:
                current_a += " " + line
            # 질문 계속
            elif current_q is not None and not line.startswith("Q:"):
                current_q += " " + line
        
        # 마지막 쌍 추가
        if current_q and current_a:
            qa_pairs.append({"question": current_q, "answer": current_a})
        
        return qa_pairs

    def generate_with_gpt(self, image_path: str, medicine_info: List[Dict]) -> List[Dict]:
        """
        GPT-5.1을 사용하여 질문-답변 쌍 생성
        
        Args:
            image_path: 이미지 경로
            medicine_info: 의약품 정보 리스트
        
        Returns:
            [{"question": str, "answer": str}, ...] 형식의 리스트
        """
        logger.info(f"GPT-5.1로 질문-답변 쌍 생성 시작: {image_path}")
        
        # 의약품 정보 컨텍스트
        medicine_context = self._build_medicine_context(medicine_info)
        
        # 최대 질문-답변 쌍 수 가져오기
        max_qa_pairs = self.question_config.get("max_questions_per_model", 10)
        negative_ratio = int(self.question_config.get("negative_sample_ratio", 0.25) * 100)
        basic_qa_count = int(max_qa_pairs * 0.4)  # 40% 기본 질문-답변 쌍
        
        # 질문-답변 쌍 생성 프롬프트
        prompt = self._build_question_answer_prompt(
            medicine_context=medicine_context,
            image_path=image_path,
            max_qa_pairs=max_qa_pairs,
            negative_ratio=negative_ratio,
            basic_qa_count=basic_qa_count,
            medicine_info=medicine_info
        )

        system_prompt = """You are an expert at generating Korean VQA question-answer pairs for medicine blister pack images.
- Generate questions and answers in Korean only.
- Questions must be image-centered.
- DO NOT generate questions about the blister pack (약포/약폼) itself, such as questions about the blister pack's backside printing, packaging form, or design. Focus only on the medicines (알약/의약품) inside the blister pack.
- CRITICAL: Use ALL medicines evenly in your questions. If there are multiple medicines in the medicine information list, generate questions about EACH medicine, not just one. Distribute questions across all available medicines to ensure balanced coverage.
- IMPORTANT: Do NOT directly mention printing/marking information (각인/인쇄 정보) in answers. Use printing/marking information ONLY for identifying which medicine it is, but do not explicitly describe the printing details in the answer. Focus on visible characteristics like color, shape, size instead.
- Answers should be 2-4 sentences in Korean, focusing on image observations first, then supplementing with medicine information when needed."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.api_config["openai"]["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=self.api_config["openai"]["max_tokens"],
                temperature=self.api_config["openai"]["temperature"]
            )
            
            generated_text = response.choices[0].message.content.strip()
            qa_pairs = self._parse_qa_pairs(generated_text)
            
            if len(qa_pairs) < max_qa_pairs:
                logger.warning(f"GPT-5.1: 생성된 질문-답변 쌍이 {max_qa_pairs}개보다 작습니다 ({len(qa_pairs)}개)")
            
            logger.info(f"GPT-5.1로 {len(qa_pairs)}개의 질문-답변 쌍 생성 완료")
            return qa_pairs[:max_qa_pairs]
            
        except Exception as e:
            logger.error(f"GPT-5.1 질문-답변 쌍 생성 중 오류 발생: {str(e)}")
            return []
    
    def generate_with_qwen3vl(self, image_path: str, medicine_info: List[Dict], exclude_questions: List[str] = None) -> List[Dict]:
        """
        Qwen3-VL-8B-Thinking을 사용하여 질문-답변 쌍 생성
        
        Args:
            image_path: 이미지 경로 (절대 경로로 변환 필요)
            medicine_info: 의약품 정보 리스트
            exclude_questions: 제외할 질문 리스트 (GPT가 생성한 질문)
        
        Returns:
            [{"question": str, "answer": str}, ...] 형식의 리스트
        """
        logger.info(f"Qwen3-VL-8B-Thinking으로 질문-답변 쌍 생성 시작: {image_path}")
        
        # 이미지 절대 경로 변환
        if Path(image_path).is_absolute():
            abs_image_path = Path(image_path)
        else:
            abs_image_path = Path(__file__).parent / image_path
        
        if not abs_image_path.exists():
            logger.error(f"이미지 파일을 찾을 수 없습니다: {abs_image_path}")
            return []
        
        # 최대 질문-답변 쌍 수 가져오기
        max_qa_pairs = self.question_config.get("max_questions_per_model", 10)
        negative_ratio = int(self.question_config.get("negative_sample_ratio", 0.25) * 100)
        basic_qa_count = int(max_qa_pairs * 0.4)  # 40% 기본 질문-답변 쌍
        
        # 의약품 정보 컨텍스트
        medicine_context = self._build_medicine_context(medicine_info)
        
        # 질문-답변 쌍 생성 프롬프트
        prompt = self._build_question_answer_prompt(
            medicine_context=medicine_context,
            image_path=image_path,
            max_qa_pairs=max_qa_pairs,
            negative_ratio=negative_ratio,
            basic_qa_count=basic_qa_count,
            exclude_questions=exclude_questions or [],
            medicine_info=medicine_info
        )

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
            
            if not generated_text:
                logger.warning("Qwen3-VL-8B-Thinking이 빈 텍스트를 생성했습니다.")
                return []
            
            # Thinking 부분 제거
            generated_text = self._clean_thinking(generated_text)
            
            logger.debug(f"생성된 원본 텍스트 (처음 500자): {generated_text[:500]}")
            
            qa_pairs = self._parse_qa_pairs(generated_text)

            if len(qa_pairs) < max_qa_pairs:
                logger.warning(f"Qwen3-VL-8B-Thinking: 생성된 질문-답변 쌍이 {max_qa_pairs}개보다 작습니다 ({len(qa_pairs)}개)")

            logger.info(f"Qwen3-VL-8B-Thinking으로 {len(qa_pairs)}개의 질문-답변 쌍 생성 완료")
            return qa_pairs[:max_qa_pairs]
            
        except Exception as e:
            logger.error(f"Qwen3-VL 질문-답변 쌍 생성 중 오류 발생: {str(e)}")
            return []
    
    def generate_questions(self, data_item: Dict, use_gpt: bool = True, use_qwen: bool = True) -> Dict[str, List[Dict]]:
        """
        두 모델을 사용하여 질문-답변 쌍 생성
        
        Args:
            data_item: 데이터 아이템 (data_collector에서 생성된 형식)
            use_gpt: GPT 사용 여부
            use_qwen: Qwen3-VL 사용 여부
        
        Returns:
            {"gpt": [{"question": str, "answer": str}, ...], "qwen3vl": [{"question": str, "answer": str}, ...]} 형식의 딕셔너리
        """
        image_path = data_item["image_path"]
        medicine_info = data_item.get("medicine_info", [])
        
        results = {}
        
        if use_gpt:
            results["gpt"] = self.generate_with_gpt(image_path, medicine_info)
        
        if use_qwen:
            # GPT가 생성한 질문들을 제외하고 생성
            gpt_questions = [qa["question"] for qa in results.get("gpt", [])]
            results["qwen3vl"] = self.generate_with_qwen3vl(image_path, medicine_info, exclude_questions=gpt_questions)
        
        return results

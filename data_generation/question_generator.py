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
        
        # Q: 또는 A:로 시작하는 라인이 있으면 그 부분부터 유지 (질문-답변 형식이 있는 경우)
        lines = text.split('\n')
        qa_start_idx = None
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("Q:") or stripped.startswith("A:") or stripped.startswith("질문:") or stripped.startswith("답변:"):
                qa_start_idx = idx
                break
        
        # 질문-답변 형식이 발견되면 그 부분부터만 사용
        if qa_start_idx is not None:
            text = '\n'.join(lines[qa_start_idx:])
            return text.strip()
        
        # 질문-답변 형식이 없으면 기존 로직 사용 (더 관대하게)
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
                # 빈 줄은 유지 (질문-답변 구분에 필요할 수 있음)
                cleaned_lines.append("")
                continue
            
            has_korean = bool(re.search(r'[가-힣]', line))
            is_reasoning = False
            
            for pattern in english_reasoning_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    is_reasoning = True
                    break
            
            # 영어로만 구성된 라인도 한글이 포함된 라인 다음에 오면 유지 (답변의 일부일 수 있음)
            is_english_only = bool(re.match(r'^[A-Za-z0-9\s\.,:;!?\-\(\)\[\]\"\'/]+$', line)) and not has_korean
            
            # reasoning 패턴이 명확한 경우만 제거
            if is_reasoning:
                continue
            
            # 영어만 있는 경우도 너무 길지 않으면 유지 (예: "The medicine info lists 가스디알정50밀리그램")
            if is_english_only and len(line) > 100:
                continue
            
            cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines).strip()
        
        # 결과가 비어있으면 원본 텍스트 반환 (최소한 뭔가는 파싱 시도)
        if not result:
            return text.strip()
        
        return result

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
            unique_medicines = list(set(medicine_names_in_list))  # 중복 제거
            medicine_list_section = f"\n\n**Available medicines in this image (MUST use ALL of them evenly):**\n" + "\n".join(f"- {name}" for name in unique_medicines)
            
            if len(unique_medicines) > 1:
                # 여러 의약품이 있는 경우
                questions_per_medicine = max(1, max_qa_pairs // len(unique_medicines))
                medicine_list_section += f"\n\n**CRITICAL DISTRIBUTION REQUIREMENT:**"
                medicine_list_section += f"\n- There are {len(unique_medicines)} different medicines: {', '.join(unique_medicines)}"
                medicine_list_section += f"\n- You MUST generate approximately {questions_per_medicine} questions about EACH medicine"
                medicine_list_section += f"\n- DO NOT generate all {max_qa_pairs} questions about only one medicine (e.g., only {unique_medicines[0]})"
                medicine_list_section += f"\n- Example distribution: {unique_medicines[0]} ({questions_per_medicine} questions), {unique_medicines[1] if len(unique_medicines) > 1 else 'other'} ({questions_per_medicine} questions), etc."
            else:
                # 하나의 의약품만 있는 경우
                medicine_list_section += f"\n\n**NOTE: Only one medicine ({unique_medicines[0]}) is in the list, but you should still generate diverse questions about it and include many negative sample questions about other medicines.**"
        
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
4. **CRITICAL: Use ALL medicines evenly in your questions. If there are multiple medicines in the medicine information list, you MUST generate questions about EACH medicine, not just one.**
   - If there are 2 medicines, generate roughly equal numbers of questions about each (e.g., 5 questions about medicine A, 5 questions about medicine B)
   - If there are 3 medicines, distribute questions evenly (e.g., 3-4 questions about each)
   - DO NOT generate 10 questions all about the same medicine. This is STRICTLY FORBIDDEN.
   - Each medicine should have questions covering different aspects: existence, counting, color, shape, ingredients, etc.
5. **IMPORTANT: Do NOT directly mention printing/marking information (각인/인쇄 정보) in answers. Use printing/marking information ONLY for identifying which medicine it is, but do not explicitly describe the printing details in the answer. For example:**
   - BAD: "뒷면에 삼각형과 숫자 2.5로 보이는 인쇄가 반복되어 있습니다. 제공된 의약품 정보에 따르면 카사반정 2.5밀리그램은 뒷면에 삼각형 모양과 2.5 표시가 있는 것으로 되어 있어..."
   - GOOD: "이미지에서 보이는 약은 흰색 계열의 작은 원형 정제로 보입니다. 제공된 의약품 정보에 따르면 카사반정 2.5밀리그램은 원형 정제로 되어 있어, 이 포장의 약은 카사반정 2.5밀리그램으로 판단할 수 있습니다."
   - Focus on visible characteristics like color, shape, size, and use medicine information to identify the medicine, but do not mention printing/marking details in the answer.
6. Question-answer pair composition:
   - Basic pairs ({basic_qa_count} pairs): Questions answerable from image only (e.g., "이 약포에 의약품이 몇 개 들어가있나요?" → "이미지에서 약포 안에 들어 있는 의약품을 직접 관찰해 보니 총 4개가 확인됩니다.")
   - Extended pairs ({max_qa_pairs - basic_qa_count} pairs): Questions combining image observation + medicine information (e.g., "이 약포에 있는 라베라톤정의 주성분은 무엇인가요?" → "이 약포에 보이는 라베라톤정의 주성분은 라베프라졸 나트륨입니다.")
7. **IMPORTANT: Generate many negative sample questions (약 {negative_ratio}% of total). For each medicine in the list, generate 1-2 negative sample questions asking about common medicines NOT in the list. Use these question formats:**
   - "이 이미지에 [약명] 정제가 포함되어 있나요?"
   - "이 약들 중 [약명] 정제가 있는지, 이미지와 제공된 약 정보 기준으로 판단해 줄 수 있나요?"
   - "이 약포에 [약명]이 들어가 있는지, 이미지와 제공된 의약품 정보를 바탕으로 설명해 줄 수 있나요?"
   - The medicine information list contains: {', '.join(medicine_names_in_list[:10]) if medicine_names_in_list else 'none'}
   - For negative samples, ask about common medicines NOT in this list (e.g., 타이레놀, 이부프로펜, 아스피린, 파라세타몰, 아목시실린, 세파클러, 로키소닌, 게보린, 부루펜, 케토톱, 아세트아미노펜, 나프록센, 디클로페낙, 멜록시캠, 셀레콕시브, etc.)
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

**Examples of using multiple medicines evenly (if multiple medicines exist) - THIS IS MANDATORY:**
- If medicines include "가스디알정" and "카사반정", you MUST generate questions about BOTH:
  - 가스디알정 관련 질문들 (약 절반):
    - Q: "이 약포에 가스디알정이 들어가 있나요?"
    - Q: "이 약포에 가스디알정의 주성분은 무엇인가요?"
    - Q: "이 약포에 가스디알정이 몇 개 있나요?"
    - Q: "이 약포에 가스디알정의 색상은 무엇인가요?"
  - 카사반정 관련 질문들 (약 절반):
    - Q: "이 약포에 카사반정이 들어가 있나요?"
    - Q: "이 약포에 카사반정의 주성분은 무엇인가요?"
    - Q: "이 약포에 카사반정이 몇 개 있나요?"
    - Q: "이 약포에 카사반정의 형태는 무엇인가요?"
  - 비교 질문:
    - Q: "이 약포에 가스디알정과 카사반정 중 어떤 것이 더 많은가요?"
- **STRICTLY FORBIDDEN: Generating all 10 questions about only one medicine (e.g., only 가스디알정 or only 카사반정). This will result in rejection.**

**Negative sample examples (medicine NOT in the list) - GENERATE MANY OF THESE:**
- Q: "이 이미지에 타이레놀 정제가 포함되어 있나요?"
  A: "사진에서 보이는 약들은 모두 흰색 계열의 원형, 타원형 정제와 캡슐로 구성되어 있습니다. 제공된 목록에는 타이레놀 성분의 약이 없고, 모양과 정보로 보아 이 이미지에는 타이레놀 정제는 포함되어 있지 않습니다."

- Q: "이 약들 중 이부프로펜 정제가 있는지, 이미지와 제공된 약 정보 기준으로 판단해 줄 수 있나요?"
  A: "사진 속 약들은 카바스타정, 위제로츄어블정, 한림알프라졸람정, 유니테론정, 코시바정, 류멜캡슐, 피나스틴정, 자누다움엠정으로 구성된 것으로 보입니다. 의약품 정보 목록에도 이부프로펜 제제는 포함되어 있지 않으므로, 이 이미지에는 이부프로펜 정제가 없는 것으로 판단됩니다."

- Q: "이 약포에 아스피린 정제가 들어 있는지, 이미지와 제공된 의약품 정보를 바탕으로 설명해 줄 수 있나요?"
  A: "이미지에서 보이는 약포는 카사반정 2.5밀리그램의 포장과 인쇄 양식이 일치하며, 제공된 정보에도 카사반정과 가스디알정만 언급되어 있습니다. 아스피린 정제에 대한 언급은 없고, 포장 표기에서도 아스피린을 나타내는 표시가 보이지 않습니다. 따라서 이 약포에는 아스피린 정제가 포함되어 있지 않은 것으로 판단됩니다."

**IMPORTANT: Generate multiple negative sample questions for each image. Use different common medicine names (타이레놀, 이부프로펜, 아스피린, 파라세타몰, 아목시실린, 게보린, 부루펜, 케토톱, 아세트아미노펜, 나프록센, 디클로페낙, 멜록시캠, 셀레콕시브, 로키소닌, 세파클러, etc.) that are NOT in the medicine information list.**

**OUTPUT FORMAT - START IMMEDIATELY WITHOUT ANY THINKING OR EXPLANATION:**

Q: [질문 내용]
A: [답변 내용]

Q: [질문 내용]
A: [답변 내용]

(Repeat for all {max_qa_pairs} pairs)

**CRITICAL INSTRUCTIONS:**
1. DO NOT write any thinking, reasoning, or explanation before the first Q:
2. DO NOT repeat the prompt or examples
3. START your response directly with "Q:" followed by the first question
4. Generate {max_qa_pairs} question-answer pairs immediately
5. Use Korean only for questions and answers

Example of correct output format:
Q: 이 약포에 의약품이 몇 개 들어가있나요?
A: 이미지에서 약포 안에 들어 있는 의약품을 직접 관찰해 보니 총 4개가 확인됩니다.

Q: 이 약포에 가스디알정이 들어가 있나요?
A: 네, 이미지에서 확인된 흰색 원형 정제가 가스디알정으로 보입니다.

Now generate your {max_qa_pairs} question-answer pairs:"""

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
            
            # 질문 시작 (더 유연한 패턴 매칭)
            # Q:, Q., Q , 질문:, 질문., 질문 등 다양한 형식 지원
            question_pattern = re.match(r'^(Q|질문)[:.\s]+(.+)', line, re.IGNORECASE)
            if question_pattern:
                if current_q and current_a:
                    qa_pairs.append({"question": current_q, "answer": current_a})
                current_q = question_pattern.group(2).strip()
                current_a = None
                continue
            
            # 답변 시작 (더 유연한 패턴 매칭)
            # A:, A., A , 답변:, 답변., 답변 등 다양한 형식 지원
            answer_pattern = re.match(r'^(A|답변)[:.\s]+(.+)', line, re.IGNORECASE)
            if answer_pattern:
                current_a = answer_pattern.group(2).strip()
                continue
            
            # 한글로 시작하는 질문 패턴 (예: "이 약포에...", "이미지에...")
            if current_q is None and current_a is None:
                # 한글로 시작하고 물음표로 끝나는 경우 질문으로 간주
                if re.search(r'[가-힣]', line) and line.endswith('?'):
                    current_q = line
                    continue
            
            # 답변 계속
            if current_a is not None:
                current_a += " " + line
            # 질문 계속 (Q:로 시작하지 않는 경우)
            elif current_q is not None:
                # 다음 질문이 시작되는 경우 (Q: 또는 한글 질문 패턴)
                if (re.match(r'^(Q|질문)[:.\s]+', line, re.IGNORECASE) or 
                    (re.search(r'[가-힣]', line) and line.endswith('?') and len(line) > 10)):
                    # 현재 쌍 저장
                    if current_q and current_a:
                        qa_pairs.append({"question": current_q, "answer": current_a})
                    # 새 질문 시작
                    if re.match(r'^(Q|질문)[:.\s]+', line, re.IGNORECASE):
                        current_q = re.sub(r'^(Q|질문)[:.\s]+', '', line, flags=re.IGNORECASE).strip()
                    else:
                        current_q = line
                    current_a = None
                else:
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
- CRITICAL: Use ALL medicines evenly in your questions. If there are multiple medicines in the medicine information list, you MUST generate questions about EACH medicine, not just one. DO NOT generate all questions about only one medicine - this is STRICTLY FORBIDDEN. Distribute questions evenly across all available medicines.
- IMPORTANT: Generate MANY negative sample questions (asking about medicines NOT in the list). Use question formats like "이 이미지에 [약명] 정제가 포함되어 있나요?" or "이 약들 중 [약명] 정제가 있는지, 이미지와 제공된 약 정보 기준으로 판단해 줄 수 있나요?". Use common medicine names like 타이레놀, 이부프로펜, 아스피린, 파라세타몰, 아목시실린, 게보린, 부루펜, 케토톱, 아세트아미노펜, 나프록센, 디클로페낙, 멜록시캠, 셀레콕시브, 로키소닌, 세파클러, etc.
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
            # Thinking 모델이 긴 reasoning을 생성할 수 있으므로 max_tokens를 늘림
            max_tokens_for_qa = max(self.api_config["qwen3vl"]["max_tokens"], 4096)
            payload = {
                "text": prompt,
                "images": [
                    {
                        "type": "image",
                        "image": str(abs_image_path)
                    }
                ],
                "max_tokens": max_tokens_for_qa,
                "temperature": self.api_config["qwen3vl"]["temperature"],
                "top_p": self.api_config["qwen3vl"]["top_p"]
            }
            logger.debug(f"Qwen3-VL API 호출: max_tokens={max_tokens_for_qa}")
            
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
                logger.debug(f"API 응답 전체: {result}")
                return []
            
            # 원본 텍스트 로깅
            breakpoint()
            logger.info(f"생성된 원본 텍스트 (처음 500자): {generated_text[:500]}")
            logger.debug(f"생성된 원본 텍스트 전체 길이: {len(generated_text)}자")
            
            # Thinking 부분이 있는 경우, 실제 질문-답변 부분만 추출 시도
            # 프롬프트 예시 키워드 제거 (예시는 제외)
            lines = generated_text.split('\n')
            
            # 프롬프트 예시 패턴 제거 (예: "**Examples of...", "**Negative sample examples..." 등)
            example_keywords = [
                "**Examples of", "**Negative sample examples", "**Basic question-answer examples",
                "**Extended question-answer examples", "Example format:", "Output format:",
                "**CRITICAL:", "**IMPORTANT:", "**NOTE:", "STRICTLY FORBIDDEN"
            ]
            
            # 실제 질문-답변 시작 지점 찾기
            qa_start_idx = None
            for idx, line in enumerate(lines):
                stripped = line.strip()
                
                # 프롬프트 예시 부분은 건너뛰기
                if any(keyword in stripped for keyword in example_keywords):
                    continue
                
                # Q: 또는 A:로 시작하는지 확인
                if (stripped.startswith("Q:") or stripped.startswith("A:") or 
                    stripped.startswith("질문:") or stripped.startswith("답변:")):
                    # 프롬프트 예시가 아닌지 확인
                    content_after_prefix = stripped[2:].strip() if stripped.startswith("Q:") or stripped.startswith("A:") else stripped[3:].strip()
                    
                    # 한글이 포함되어 있고, 따옴표로 시작하지 않으며, "이 약포에" 같은 실제 질문 패턴이면 실제 질문으로 간주
                    if (re.search(r'[가-힣]', content_after_prefix) and 
                        not content_after_prefix.startswith('"') and
                        not content_after_prefix.startswith("이 약포에 가스디알정이 들어가 있나요?") and  # 예시 제외
                        len(content_after_prefix) > 5):  # 너무 짧으면 예시일 가능성
                        qa_start_idx = idx
                        logger.info(f"질문-답변 시작 라인 발견: {idx}번째 라인")
                        logger.info(f"라인 내용: {stripped[:150]}")
                        # 주변 라인도 확인
                        if idx > 0:
                            logger.debug(f"이전 라인: {lines[idx-1].strip()[:100]}")
                        if idx < len(lines) - 1:
                            logger.debug(f"다음 라인: {lines[idx+1].strip()[:100]}")
                        break
            
            # 질문-답변 부분이 발견되면 그 부분부터 파싱
            if qa_start_idx is not None:
                qa_text = '\n'.join(lines[qa_start_idx:])
                logger.info(f"질문-답변 부분 추출 (처음 1500자): {qa_text[:1500]}")
                qa_pairs = self._parse_qa_pairs(qa_text)
            else:
                # Q: 또는 A:를 찾지 못한 경우, 한글로 시작하고 ?로 끝나는 라인을 찾아서 파싱 시도
                logger.warning("Q: 또는 A:로 시작하는 실제 질문-답변 라인을 찾지 못했습니다.")
                logger.warning("한글 질문 패턴으로 파싱 시도...")
                qa_pairs = self._parse_qa_pairs(generated_text)
            
            if len(qa_pairs) == 0:
                logger.warning(f"파싱된 질문-답변 쌍이 없습니다.")
                logger.warning(f"원본 텍스트 (처음 3000자): {generated_text[:3000]}")
                logger.warning(f"원본 텍스트 (마지막 1000자): {generated_text[-1000:] if len(generated_text) > 1000 else generated_text}")

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

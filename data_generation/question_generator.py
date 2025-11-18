"""
질문 생성 모듈
GPT-5.1과 Qwen3-VL-8B-Thinking을 사용하여 VQA 질문-답변 쌍을 생성합니다.
"""

import os
import logging
import requests
import re
import base64
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import io

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
        medicine_info: List[Dict] = None,
        llm_type: str = "gpt5.1",
    ) -> str:
        """GPT‑5.1 스타일의 고품질 질문-답변 생성 프롬프트 (GPT / Qwen 공용)

        - 프롬프트는 영어, 생성되는 질문/답변은 한국어.
        - 최대 10개 QA, 기본 목표는 10개 생성 시 '긍정형 7개 / 부정형 3개'.
        - Qwen3-VL은 GPT-5.1이 이미 생성한 질문(exclude_questions)을 피해서 다른 질문을 생성.
        """

        # 현재 구현에서는 negative_ratio, basic_qa_count는 직접 사용하지 않지만
        # 프롬프트 정책 확장을 위해 시그니처에 유지한다.
        _ = (negative_ratio, basic_qa_count)

        # 제외할 질문들 섹션 (Qwen이 GPT 질문을 피하도록 사용)
        exclude_section = ""
        if exclude_questions:
            exclude_section = (
                "\n\nPreviously generated questions (MUST NOT be repeated or paraphrased):\n"
                + "\n".join(f"- {q}" for q in exclude_questions[:20])
            )

        # 의약품 이름 목록
        medicine_names_in_list: List[str] = []
        if medicine_info:
            medicine_names_in_list = [
                med.get("item_name", "")
                for med in medicine_info
                if med.get("item_name")
            ]

        # 의약품 목록 섹션
        medicine_list_section = ""
        unique_medicines: List[str] = []
        if medicine_names_in_list:
            unique_medicines = list(set(medicine_names_in_list))
            medicine_list_section = (
                "\n\n**Available medicines in this image (reference list):**\n"
                + "\n".join(f"- {name}" for name in unique_medicines)
            )

        # 공통: 최대 10개까지, 10개 이상 요청되면 10개로 클램프
        max_pairs = min(max_qa_pairs, 10)

        # 공통: 긍/부정 비율 계산 (7:3 근사)
        if max_pairs >= 10:
            target_total = 10
            target_positive = 7
            target_negative = 3
        else:
            target_total = max_pairs
            target_negative = max(1, round(target_total * 0.3))
            target_positive = max(1, target_total - target_negative)

        # Explicit print for debug: Show the 3 core sections (remove in production)
        logger.info("=== medicine_context ===")
        print(medicine_context)
        logger.info("=== medicine_list_section ===")
        print(medicine_list_section)
        logger.info("=== exclude_section ===")
        print(exclude_section)

        # 공통 설명 블록 (GPT / Qwen 둘 다 공유)
        base_body = f"""You are an expert at analyzing medicine blister pack images and generating **high-quality** Korean VQA question–answer pairs for **fine-tuning**.

<task_context>
- Input:
  - A blister pack image (tablets/capsules and their packaging)
  - A structured medicine information list for candidate medicines in the image (NOTE: This is ONLY for dataset generation purposes. In real usage, users will ONLY provide the image, NOT the medicine information list.)
- Goal:
  - Generate Korean question–answer pairs that are suitable for supervised fine-tuning of a multimodal LLM.
  - **CRITICAL**: Focus on questions that are **ambiguous or difficult for users to confirm** by themselves. Avoid questions that users can easily verify by simply looking at the image (e.g., "How many tablets that look like X are there?" - users can count them themselves).
  - Prioritize questions that require **uncertainty reasoning** or **identification based on limited visual cues** (e.g., when imprints are not clearly visible).
- All questions and answers must be written **only in Korean**.
- Questions do NOT need to end with a question mark (?). Use natural Korean phrasing.
</task_context>

<input_data>
Medicine information (reference, may include imprint, color, shape, thickness, ingredients, etc.):
{medicine_context}
{medicine_list_section}

Image path (for your reference as a mental pointer; do NOT hallucinate unseen content):
{image_path}
{exclude_section}
</input_data>

<qa_requirements_core>
1. **Number of pairs and polarity (positive vs negative)**
   - Generate **up to {target_total} question–answer pairs** (never exceed 10).
   - Aim for approximately **{target_positive} positive questions** and **{target_negative} negative questions**:
     - Positive question: about medicines that are actually present or plausibly present in the image.
     - Negative question: about medicines that are clearly **not** present in the image (e.g., common OTC drugs not in the list and not visually observed).
   - **CRITICAL**: Avoid generating duplicate or very similar questions. Each question must be unique and cover different aspects.

2. **Question focus (ambiguous/uncertain questions first)**
   - **PRIORITY**: Generate questions that are **ambiguous or difficult for users to confirm** by themselves. These are questions where users need AI assistance because:
     - The imprint/marking is not clearly visible, so identification requires reasoning from color/shape/size alone.
     - Multiple similar-looking medicines exist, making it hard to distinguish which is which.
     - The question asks for confirmation of user's hypothesis (e.g., "I think this looks like medicine X, is it correct?").
   - **AVOID**: Questions that users can easily verify by simply looking at the image:
     - "How many tablets that look like X are there?" (users can count them)
     - "What color are the tablets?" (users can see it directly)
     - "Where are the tablets located?" (users can see it directly)
   - **STRICTLY FORBIDDEN**: Meta-questions about how to answer or explain things:
     - "어떻게 설명해 줄 수 있을까요?" (How can I explain this?)
     - "어떻게 답해야 할까요?" (How should I answer?)
     - "어떻게 대답하는 것이 적절할까요?" (How should I respond appropriately?)
     - "~라고 하면 어떻게 설명해 줄 수 있을까요?" (If someone asks ~, how can I explain?)
     - "~라는 요청이 들어오면 어떻게 답해야 할까요?" (If a request comes in asking ~, how should I answer?)
     - These are questions about the AI's response strategy, NOT questions that users would ask about the medicine image.
   - **GOOD question types** (prioritize these):
     - "Is this medicine X?" when imprint is not clearly visible (requires reasoning from shape/color/size).
     - "Can you confirm if this is medicine X?" (user's hypothesis confirmation).
     - "Is medicine X present in this image?" when visual identification is ambiguous.
     - "Is there any medicine containing ingredient Y?" (requires knowledge + visual matching).
   - **Question style**:
     - Questions do NOT need to end with a question mark (?). Use natural Korean phrasing.
     - Each question must be a **single-line Korean sentence**.
     - Questions must be **direct questions about the image**, NOT questions about how to answer or explain things.
     - Do NOT generate meta-questions about the dataset, training process, or this prompt itself.

3. **Positive vs negative sample design**
   - Positive questions:
     - Target medicines that actually exist in the image, or can be reasonably inferred from visual cues (imprint/color/shape/thickness) plus the reference list.
     - Focus on **uncertain identification scenarios** (when imprint is unclear).
   - Negative questions:
     - Use **common medicine names** that are not visible in the image and preferably not in the provided medicine list.
     - Example negative medicine names (do NOT assume they are present unless clearly visible and listed): 타이레놀, 이부프로펜, 아스피린, 파라세타몰, 아목시실린, 세파클러, 로키소닌, 게보린, 부루펜, 케토톱, 아세트아미노펜, 나프록센, 디클로페낙, 멜록시캠, 셀레콕시브, etc.
     - For each negative question, the answer must clearly explain that the medicine is **not present in the image**, based on both the visual evidence and the reference list.
</qa_requirements_core>

<answer_requirements_visual_reasoning>
1. **Visual cue priority (imprint > color/shape/thickness)**
   - When identifying a medicine in the image, use the following priority order:
     1) Imprint (letters/numbers/logo and their placement) - **most reliable**
     2) Color (or color combination)
     3) Tablet/capsule shape (round, oval, oblong, etc.)
     4) Thickness, size, and proportions
   - In answers, whenever possible, treat **imprint information** as the most important visual cue.

2. **When the imprint is clearly visible and matches**
   - If the imprint in the image **clearly matches** the imprint described in the medicine information:
     - State confidently that the corresponding medicine **is present** in the image.
     - Then add 1–2 more sentences giving concise key information about that medicine from the provided list, such as:
       - Main active ingredient(s)
       - High-level indication or use case
       - Dosage form/strength (only if given; do not invent details).

3. **When the imprint is unclear or not visible (CRITICAL)**
   - **MANDATORY**: If the imprint is not clearly visible, you MUST explicitly state that **100% certainty is not possible** and that identification is based on shape/color/size alone.
   - **CRITICAL**: When the imprint is not visible, do NOT identify a single specific medicine. Instead, provide **3-5 candidate medicines** that match the visual characteristics (color/shape/size), and explain that without the imprint, it is difficult to determine which one it is.
   - Example answer structures (describe in Korean when you generate):
     - "각인이 명확하게 보이지 않아 정확히 어떤 약인지 단정하기는 어렵습니다. 다만 하얀색 작은 원형 정제의 모양과 크기로 보아 [약명1], [약명2], [약명3] 등이 가능성이 있습니다."
     - "각인이 보이지 않아 100% 확실하게 판단할 수는 없지만, 하얀색 장방형 필름코팅정의 형태와 색상으로 보아 [약명1], [약명2], [약명3] 등이 후보가 될 수 있습니다."
     - "각인을 확인할 수 없어 정확한 식별은 어렵습니다. 노란색 캡슐의 경우 [약명1], [약명2], [약명3], [약명4] 등 여러 약이 같은 색상과 형태를 가지고 있어, 각인 없이는 특정 약을 확정하기 어렵습니다."
   - **NEVER** claim a single specific medicine when the imprint is not visible, even if color/shape/size match perfectly. Always provide multiple candidates (3-5 medicines) that share similar visual characteristics.

4. **When multiple candidates overlap or no match exists**
   - If there are many tablets with very similar color/shape/thickness so that a 1:1 mapping to a specific medicine is **not reliable**:
     - Answer that it is **difficult to determine**, and explain briefly that several medicines look similar in the image.
     - Example: "같은 색과 모양의 다른 약도 함께 있어 정확히 어떤 것이 [약명]인지 단정하기는 어렵습니다."
   - If none of the visual cues in the image match the description of a specific medicine in the list:
     - Clearly answer that the medicine **cannot be confirmed or does not appear to be present** based on this image alone.

5. **Answer length and composition**
   - Each answer must consist of **2–4 sentences in Korean**.
   - At least one sentence must describe the **visual evidence** (imprint/color/shape/thickness/position/count).
   - The remaining 1–3 sentences may incorporate relevant medicine information (ingredient/indication/dosage form), but always **ground the explanation in the image first**.
   - Always distinguish between:
     - Cases where presence is certain (imprint clearly visible and matches),
     - Cases where presence is plausible but uncertain (imprint not visible, identification based on shape/color/size only), and
     - Cases where presence cannot be confirmed or the medicine is clearly absent.
   - **CRITICAL**: When counting tablets/capsules, be accurate. If you cannot count accurately from the image, state the uncertainty (e.g., "대략 4~5캡슐 정도" instead of claiming exact numbers).
   - **CRITICAL**: NEVER describe visual elements (color, shape, size, count) that are NOT actually visible in the image. Only mention what you can actually see in the image. If you cannot see certain tablets/capsules clearly, state that explicitly rather than inventing descriptions.

6. **Medicine information usage**
   - **IMPORTANT**: The medicine information list is provided ONLY for dataset generation purposes. In real usage, users will ONLY provide the image, NOT the medicine information.
   - In answers, do NOT explicitly mention "제공된 정보에 따르면" or "약 정보 목록에 따르면". Instead, phrase it naturally as if you are identifying from the image alone (e.g., "이미지에 보이는 [특징]으로 보아...").
   - However, you can still use the medicine information to provide accurate ingredient/indication details in your answers, but frame it as knowledge about the identified medicine, not as reference to a provided list.
</answer_requirements_visual_reasoning>

<consistency_and_sanity_checks>
1. **Maintain consistency within a single image (critical)**
   - For the same image, **never make contradictory claims about the same medicine** across different Q/A pairs:
     - Example of forbidden behavior: one Q/A says “Medicine X is not visible in this image”, while another Q/A for the same image says “Medicine X is clearly visible in this image.”
   - For each distinct medicine in a given image, keep its status consistently as one of:
     - (a) Clearly present,
     - (b) Clearly absent, or
     - (c) Cannot be determined from visual information alone.

2. **Avoid self-contradiction in imprint/shape descriptions**
   - For tablets/capsules that obviously belong to the same visual group (same color/shape/size), do NOT produce mutually exclusive imprint descriptions, such as:
     - First Q/A: “The imprint on these tablets looks like ‘IDG’.”
     - Second Q/A: “The imprint on the same group of tablets looks like ‘S’.”
   - Once you have committed to a specific imprint-based identification (or lack thereof) for a visual group, **do not contradict it** in other Q/As for the same image.

3. **Avoid contradictions about capsule vs tablet presence**
   - If one Q/A states that “only white round tablets are visible and no oblong capsules can be seen”,
     then another Q/A for the same image must not claim that “an oblong capsule is clearly visible”.
   - In other words, do NOT invent new visual facts that contradict earlier descriptions; keep all Q/As for the same image mutually consistent.
</consistency_and_sanity_checks>

<output_format>
- Output **only** the question–answer pairs, nothing else.
- Do NOT include your reasoning process, bullet lists, or summaries.
- Format each pair exactly as:

Q: [질문 내용]
A: [답변 내용]

Q: [질문 내용]
A: [답변 내용]

- Do not number the questions.
- Do not write any text before the first "Q:".
- Generate all {target_total} pairs in this format.
</output_format>
"""

        # ---------- GPT‑5.1용 프롬프트 ----------
        if llm_type.lower().startswith("gpt"):
            prompt = f"""{base_body}

<meta_guidance_for_gpt5_1>
- You are GPT-5.1 optimized for:
  - **High-quality, instruction-following** generation suitable for fine-tuning datasets.
  - **Crisp but complete** answers: avoid unnecessary verbosity, but never omit key visual reasoning steps and uncertainty statements.
- Obey the <qa_requirements_core>, <answer_requirements_visual_reasoning>, and <output_format> sections strictly.
- Do NOT show chain-of-thought; apply the rules internally and only output final Q/A pairs.
</meta_guidance_for_gpt5_1>

Now generate all {target_total} Korean question–answer pairs.
"""
            return prompt

        # ---------- Qwen3‑VL용 프롬프트 ----------
        prompt = f"""{base_body}

<model_specific_guidance_for_qwen3_vl>
- You are Qwen3-VL generating **additional** question–answer pairs for the same image.
- You MUST:
  - Respect the list of previously generated questions shown above and **avoid repeating or paraphrasing them**.
  - Follow all rules in <qa_requirements_core>, <answer_requirements_visual_reasoning>, and <output_format>.
  - Start your output **directly with "Q:"** without any explanation or meta text.
- Do NOT output any internal thinking or reasoning tags.
</model_specific_guidance_for_qwen3_vl>

Now generate all {target_total} Korean question–answer pairs.
"""
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

    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """
        이미지를 base64로 인코딩
        
        Args:
            image_path: 이미지 파일 경로
        
        Returns:
            base64 인코딩된 이미지 문자열 (실패 시 None)
        """
        try:
            # 절대 경로로 변환
            if Path(image_path).is_absolute():
                abs_image_path = Path(image_path)
            else:
                abs_image_path = Path(__file__).parent / image_path
            
            if not abs_image_path.exists():
                logger.error(f"이미지 파일을 찾을 수 없습니다: {abs_image_path}")
                return None
            
            # 이미지 열기 및 리사이즈 (너무 크면 API 제한에 걸릴 수 있음)
            img = Image.open(abs_image_path)
            
            # 이미지가 너무 크면 리사이즈 (최대 2048x2048)
            max_size = 2048
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # JPEG로 변환하여 base64 인코딩
            buffer = io.BytesIO()
            # RGBA 모드면 RGB로 변환
            if img.mode == 'RGBA':
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3])  # alpha 채널을 마스크로 사용
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            img.save(buffer, format='JPEG', quality=95)
            img_bytes = buffer.getvalue()
            
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            return base64_image
            
        except Exception as e:
            logger.error(f"이미지 인코딩 중 오류 발생: {str(e)}")
            return None

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
        
        # 이미지를 base64로 인코딩
        base64_image = self._encode_image_to_base64(image_path)
        if base64_image is None:
            logger.error("이미지 인코딩 실패, 텍스트만으로 진행합니다.")
            # 이미지 인코딩 실패 시 기존 방식으로 fallback
            base64_image = None
        
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
            medicine_info=medicine_info,
            llm_type="gpt5.1",
        )

        system_prompt = """You are an expert at generating Korean VQA question-answer pairs for medicine blister pack images.
- Generate questions and answers in Korean only.
- Questions must be image-centered.
- DO NOT generate questions about the blister pack (약포/약폼) itself, such as questions about the blister pack's backside printing, packaging form, or design. Focus only on the medicines (알약/의약품) inside the blister pack.
- CRITICAL: Focus on questions that are ambiguous or difficult for users to confirm by themselves. Avoid questions that users can easily verify (e.g., counting tablets, seeing colors directly).
- CRITICAL: Use ALL medicines evenly in your questions. If there are multiple medicines in the medicine information list, you MUST generate questions about EACH medicine, not just one. DO NOT generate all questions about only one medicine - this is STRICTLY FORBIDDEN. Distribute questions evenly across all available medicines.
- CRITICAL: Avoid generating duplicate or very similar questions. Each question must be unique.
- STRICTLY FORBIDDEN: Do NOT generate meta-questions about how to answer or explain things. Examples of FORBIDDEN questions:
  - "어떻게 설명해 줄 수 있을까요?" (How can I explain this?)
  - "어떻게 답해야 할까요?" (How should I answer?)
  - "어떻게 대답하는 것이 적절할까요?" (How should I respond appropriately?)
  - "~라고 하면 어떻게 설명해 줄 수 있을까요?" (If someone asks ~, how can I explain?)
  - "~라는 요청이 들어오면 어떻게 답해야 할까요?" (If a request comes in asking ~, how should I answer?)
  - These are questions about AI response strategy, NOT direct questions about the medicine image that users would ask.
- IMPORTANT: Generate MANY negative sample questions (asking about medicines NOT in the list). Use question formats like "이 이미지에 [약명] 정제가 포함되어 있는지 확인해 주세요" or "이 이미지에 [약명] 정제가 포함되어 있나요". Use common medicine names like 타이레놀, 이부프로펜, 아스피린, 파라세타몰, 아목시실린, 게보린, 부루펜, 케토톱, 아세트아미노펜, 나프록센, 디클로페낙, 멜록시캠, 셀레콕시브, 로키소닌, 세파클러, etc.
- IMPORTANT: When the imprint is not clearly visible, you MUST state that 100% certainty is not possible. Do NOT identify a single specific medicine. Instead, provide 3-5 candidate medicines that match the visual characteristics, and explain that without the imprint, it is difficult to determine which one it is. Never claim a single specific medicine when identification is based on shape/color/size alone.
- CRITICAL: NEVER describe visual elements (color, shape, size, count, position) that are NOT actually visible in the image. Only mention what you can actually see in the image. If you cannot see certain tablets/capsules clearly, state that explicitly rather than inventing descriptions. This is a critical requirement to avoid hallucination.
- IMPORTANT: The medicine information list is provided ONLY for dataset generation. In real usage, users will ONLY provide the image. Do NOT explicitly mention "제공된 정보에 따르면" or "약 정보 목록에 따르면" in answers. Phrase answers as if identifying from the image alone.
- Questions do NOT need to end with a question mark (?). Use natural Korean phrasing.
- Questions must be DIRECT questions about the image content, NOT questions about how to answer or explain things.
- Answers should be 2-4 sentences in Korean, focusing on image observations first, then supplementing with medicine information when needed."""

        try:
            # 이미지가 있으면 vision API 사용, 없으면 텍스트만
            if base64_image:
                user_message = [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            else:
                user_message = prompt
            
            response = self.openai_client.chat.completions.create(
                model=self.api_config["openai"]["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_completion_tokens=self.api_config["openai"]["max_tokens"],
                temperature=self.api_config["openai"]["temperature"]
            )
            
            generated_text = response.choices[0].message.content.strip()
            qa_pairs = self._parse_qa_pairs(generated_text)
            
            if len(qa_pairs) < max_qa_pairs:
                logger.warning(f"GPT-5.1: 생성된 질문-답변 쌍이 {max_qa_pairs}개보다 작습니다 ({len(qa_pairs)}개)")
                breakpoint()
            
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
            medicine_info=medicine_info,
            llm_type="qwen3-vl",
        )

        try:
            # Qwen3-VL API 호출
            # Thinking 모델이 긴 reasoning을 생성할 수 있으므로 max_tokens를 늘림
            max_tokens_for_qa = max(self.api_config["qwen3vl"]["max_tokens"], 8192)
            
            # 이미지를 base64로 인코딩 (API가 로컬 경로를 직접 읽지 못할 수 있으므로)
            base64_image = self._encode_image_to_base64(str(abs_image_path))
            if base64_image is None:
                logger.error("Qwen3-VL: 이미지 인코딩 실패")
                return []
            
            payload = {
                "text": prompt,
                "images": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{base64_image}"
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
            logger.debug(f"생성된 원본 텍스트 (처음 500자): {generated_text[:500]}")
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
                logger.debug(f"질문-답변 부분 추출 (처음 1500자): {qa_text[:1500]}")
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

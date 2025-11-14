"""
데이터셋 변환 모듈
Qwen3-VL 학습 형식으로 데이터를 변환합니다.
"""

import json
import logging
import random
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetConverter:
    """데이터셋 변환 클래스"""
    
    def __init__(self, difficulty_distribution: Dict[str, float] = None):
        """
        Args:
            difficulty_distribution: 난이도 분포 {"easy": 0.5, "medium": 0.3, "hard": 0.2}
        """
        if difficulty_distribution is None:
            difficulty_distribution = {"easy": 0.5, "medium": 0.3, "hard": 0.2}
        
        self.difficulty_distribution = difficulty_distribution
        logger.info("데이터셋 변환기 초기화 완료")
    
    def _classify_difficulty(self, question: str) -> str:
        """
        질문의 난이도를 분류
        
        Args:
            question: 질문
        
        Returns:
            "easy", "medium", "hard" 중 하나
        """
        question_lower = question.lower()
        
        # 쉬운 질문: 단순 존재 확인, 색상 식별
        easy_keywords = ["있나요", "있나", "무엇인가요", "무엇", "색깔", "색상", "어떤 색"]
        if any(keyword in question_lower for keyword in easy_keywords):
            return "easy"
        
        # 어려운 질문: 복합적인 분석, 위치, 비교
        hard_keywords = ["차이점", "비교", "위치", "어디", "어느", "가장", "크기", "형태 설명"]
        if any(keyword in question_lower for keyword in hard_keywords):
            return "hard"
        
        # 중간: 개수 세기 등
        return "medium"
    
    def _assign_difficulty(self, questions: List[str]) -> List[Dict[str, any]]:
        """
        질문들에 난이도를 할당하고 분포에 맞게 조정
        
        Args:
            questions: 질문 리스트
        
        Returns:
            [{"question": str, "difficulty": str}, ...] 형식
        """
        # 각 질문의 난이도 분류
        classified = []
        for q in questions:
            difficulty = self._classify_difficulty(q)
            classified.append({"question": q, "difficulty": difficulty})
        
        # 난이도별 개수 계산
        total = len(classified)
        easy_count = int(total * self.difficulty_distribution["easy"])
        medium_count = int(total * self.difficulty_distribution["medium"])
        hard_count = total - easy_count - medium_count
        
        # 난이도별로 분류
        easy_questions = [q for q in classified if q["difficulty"] == "easy"]
        medium_questions = [q for q in classified if q["difficulty"] == "medium"]
        hard_questions = [q for q in classified if q["difficulty"] == "hard"]
        
        # 부족한 난이도는 다른 난이도에서 보충
        selected = []
        selected.extend(random.sample(easy_questions, min(easy_count, len(easy_questions))))
        remaining = easy_count - len(selected)
        if remaining > 0:
            selected.extend(random.sample(medium_questions, min(remaining, len(medium_questions))))
            remaining = easy_count - len(selected)
            if remaining > 0:
                selected.extend(random.sample(hard_questions, min(remaining, len(hard_questions))))
        
        selected.extend(random.sample(medium_questions, min(medium_count, len(medium_questions))))
        remaining = medium_count - (len(selected) - easy_count)
        if remaining > 0:
            selected.extend(random.sample(hard_questions, min(remaining, len(hard_questions))))
        
        selected.extend(random.sample(hard_questions, min(hard_count, len(hard_questions))))
        
        # 중복 제거 (질문 기준)
        seen_questions = set()
        final_selected = []
        for item in selected:
            if item["question"] not in seen_questions:
                seen_questions.add(item["question"])
                final_selected.append(item)
        
        return final_selected
    
    def convert_to_qwen_format(self, image_path: str, qa_pairs: List[Dict[str, str]]) -> List[Dict]:
        """
        Qwen3-VL 학습 형식으로 변환
        
        Args:
            image_path: 이미지 경로
            qa_pairs: [{"question": str, "answer": str}, ...] 형식의 리스트
        
        Returns:
            Qwen3-VL 형식의 데이터 리스트
        """
        if not qa_pairs:
            return []
        
        # 난이도 할당
        questions_with_difficulty = self._assign_difficulty([qa["question"] for qa in qa_pairs])
        
        # Qwen3-VL 형식으로 변환
        qwen_data = []
        
        for qa in qa_pairs:
            # 해당 질문의 난이도 찾기
            difficulty = "medium"  # 기본값
            for qwd in questions_with_difficulty:
                if qwd["question"] == qa["question"]:
                    difficulty = qwd["difficulty"]
                    break
            
            # Qwen3-VL 형식
            qwen_item = {
                "image": image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{qa['question']}"
                    },
                    {
                        "from": "gpt",
                        "value": qa["answer"]
                    }
                ],
                "difficulty": difficulty  # 메타데이터로 저장
            }
            
            qwen_data.append(qwen_item)
        
        logger.info(f"{len(qwen_data)}개의 데이터를 Qwen3-VL 형식으로 변환 완료")
        return qwen_data
    
    def save_dataset(self, dataset: List[Dict], output_path: str):
        """
        데이터셋을 JSON 파일로 저장
        
        Args:
            dataset: 데이터셋 리스트
            output_path: 출력 파일 경로
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"데이터셋 저장 완료: {output_path} ({len(dataset)}개 항목)")
    
    def load_dataset(self, input_path: str) -> List[Dict]:
        """
        데이터셋을 JSON 파일에서 로드
        
        Args:
            input_path: 입력 파일 경로
        
        Returns:
            데이터셋 리스트
        """
        input_file = Path(input_path)
        
        if not input_file.exists():
            logger.warning(f"파일을 찾을 수 없습니다: {input_path}")
            return []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        logger.info(f"데이터셋 로드 완료: {input_path} ({len(dataset)}개 항목)")
        return dataset
    
    def merge_datasets(self, datasets: List[List[Dict]]) -> List[Dict]:
        """
        여러 데이터셋을 병합
        
        Args:
            datasets: 데이터셋 리스트들의 리스트
        
        Returns:
            병합된 데이터셋
        """
        merged = []
        for dataset in datasets:
            merged.extend(dataset)
        
        logger.info(f"{len(datasets)}개의 데이터셋 병합 완료: 총 {len(merged)}개 항목")
        return merged


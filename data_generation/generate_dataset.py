"""
메인 파이프라인
전체 워크플로우를 조율하여 데이터셋을 생성합니다.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

# 환경변수 로드 (.env 파일)
load_dotenv(Path(__file__).parent / ".env")

# 모듈 import 경로 설정
sys.path.insert(0, str(Path(__file__).parent))

from data_collector import DataCollector
from question_generator import QuestionGenerator
from reviewer import Reviewer
from answer_generator import AnswerGenerator
from dataset_converter import DatasetConverter

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DatasetGenerator:
    """데이터셋 생성 메인 클래스"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = config_path
        
        # 각 모듈 초기화
        logger.info("모듈 초기화 시작...")
        self.data_collector = DataCollector(config_path)
        self.question_generator = QuestionGenerator(config_path)
        self.reviewer = Reviewer(config_path)
        self.answer_generator = AnswerGenerator(config_path)
        
        question_generation_config = self.question_generator.config.get("question_generation", {})
        difficulty_dist = question_generation_config.get("difficulty_distribution", {})
        self.dataset_converter = DatasetConverter(difficulty_dist)
        
        self.output_config = self.question_generator.config.get("output", {})
        logger.info("모듈 초기화 완료")
    
    def process_single_image(self, data_item: Dict) -> List[Dict]:
        """
        단일 이미지에 대한 전체 처리 파이프라인
        
        Args:
            data_item: 데이터 아이템
        
        Returns:
            Qwen3-VL 형식의 데이터 리스트
        """
        image_path = data_item["image_path"]
        logger.info(f"이미지 처리 시작: {image_path}")
        
        try:
            # 1. 질문-답변 쌍 생성 (GPT-5.1과 Qwen3-VL-8B-Thinking)
            logger.info("1단계: 질문-답변 쌍 생성")
            qa_pairs_dict = self.question_generator.generate_questions(data_item)
            
            gpt_qa_pairs = qa_pairs_dict.get("gpt", [])
            qwen_qa_pairs = qa_pairs_dict.get("qwen3vl", [])
            
            # 전체 질문-답변 쌍 수 제한
            max_total = self.question_generator.question_config.get("max_total_questions", 20)
            force_per_model = self.question_generator.question_config.get("force_questions_per_model", False)
            
            # 각 모델에서 균등하게 가져오기
            total_qa_pairs = len(gpt_qa_pairs) + len(qwen_qa_pairs)
            if total_qa_pairs > max_total:
                if force_per_model:
                    logger.info(
                        f"  - force_questions_per_model 활성화로 {total_qa_pairs}개 질문-답변 쌍을 그대로 유지 (max_total={max_total})"
                    )
                else:
                    # 비율에 맞춰 제한
                    gpt_ratio = len(gpt_qa_pairs) / total_qa_pairs if total_qa_pairs > 0 else 0.5
                    gpt_limit = max(1, int(max_total * gpt_ratio))
                    qwen_limit = max_total - gpt_limit
                    
                    gpt_qa_pairs = gpt_qa_pairs[:gpt_limit]
                    qwen_qa_pairs = qwen_qa_pairs[:qwen_limit]
                    logger.info(f"  - 질문-답변 쌍 수 제한 적용: 총 {max_total}개로 제한")
            
            logger.info(f"  - GPT-5.1: {len(gpt_qa_pairs)}개 질문-답변 쌍 생성")
            logger.info(f"  - Qwen3-VL-8B-Thinking: {len(qwen_qa_pairs)}개 질문-답변 쌍 생성")
            
            if not gpt_qa_pairs and not qwen_qa_pairs:
                logger.warning("생성된 질문-답변 쌍이 없습니다.")
                return []
            
            # 2. 상호 검수 (질문-답변 쌍 검수)
            logger.info("2단계: 상호 검수 (hallucination 체크 + 완전한 문장 체크)")
            
            reviewed_gpt = []
            reviewed_qwen = []
            
            # GPT 질문-답변 쌍 검수 (Qwen3-VL로)
            if gpt_qa_pairs:
                reviewed_gpt = self.reviewer.review_qa_pairs(
                    gpt_qa_pairs,
                    image_path,
                    data_item.get("medicine_info", []),
                    "gpt"
                )
            
            # Qwen3-VL 질문-답변 쌍 검수 (GPT로)
            if qwen_qa_pairs:
                reviewed_qwen = self.reviewer.review_qa_pairs(
                    qwen_qa_pairs,
                    image_path,
                    data_item.get("medicine_info", []),
                    "qwen3vl"
                )
            
            # 검수 통과한 질문-답변 쌍만 추출
            approved_qa_pairs = []
            for reviewed in reviewed_gpt + reviewed_qwen:
                if reviewed.get("approved", False):
                    approved_qa_pairs.append({
                        "question": reviewed["question"],
                        "answer": reviewed["answer"]
                    })
            
            gpt_approved = sum(1 for qa in reviewed_gpt if qa.get("approved", False))
            qwen_approved = sum(1 for qa in reviewed_qwen if qa.get("approved", False))
            
            logger.info(f"  - GPT 질문-답변 쌍 검수 통과: {gpt_approved}/{len(gpt_qa_pairs)}")
            logger.info(f"  - Qwen3-VL 질문-답변 쌍 검수 통과: {qwen_approved}/{len(qwen_qa_pairs)}")
            
            if not approved_qa_pairs:
                logger.warning("검수 통과한 질문-답변 쌍이 없습니다.")
                return []
            
            # 3. Qwen3-VL 형식으로 변환
            logger.info("3단계: 데이터셋 형식 변환")
            qwen_data = self.dataset_converter.convert_to_qwen_format(image_path, approved_qa_pairs)
            
            logger.info(f"이미지 처리 완료: {image_path} ({len(qwen_data)}개 항목 생성)")
            return qwen_data
            
        except Exception as e:
            logger.error(f"이미지 처리 중 오류 발생: {image_path}, {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def generate_dataset(self, num_images: int = None, checkpoint_interval: int = 10):
        """
        데이터셋 생성 메인 함수
        
        Args:
            num_images: 처리할 이미지 수 (None이면 config에서 가져옴)
            checkpoint_interval: 체크포인트 저장 간격
        """
        logger.info("=" * 80)
        logger.info("데이터셋 생성 시작")
        logger.info("=" * 80)
        
        # 데이터 수집
        logger.info("데이터 수집 시작...")
        collected_data = self.data_collector.collect_data(num_images)
        
        if not collected_data:
            logger.error("수집된 데이터가 없습니다.")
            return
        
        logger.info(f"총 {len(collected_data)}개의 이미지 데이터 수집 완료")
        
        # 각 이미지 처리
        all_dataset = []
        checkpoint_dir = Path(__file__).parent / self.output_config.get("checkpoint_dir", "checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        for idx, data_item in enumerate(collected_data, 1):
            logger.info(f"\n[{idx}/{len(collected_data)}] 이미지 처리 중...")
            
            # 단일 이미지 처리
            qwen_data = self.process_single_image(data_item)
            all_dataset.extend(qwen_data)
            
            # 체크포인트 저장
            if idx % checkpoint_interval == 0:
                checkpoint_file = checkpoint_dir / f"checkpoint_{idx}.json"
                self.dataset_converter.save_dataset(all_dataset, str(checkpoint_file))
                logger.info(f"체크포인트 저장: {checkpoint_file}")
        
        # 최종 데이터셋 저장
        logger.info("\n" + "=" * 80)
        logger.info("최종 데이터셋 저장")
        logger.info("=" * 80)
        
        output_file = Path(__file__).parent / self.output_config.get("dataset_file", "dataset.json")
        self.dataset_converter.save_dataset(all_dataset, str(output_file))
        
        logger.info(f"\n데이터셋 생성 완료!")
        logger.info(f"  - 처리된 이미지 수: {len(collected_data)}")
        logger.info(f"  - 생성된 데이터 항목 수: {len(all_dataset)}")
        logger.info(f"  - 출력 파일: {output_file}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Qwen3-VL Fine-tuning 데이터셋 생성")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="설정 파일 경로"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="처리할 이미지 수 (기본값: config에서 가져옴)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="체크포인트 저장 간격"
    )
    
    args = parser.parse_args()
    
    # 데이터셋 생성기 초기화 및 실행
    generator = DatasetGenerator(args.config)
    generator.generate_dataset(
        num_images=args.num_images,
        checkpoint_interval=args.checkpoint_interval
    )


if __name__ == "__main__":
    main()


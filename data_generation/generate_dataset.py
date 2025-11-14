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
        
        # 설정 로드
        import yaml
        with open(Path(__file__).parent / config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        difficulty_dist = config.get("question_generation", {}).get("difficulty_distribution", {})
        self.dataset_converter = DatasetConverter(difficulty_dist)
        
        self.output_config = config.get("output", {})
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
            # 1. 질문 생성 (GPT-5.1과 Qwen3-VL-8B-Thinking)
            logger.info("1단계: 질문 생성")
            questions_dict = self.question_generator.generate_questions(data_item)
            
            gpt_questions = questions_dict.get("gpt", [])
            qwen_questions = questions_dict.get("qwen3vl", [])
            
            # 전체 질문 수 제한
            import yaml
            with open(Path(__file__).parent / self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            max_total = config.get("question_generation", {}).get("max_total_questions", 15)
            
            # 각 모델에서 균등하게 가져오기
            total_questions = len(gpt_questions) + len(qwen_questions)
            if total_questions > max_total:
                # 비율에 맞춰 제한
                gpt_ratio = len(gpt_questions) / total_questions if total_questions > 0 else 0.5
                gpt_limit = max(1, int(max_total * gpt_ratio))
                qwen_limit = max_total - gpt_limit
                
                gpt_questions = gpt_questions[:gpt_limit]
                qwen_questions = qwen_questions[:qwen_limit]
                logger.info(f"  - 질문 수 제한 적용: 총 {max_total}개로 제한")
            
            logger.info(f"  - GPT-5.1: {len(gpt_questions)}개 질문 생성")
            logger.info(f"  - Qwen3-VL-8B-Thinking: {len(qwen_questions)}개 질문 생성")
            
            if not gpt_questions and not qwen_questions:
                logger.warning("생성된 질문이 없습니다.")
                return []
            
            # 2. 상호 검수
            logger.info("2단계: 상호 검수")
            reviewed_gpt = []
            reviewed_qwen = []
            
            if gpt_questions:
                reviewed_gpt = self.reviewer.review_questions(gpt_questions, image_path, "gpt")
            
            if qwen_questions:
                reviewed_qwen = self.reviewer.review_questions(qwen_questions, image_path, "qwen3vl")
            
            # 검수 통과한 질문만 추출
            approved_gpt = [q["question"] for q in reviewed_gpt if q["approved"]]
            approved_qwen = [q["question"] for q in reviewed_qwen if q["approved"]]
            
            logger.info(f"  - GPT 질문 검수 통과: {len(approved_gpt)}/{len(gpt_questions)}")
            logger.info(f"  - Qwen3-VL 질문 검수 통과: {len(approved_qwen)}/{len(qwen_questions)}")
            
            # 3. 중복 제거 (Qwen3-VL-8B-Thinking이 수행)
            logger.info("3단계: 중복 제거")
            all_approved_questions = approved_gpt + approved_qwen
            
            if len(all_approved_questions) > 1:
                deduplicated_questions = self.reviewer.deduplicate_questions(
                    all_approved_questions,
                    image_path
                )
            else:
                deduplicated_questions = all_approved_questions
            
            logger.info(f"  - 중복 제거 후: {len(deduplicated_questions)}개 질문")
            
            if not deduplicated_questions:
                logger.warning("중복 제거 후 질문이 없습니다.")
                return []
            
            # 4. 답변 생성
            logger.info("4단계: 답변 생성")
            qa_pairs = self.answer_generator.generate_answers(
                deduplicated_questions,
                image_path,
                use_qwen=True  # Qwen3-VL 사용
            )
            
            logger.info(f"  - 답변 생성 완료: {len(qa_pairs)}개")
            
            if not qa_pairs:
                logger.warning("생성된 답변이 없습니다.")
                return []
            
            # 5. Qwen3-VL 형식으로 변환
            logger.info("5단계: 데이터셋 형식 변환")
            qwen_data = self.dataset_converter.convert_to_qwen_format(image_path, qa_pairs)
            
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


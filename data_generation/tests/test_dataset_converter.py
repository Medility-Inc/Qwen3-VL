"""
데이터셋 변환 모듈 테스트
"""

import unittest
import tempfile
import shutil
from pathlib import Path

# 상위 디렉토리에서 모듈 import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_converter import DatasetConverter


class TestDatasetConverter(unittest.TestCase):
    """DatasetConverter 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.converter = DatasetConverter()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_classify_difficulty(self):
        """난이도 분류 테스트"""
        # 쉬운 질문
        easy_q = "약포에 파란색 약이 있나요?"
        self.assertEqual(self.converter._classify_difficulty(easy_q), "easy")
        
        # 중간 질문
        medium_q = "약포에 몇 개의 알약이 있나요?"
        self.assertEqual(self.converter._classify_difficulty(medium_q), "medium")
        
        # 어려운 질문
        hard_q = "파란색 약과 빨간색 약의 차이점은 무엇인가요?"
        self.assertEqual(self.converter._classify_difficulty(hard_q), "hard")
    
    def test_assign_difficulty(self):
        """난이도 할당 테스트"""
        questions = [
            "약포에 파란색 약이 있나요?",
            "약포에 몇 개의 알약이 있나요?",
            "파란색 약과 빨간색 약의 차이점은 무엇인가요?",
            "가장 큰 알약의 색깔은 무엇인가요?",
            "원형 알약이 포함되어 있나요?"
        ]
        
        result = self.converter._assign_difficulty(questions)
        
        self.assertGreater(len(result), 0)
        for item in result:
            self.assertIn("question", item)
            self.assertIn("difficulty", item)
            self.assertIn(item["difficulty"], ["easy", "medium", "hard"])
    
    def test_convert_to_qwen_format(self):
        """Qwen3-VL 형식 변환 테스트"""
        image_path = "images/test.jpg"
        qa_pairs = [
            {"question": "약포에 파란색 약이 있나요?", "answer": "예, 있습니다."},
            {"question": "약포에 몇 개의 알약이 있나요?", "answer": "3개입니다."}
        ]
        
        result = self.converter.convert_to_qwen_format(image_path, qa_pairs)
        
        self.assertEqual(len(result), len(qa_pairs))
        for item in result:
            self.assertIn("image", item)
            self.assertIn("conversations", item)
            self.assertEqual(len(item["conversations"]), 2)
            self.assertEqual(item["conversations"][0]["from"], "human")
            self.assertEqual(item["conversations"][1]["from"], "gpt")
            self.assertIn("<image>", item["conversations"][0]["value"])
    
    def test_save_and_load_dataset(self):
        """데이터셋 저장 및 로드 테스트"""
        dataset = [
            {
                "image": "images/test.jpg",
                "conversations": [
                    {"from": "human", "value": "<image>\n테스트 질문"},
                    {"from": "gpt", "value": "테스트 답변"}
                ]
            }
        ]
        
        output_path = Path(self.temp_dir) / "test_dataset.json"
        
        # 저장
        self.converter.save_dataset(dataset, str(output_path))
        self.assertTrue(output_path.exists())
        
        # 로드
        loaded = self.converter.load_dataset(str(output_path))
        self.assertEqual(len(loaded), len(dataset))
        self.assertEqual(loaded[0]["image"], dataset[0]["image"])
    
    def test_merge_datasets(self):
        """데이터셋 병합 테스트"""
        dataset1 = [{"image": "img1.jpg", "conversations": []}]
        dataset2 = [{"image": "img2.jpg", "conversations": []}]
        dataset3 = [{"image": "img3.jpg", "conversations": []}]
        
        merged = self.converter.merge_datasets([dataset1, dataset2, dataset3])
        
        self.assertEqual(len(merged), 3)


if __name__ == '__main__':
    unittest.main()


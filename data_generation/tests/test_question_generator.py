"""
질문 생성 모듈 테스트
"""

import unittest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import yaml

# 상위 디렉토리에서 모듈 import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from question_generator import QuestionGenerator


class TestQuestionGenerator(unittest.TestCase):
    """QuestionGenerator 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        
        # 테스트용 config 파일 생성
        test_config = {
            "api": {
                "openai": {
                    "model": "gpt-4o",
                    "max_tokens": 2048,
                    "temperature": 0.7
                },
                "qwen3vl": {
                    "base_url": "http://localhost:1119",
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "top_p": 0.8
                }
            },
            "question_generation": {
                "difficulty_distribution": {
                    "easy": 0.5,
                    "medium": 0.3,
                    "hard": 0.2
                }
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f)
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_config(self):
        """설정 파일 로드 테스트"""
        generator = QuestionGenerator(str(self.config_path))
        self.assertIsNotNone(generator.config)
        self.assertEqual(generator.config["api"]["openai"]["model"], "gpt-4o")
    
    def test_build_medicine_context(self):
        """의약품 정보 컨텍스트 빌드 테스트"""
        generator = QuestionGenerator(str(self.config_path))
        
        medicine_info = [
            {
                "item_name": "테스트약",
                "drug_shape": "원형",
                "print_front": "TEST"
            }
        ]
        
        context = generator._build_medicine_context(medicine_info)
        self.assertIn("테스트약", context)
        self.assertIn("원형", context)
    
    def test_build_medicine_context_empty(self):
        """빈 의약품 정보 테스트"""
        generator = QuestionGenerator(str(self.config_path))
        
        context = generator._build_medicine_context([])
        self.assertIn("없습니다", context)


if __name__ == '__main__':
    unittest.main()


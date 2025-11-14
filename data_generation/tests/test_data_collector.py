"""
데이터 수집 모듈 테스트
"""

import unittest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd

# 상위 디렉토리에서 모듈 import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collector import DataCollector


class TestDataCollector(unittest.TestCase):
    """DataCollector 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        
        # 테스트용 config 파일 생성
        import yaml
        test_config = {
            "database": {
                "host": "localhost",
                "username": "test_user",
                "password": "test_pass",
                "schema": "test_db",
                "port": 3306
            },
            "data_collection": {
                "split_version_id": 128,
                "min_date": "2025-08-01",
                "side": "inner",
                "random_sample_size": 10,
                "image_dir": "test_images"
            },
            "output": {
                "used_images_file": "test_used_images.json"
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f)
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_config(self):
        """설정 파일 로드 테스트"""
        collector = DataCollector(str(self.config_path))
        self.assertIsNotNone(collector.config)
        self.assertEqual(collector.config["database"]["host"], "localhost")
    
    def test_load_used_images(self):
        """사용된 이미지 로드 테스트"""
        collector = DataCollector(str(self.config_path))
        self.assertIsInstance(collector.used_image_ids, set)
    
    @patch('data_collector.pymysql.connect')
    def test_get_db_connection(self, mock_connect):
        """DB 연결 테스트"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        collector = DataCollector(str(self.config_path))
        conn = collector.get_db_connection()
        
        mock_connect.assert_called_once()
        self.assertEqual(conn, mock_conn)
    
    @patch('data_collector.DataCollector.get_db_connection')
    def test_sample_images(self, mock_get_conn):
        """이미지 샘플링 테스트"""
        # Mock 설정
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"id": 1, "gt_id": 1, "width": 100, "height": 100, "url": "http://test.com/img1.jpg"},
            {"id": 2, "gt_id": 2, "width": 200, "height": 200, "url": "http://test.com/img2.jpg"},
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        collector = DataCollector(str(self.config_path))
        collector.used_image_ids = set()  # 빈 세트로 설정
        
        result = collector.sample_images(2)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertLessEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()


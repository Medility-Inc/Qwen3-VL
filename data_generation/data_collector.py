"""
데이터 수집 모듈
DB에서 이미지와 의약품 정보를 수집하고, 이미지를 다운로드하여 로컬에 저장합니다.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from io import BytesIO
from dotenv import load_dotenv

import pymysql
import pandas as pd
import requests
from PIL import Image
import yaml

# 환경변수 로드
load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger(__name__)


class DataCollector:
    """데이터 수집 클래스"""
    
    def __init__(self, config_path: str = "config.yaml", run_folder: Optional[Path] = None):
        """
        Args:
            config_path: 설정 파일 경로
            run_folder: 실행 폴더 경로 (None이면 기본 경로 사용)
        """
        self.config = self._load_config(config_path)
        self.db_config = self.config["database"]
        self.data_config = self.config["data_collection"]
        self.output_config = self.config["output"]
        
        # 실행 폴더 설정
        self.run_folder = run_folder
        
        # 이미지 저장 디렉토리 (기본값, 나중에 set_run_folder로 업데이트 가능)
        if run_folder:
            self.image_dir = run_folder / "images"
        else:
            self.image_dir = Path(__file__).parent / self.data_config["image_dir"]
        self.image_dir.mkdir(parents=True, exist_ok=True)
        
        # 사용된 이미지 추적 파일
        self.used_images_file = Path(__file__).parent / self.output_config["used_images_file"]
        self.used_image_ids = self._load_used_images()
        
        logger.info(f"이미지 저장 디렉토리: {self.image_dir}")
        logger.info(f"사용된 이미지 수: {len(self.used_image_ids)}")
    
    def set_run_folder(self, run_folder: Path):
        """
        실행 폴더 설정 및 이미지 디렉토리 업데이트
        
        Args:
            run_folder: 실행 폴더 경로
        """
        self.run_folder = run_folder
        self.image_dir = run_folder / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"이미지 저장 디렉토리 업데이트: {self.image_dir}")
    
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        config_file = Path(__file__).parent / config_path
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 환경변수 치환
        for key, value in config["database"].items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_key = value[2:-1]
                config["database"][key] = os.getenv(env_key, value)
        
        return config
    
    def _load_used_images(self) -> set:
        """사용된 이미지 ID 로드"""
        if self.used_images_file.exists():
            with open(self.used_images_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get("used_image_ids", []))
        return set()
    
    def _save_used_images(self):
        """사용된 이미지 ID 저장"""
        data = {
            "used_image_ids": list(self.used_image_ids),
            "total_count": len(self.used_image_ids)
        }
        with open(self.used_images_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_db_connection(self):
        """DB 연결 생성"""
        return pymysql.connect(
            host=self.db_config["host"],
            user=self.db_config["username"],
            password=self.db_config["password"],
            database=self.db_config["schema"],
            port=self.db_config["port"],
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
    
    def sample_images(self, n: Optional[int] = None) -> pd.DataFrame:
        """
        DB에서 랜덤 이미지 샘플링
        
        Args:
            n: 샘플링할 이미지 수 (None이면 config에서 가져옴)
        
        Returns:
            이미지 정보가 담긴 DataFrame
        """
        if n is None:
            n = self.data_config["random_sample_size"]
        
        logger.info(f"DB에서 {n}개의 이미지 샘플링 시작...")
        
        with self.get_db_connection().cursor() as cursor:
            # 사용되지 않은 이미지만 조회
            query = """
                SELECT i.id, gt.id AS gt_id, i.width, i.height, i.url
                FROM image i
                JOIN groundtruth gt ON i.id = gt.image_id
                JOIN split_image si ON gt.id = si.groundtruth_id
                JOIN split_version sv ON si.split_version_id = sv.id
                WHERE sv.id = %s
                    AND i.created_at >= %s
                    AND JSON_UNQUOTE(JSON_EXTRACT(i.attributes, '$.side')) = %s
                    AND i.id NOT IN ({})
            """.format(','.join(map(str, self.used_image_ids)) if self.used_image_ids else '0')
            
            cursor.execute(query, (
                self.data_config["split_version_id"],
                self.data_config["min_date"],
                self.data_config["side"]
            ))
            all_images = pd.DataFrame(cursor.fetchall())
        
        if len(all_images) == 0:
            logger.warning("샘플링할 이미지가 없습니다.")
            return pd.DataFrame()
        
        # 랜덤 샘플링
        n = min(n, len(all_images))
        sampled = all_images.sample(n=n, random_state=None).reset_index(drop=True)
        
        logger.info(f"{len(sampled)}개의 이미지 샘플링 완료")
        return sampled
    
    def get_medicine_info(self, gt_ids: List[int]) -> pd.DataFrame:
        """
        의약품 정보 조회
        
        Args:
            gt_ids: groundtruth ID 리스트
        
        Returns:
            의약품 정보가 담긴 DataFrame
        """
        if not gt_ids:
            return pd.DataFrame()
        
        logger.info(f"{len(gt_ids)}개의 groundtruth에 대한 의약품 정보 조회...")
        
        with self.get_db_connection().cursor() as cursor:
            query = f"""
                SELECT 
                    gt.id AS gt_id, gts.id AS gts_id, em.id AS em_id,
                    gts.rle_encoded_mask, gts.x_min, gts.y_min, gts.width AS box_width, gts.height AS box_height,
                    gts.occluded, gts.size, JSON_UNQUOTE(JSON_EXTRACT(gts.attributes, '$.broken_minor')) AS broken_minor,
                    em.item_name, em.drug_shape, em.print_front, em.print_back, em.thick, em.class_no, em.form_code_name, 
                    em.onesglobal_item_name, em.onesglobal_material_name, em.onesglobal_storage, em.onesglobal_valid_term, em.onesglobal_indication,
                    em.onesglobal_ethical_type, em.onesglobal_ingredient_ko, em.onesglobal_form_type, em.onesglobal_route, em.onesglobal_pack_unit
                FROM gt_shape gts
                JOIN groundtruth gt ON gts.groundtruth_id = gt.id
                JOIN enhanced_medicine em ON gts.label_id = em.id
                WHERE gt.id IN ({', '.join(map(str, gt_ids))})
            """
            
            cursor.execute(query)
            results = pd.DataFrame(cursor.fetchall())
        
        logger.info(f"{len(results)}개의 의약품 정보 조회 완료")
        return results
    
    def download_image(self, url: str, image_id: int) -> Optional[str]:
        """
        이미지 다운로드 및 로컬 저장
        
        Args:
            url: 이미지 URL
            image_id: 이미지 ID
        
        Returns:
            저장된 이미지의 상대 경로 (실패 시 None)
        """
        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                logger.warning(f"이미지 다운로드 실패: {url} (Status: {response.status_code})")
                return None
            
            # 이미지 확장자 확인
            image = Image.open(BytesIO(response.content))
            ext = image.format.lower() if image.format else 'jpg'
            if ext not in ['jpg', 'jpeg', 'png']:
                ext = 'jpg'
            
            # 파일명 생성
            filename = f"{image_id}.{ext}"
            filepath = self.image_dir / filename
            
            # 이미지 저장
            image.save(filepath, format=ext.upper())
            
            # 상대 경로 반환 (실행 폴더 기준)
            if self.run_folder:
                # 실행 폴더 기준 상대 경로
                relative_path = f"images/{filename}"
            else:
                # 기본 경로 기준 상대 경로
                relative_path = f"{self.data_config['image_dir']}/{filename}"
            logger.debug(f"이미지 저장 완료: {relative_path}")
            return relative_path
            
        except Exception as e:
            logger.error(f"이미지 다운로드 중 오류 발생: {url}, {str(e)}")
            return None
    
    def collect_data(self, n: Optional[int] = None) -> List[Dict]:
        """
        데이터 수집 메인 함수
        
        Args:
            n: 샘플링할 이미지 수
        
        Returns:
            수집된 데이터 리스트
        """
        # 이미지 샘플링
        images_df = self.sample_images(n)
        if len(images_df) == 0:
            return []
        
        # 의약품 정보 조회
        gt_ids = images_df["gt_id"].unique().tolist()
        medicine_df = self.get_medicine_info(gt_ids)
        
        # 데이터 조합
        collected_data = []
        for _, img_row in images_df.iterrows():
            # 이미지 다운로드
            local_path = self.download_image(img_row["url"], img_row["id"])
            if local_path is None:
                logger.warning(f"이미지 다운로드 실패: {img_row['id']}")
                continue
            
            # 해당 이미지의 의약품 정보
            medicine_info = medicine_df[medicine_df["gt_id"] == img_row["gt_id"]]
            
            data_item = {
                "image_id": int(img_row["id"]),
                "gt_id": int(img_row["gt_id"]),
                "image_path": local_path,  # 상대 경로 (예: images/001.jpg)
                "image_url": img_row["url"],
                "image_width": int(img_row["width"]),
                "image_height": int(img_row["height"]),
                "medicine_info": medicine_info.to_dict('records') if len(medicine_info) > 0 else []
            }
            
            collected_data.append(data_item)
            self.used_image_ids.add(int(img_row["id"]))
        
        # 사용된 이미지 ID 저장
        self._save_used_images()
        
        logger.info(f"{len(collected_data)}개의 데이터 수집 완료")
        return collected_data


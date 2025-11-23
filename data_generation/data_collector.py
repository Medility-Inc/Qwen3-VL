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
    
    def get_db_connection(self, schema: str = "data-catalog"):
        """DB 연결 생성
        
        Args:
            schema: 데이터베이스 스키마
        """
        return pymysql.connect(
            host=self.db_config["host"],
            user=self.db_config["username"],
            password=self.db_config["password"],
            database=schema,
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
            이미지 정보와 gt_shape 정보가 담긴 DataFrame (하나의 gt에 여러 gt_shape가 있으면 여러 행 반환)
        """
        if n is None:
            n = self.data_config["random_sample_size"]
        
        logger.info(f"DB에서 {n}개의 이미지 샘플링 시작...")
        
        with self.get_db_connection(schema="data-catalog").cursor() as cursor:
            # 사용되지 않은 이미지만 조회하고, label_id=1이 있는 gt는 제외
            # gt_shape 정보도 함께 가져옴 (여러 gt_shape 포함)
            query = """
                SELECT 
                    i.id, gt.id AS gt_id, i.width, i.height, i.url,
                    gts.id AS gts_id, l.id AS label_id, l.medicine_id AS medicine_id, l.reference_image_url AS reference_image_url,
                    gts.rle_encoded_mask, gts.x_min, gts.y_min, gts.width AS box_width, gts.height AS box_height,
                    gts.occluded, gts.size, JSON_UNQUOTE(JSON_EXTRACT(gts.attributes, '$.broken_minor')) AS broken_minor
                FROM image i
                JOIN groundtruth gt ON i.id = gt.image_id
                JOIN split_image si ON gt.id = si.groundtruth_id
                JOIN split_version sv ON si.split_version_id = sv.id
                LEFT JOIN gt_shape gts ON gt.id = gts.groundtruth_id
                LEFT JOIN label l ON gts.label_id = l.id
                WHERE sv.id = %s
                    AND i.created_at >= %s
                    AND JSON_UNQUOTE(JSON_EXTRACT(i.attributes, '$.side')) = %s
                    AND i.id NOT IN ({})
                    AND gt.id NOT IN (
                        SELECT DISTINCT gts2.groundtruth_id 
                        FROM gt_shape gts2 
                        JOIN label l2 ON gts2.label_id = l2.id 
                        WHERE l2.id = 1
                    )
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
        
        # 고유한 이미지 ID 목록 추출 (하나의 이미지에 여러 gt_shape가 있을 수 있음)
        unique_image_ids = all_images["id"].unique()
        
        # 랜덤 샘플링 (이미지 단위로)
        n = min(n, len(unique_image_ids))
        sampled_image_ids = pd.Series(unique_image_ids).sample(n=n, random_state=None).tolist()
        
        # 샘플링된 이미지 ID에 해당하는 모든 행 반환 (gt_shape 포함)
        sampled = all_images[all_images["id"].isin(sampled_image_ids)].reset_index(drop=True)
        
        logger.info(f"{len(sampled_image_ids)}개의 이미지 샘플링 완료 (총 {len(sampled)}개 행, gt_shape 포함)")
        return sampled
    
    def get_medicine_info(self, gt_shape_df: pd.DataFrame) -> pd.DataFrame:
        """
        의약품 정보 조회
        
        Args:
            gt_shape_df: sample_images에서 반환된 gt_shape 정보가 포함된 DataFrame
        
        Returns:
            의약품 정보가 담긴 DataFrame (없으면 빈 DataFrame)
        """
        if gt_shape_df is None or len(gt_shape_df) == 0:
            return pd.DataFrame()
        
        logger.info(f"{gt_shape_df['gt_id'].nunique()}개의 groundtruth에 대한 의약품 정보 조회...")
        
        # sample_images에서 이미 가져온 gt_shape 정보 사용
        # label_id=1은 이미 필터링되어 있음
        results_shape = gt_shape_df.copy()
        
        # gts_id가 None인 행 제거 (gt_shape가 없는 경우)
        results_shape = results_shape[results_shape["gts_id"].notna()].copy()
        
        if len(results_shape) == 0:
            return pd.DataFrame()
        
        # medicine_id가 None인 행 제거
        results_shape = results_shape[results_shape["medicine_id"].notna()].copy()
        results_shape["medicine_id"] = results_shape["medicine_id"].astype(int)

        # reference_image_url 기반으로 onesglobal_medicine과 medicine 분리
        onesglobal_medicine_ids = results_shape[
            results_shape["reference_image_url"].apply(lambda x: x is not None and "/onesglobal/" in str(x))
        ]["medicine_id"].unique().tolist()
        
        medicine_ids = results_shape[
            results_shape["reference_image_url"].apply(lambda x: x is not None and "/medicine/" in str(x))
        ]["medicine_id"].unique().tolist()
        
        results_onesglobal_medicine = None
        results_medicine = None
        
        # onesglobal_medicine 조회
        if len(onesglobal_medicine_ids) > 0:
            with self.get_db_connection(schema="poucheye").cursor() as cursor:
                query = f"""
                    SELECT
                        id AS medicine_id,
                        Idfy_print_front AS print_front,
                        Idfy_print_back AS print_back,
                        Idfy_drug_shape AS drug_shape,
                        Idfy_thick AS thick,
                        Idfy_leng_long AS length_long,
                        Idfy_leng_short AS length_short,
                        Idfy_color_class1 AS color_front,
                        Idfy_color_class2 AS color_back,
                        Di_medicine_bag_class AS class_name,
                        Di_ethical_type AS etc_otc_name,
                        Idfy_form_type AS form_code_name,
                        Di_item_name AS item_name,
                        Di_cp_name AS manufacturer_name,
                        Di_charact AS visual_description
                    FROM onesglobal_medicine om
                    WHERE id IN ({', '.join(map(str, onesglobal_medicine_ids))})
                    """
                
                cursor.execute(query)
                results_onesglobal_medicine = pd.DataFrame(cursor.fetchall())
        
        # medicine 조회
        if len(medicine_ids) > 0:
            with self.get_db_connection(schema="poucheye").cursor() as cursor:
                query = f"""
                    SELECT
                        id AS medicine_id,
                        print_front,
                        print_back,
                        drug_shape,
                        thick,
                        length_long,
                        length_short,
                        color_front,
                        color_back,
                        class_name,
                        etc_otc_name,
                        form_code_name,
                        item_name,
                        manufacturer_name, 
                        visual_description
                    FROM medicine m
                    WHERE id IN ({', '.join(map(str, medicine_ids))})
                    """
                
                cursor.execute(query)
                results_medicine = pd.DataFrame(cursor.fetchall())
        
        # 두 결과 합치기
        if results_onesglobal_medicine is not None and results_medicine is not None:
            results_medicine_combined = pd.concat([results_onesglobal_medicine, results_medicine], ignore_index=True)
        else:
            results_medicine_combined = results_onesglobal_medicine if results_onesglobal_medicine is not None else results_medicine
        
        if results_medicine_combined is None or len(results_medicine_combined) == 0:
            logger.warning("의약품 정보를 조회할 수 없습니다.")
            return pd.DataFrame()

        results = pd.merge(results_shape, results_medicine_combined, on="medicine_id", how="left")
            
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
        # 이미지 샘플링 (gt_shape 정보 포함)
        images_df = self.sample_images(n)
        if len(images_df) == 0:
            return []
        
        # 의약품 정보 조회 (sample_images에서 가져온 gt_shape 정보 재사용)
        medicine_df = self.get_medicine_info(images_df)
        
        # 데이터 조합 (이미지 ID별로 그룹화하여 처리)
        collected_data = []
        for image_id in images_df["id"].unique():
            # 해당 이미지의 첫 번째 행 가져오기 (이미지 정보는 모든 행에서 동일)
            img_row = images_df[images_df["id"] == image_id].iloc[0]
            
            # 이미지 다운로드
            local_path = self.download_image(img_row["url"], img_row["id"])
            if local_path is None:
                logger.warning(f"이미지 다운로드 실패: {img_row['id']}")
                continue
            
            # 해당 이미지의 gt_id 목록 (하나의 이미지에 여러 gt가 있을 수 있음)
            image_gt_ids = images_df[images_df["id"] == image_id]["gt_id"].unique().tolist()
            
            # 해당 이미지의 모든 gt에 대한 의약품 정보 수집
            medicine_info_list = []
            for gt_id in image_gt_ids:
                gt_medicine_info = medicine_df[medicine_df["gt_id"] == gt_id]
                if len(gt_medicine_info) > 0:
                    medicine_info_list.append(gt_medicine_info)
            
            # 모든 gt의 의약품 정보 병합
            if medicine_info_list:
                medicine_info = pd.concat(medicine_info_list, ignore_index=True)
                # 같은 의약품(em_id / item_name)이 여러 박스에 반복될 수 있으므로,
                # fine-tuning용 컨텍스트에서는 의약품 단위로만 사용하도록 중복 제거
                medicine_info = (
                    medicine_info
                    .drop_duplicates(subset=["medicine_id"])
                    .reset_index(drop=True)
                )
            else:
                medicine_info = pd.DataFrame()
            
            # 각 gt별로 데이터 항목 생성
            for gt_id in image_gt_ids:
                data_item = {
                    "image_id": int(img_row["id"]),
                    "gt_id": int(gt_id),
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


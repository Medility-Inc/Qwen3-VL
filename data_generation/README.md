# Qwen3-VL Fine-tuning 데이터셋 생성 파이프라인

의약품 이미지와 정보를 기반으로 GPT-5.1과 Qwen3-VL-8B-Thinking을 활용하여 상호 검수 방식으로 VQA 데이터셋을 생성하는 파이프라인입니다.

## 주요 기능

1. **데이터 수집**: DB에서 랜덤 이미지 샘플링 및 의약품 정보 수집
2. **질문 생성**: GPT-5.1과 Qwen3-VL-8B-Thinking을 사용하여 다양한 VQA 질문 생성
3. **상호 검수**: 두 모델이 생성한 질문을 상호 검수하여 품질 보장
4. **중복 제거**: Qwen3-VL-8B-Thinking을 사용하여 중복 질문 제거
5. **답변 생성**: 검수 통과한 질문에 대해 답변 생성
6. **데이터셋 변환**: Qwen3-VL 학습 형식으로 변환

## 설치

### uv 사용 (권장)

```bash
cd data_generation

# uv가 설치되어 있지 않다면 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# Python 버전 설정 및 가상 환경 생성
uv python install 3.11

# 의존성 설치
uv sync

# 개발 의존성 포함 설치
uv sync --dev
```

### pip 사용 (대안)

```bash
cd data_generation
pip install -r requirements.txt
```

## 설정

1. `.env` 파일 생성 (`data_generation` 디렉토리에)
```bash
cd data_generation
cp .env.example .env  # 또는 직접 생성
```

`.env` 파일 내용:
```bash
# 데이터베이스 설정
DB_HOST=your_db_host
DB_USERNAME=your_db_username
DB_PASSWORD=your_db_password

# OpenAI API 설정
OPENAI_API_KEY=your_openai_api_key
```

2. `config.yaml` 파일 수정 (필요시)

## 사용 방법

### uv 사용

```bash
# 기본 사용
uv run generate_dataset.py

# 옵션 지정
uv run generate_dataset.py \
    --config config.yaml \
    --num-images 100 \
    --checkpoint-interval 10
```

### 일반 Python 사용

```bash
# 기본 사용
python generate_dataset.py

# 옵션 지정
python generate_dataset.py \
    --config config.yaml \
    --num-images 100 \
    --checkpoint-interval 10
```

## 워크플로우

1. **데이터 수집**: DB에서 랜덤 이미지 샘플링 (중복 체크)
2. **이미지 다운로드**: 이미지를 로컬에 저장
3. **질문 생성**: GPT-5.1과 Qwen3-VL-8B-Thinking이 각각 질문 생성
4. **상호 검수**: 
   - GPT-5.1이 생성한 질문 → Qwen3-VL-8B-Thinking 검수
   - Qwen3-VL-8B-Thinking이 생성한 질문 → GPT-5.1 검수
5. **중복 제거**: Qwen3-VL-8B-Thinking이 중복 질문 제거
6. **답변 생성**: 검수 통과한 질문에 대해 답변 생성
7. **데이터셋 변환**: Qwen3-VL 학습 형식으로 변환 및 저장
8. **이미지 ID 기록**: 사용된 이미지 ID를 `used_images.json`에 기록

## 출력 파일

- `dataset.json`: 최종 생성된 데이터셋 (Qwen3-VL 형식)
- `used_images.json`: 사용된 이미지 ID 기록 (중복 방지)
- `checkpoints/checkpoint_*.json`: 중간 체크포인트 파일
- `generation.log`: 생성 로그

## 데이터 형식

생성된 데이터셋은 다음 형식을 따릅니다:

```json
{
  "image": "images/001.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n약포에 파란색 약이 있나요?"
    },
    {
      "from": "gpt",
      "value": "예, 파란색 약이 있습니다."
    }
  ],
  "difficulty": "easy"
}
```

## 테스트

### uv 사용

```bash
uv run pytest tests/
```

### 일반 Python 사용

```bash
pytest tests/
```

## 주의사항

- Qwen3-VL-8B-Thinking API 서버가 `http://localhost:1119`에서 실행 중이어야 합니다.
- OpenAI API 키가 환경변수에 설정되어 있어야 합니다.
- DB 연결 정보가 올바르게 설정되어 있어야 합니다.


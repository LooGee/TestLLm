---
name: 노트북 전용 스크립트 개발 계획
overview: 노트북(Mac)에서 실행할 스크립트 개발 계획 - PostgreSQL DB에서 데이터를 조회하고 프롬프트를 생성한 후 PC의 API 서버로 전송하여 모델 응답을 받는 클라이언트 시스템 구축
todos:
  - id: create-config
    content: config.py 파일 생성 - 환경 변수 로드, DB 연결 정보, API URL 관리
    status: pending
  - id: setup-env
    content: .env 파일 생성 (노트북용) - DB 설정, 네트워크 설정, API 클라이언트 설정
    status: pending
  - id: create-db-connector
    content: db_connector.py 파일 생성 - PostgreSQL 연결, 쿼리 실행, 연결 테스트 함수 구현
    status: pending
    dependencies:
      - create-config
  - id: create-prompt-builder
    content: prompt_builder.py 파일 생성 - DB 데이터를 프롬프트로 변환, 템플릿 관리, 데이터 포맷팅
    status: pending
  - id: create-api-client
    content: model_api_client.py 파일 생성 - PC API 서버와 통신, 프롬프트 전송, 응답 수신, 오류 처리
    status: pending
    dependencies:
      - create-config
  - id: create-pipeline
    content: db_to_model_pipeline.py 파일 생성 - 전체 파이프라인 오케스트레이션, DB 조회 → 프롬프트 생성 → API 전송
    status: pending
    dependencies:
      - create-db-connector
      - create-prompt-builder
      - create-api-client
  - id: setup-docker-postgres
    content: Docker PostgreSQL 설정 - docker-compose.yml 생성 또는 Docker 명령어로 컨테이너 실행
    status: pending
  - id: test-db-connection
    content: DB 연결 테스트 - db_connector의 test_connection() 함수로 연결 확인
    status: pending
    dependencies:
      - create-db-connector
      - setup-docker-postgres
  - id: test-api-connection
    content: PC API 서버 연결 테스트 - model_api_client의 check_server_health() 함수로 서버 상태 확인
    status: pending
    dependencies:
      - create-api-client
  - id: test-full-pipeline
    content: 전체 파이프라인 테스트 - 간단한 쿼리로 DB 조회부터 모델 응답까지 전체 프로세스 테스트
    status: pending
    dependencies:
      - create-pipeline
      - test-db-connection
      - test-api-connection
  - id: implement-logging
    content: 로깅 설정 구현 - 파일 및 콘솔 출력, DB 연결, 쿼리 실행, API 요청 로깅
    status: pending
  - id: implement-error-handling
    content: 오류 처리 구현 - DB 연결 실패, 쿼리 오류, API 서버 연결 실패에 대한 예외 처리
    status: pending
---

# 노트북 전용 스크립트 개발 계획서

## 1. 개요

### 1.1 목적

노트북(Mac)에서 PostgreSQL DB를 조회하고, 조회한 데이터로부터 프롬프트를 생성한 후, PC(Windows)의 API 서버로 전송하여 Phi-4 모델의 응답을 받는 클라이언트 시스템을 구축합니다.

### 1.2 역할

- **DB 조회**: Docker 컨테이너의 PostgreSQL에서 데이터 조회
- **프롬프트 생성**: DB 데이터를 기반으로 프롬프트 생성
- **API 클라이언트**: PC의 FastAPI 서버에 프롬프트 전송 및 응답 수신
- **파이프라인 오케스트레이션**: 전체 프로세스 통합 관리

### 1.3 네트워크 환경

- **노트북**: Mac, WiFi 연결, 로컬 IP (예: 192.168.0.YYY)
- **PC**: Windows, 유선 연결, 로컬 IP (예: 192.168.0.XXX)
- **DB**: Docker 컨테이너 내 PostgreSQL (localhost:5432)
- **통신**: 노트북 → PC (HTTP API)

## 2. 개발할 파일

### 2.1 `config.py` (공통 파일 생성)

#### 2.1.1 기능

- 환경 변수 로드 및 관리
- DB 연결 정보 관리
- PC API 서버 URL 관리
- 공통 설정 제공

#### 2.1.2 구현 구조

```python
"""
공통 설정 관리 모듈
노트북과 PC 모두에서 사용
"""
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    """공통 설정 클래스"""
    
    # API 서버 설정 (PC)
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    # 모델 설정
    MODEL_TYPE = os.getenv("MODEL_TYPE", "phi4-quantized")
    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
    
    # DB 설정 (노트북)
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", 5432))
    DB_NAME = os.getenv("DB_NAME", "your_database")
    DB_USER = os.getenv("DB_USER", "your_user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")
    
    # 네트워크 설정
    PC_IP = os.getenv("PC_IP", "192.168.0.XXX")
    LAPTOP_IP = os.getenv("LAPTOP_IP", "192.168.0.YYY")
    
    # 타임아웃 설정
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", 300))
    
    @classmethod
    def get_api_url(cls):
        """PC API 서버의 전체 URL 반환"""
        return f"http://{cls.PC_IP}:{cls.API_PORT}"
    
    @classmethod
    def get_db_connection_string(cls):
        """DB 연결 문자열 반환"""
        return f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
```

### 2.2 `db_connector.py` (신규 생성)

#### 2.2.1 기능

- PostgreSQL DB 연결 관리
- SQL 쿼리 실행
- 결과 데이터 반환
- 연결 풀 관리 (선택사항)

#### 2.2.2 주요 함수

**`connect_db()`**

- 목적: PostgreSQL DB 연결
- 입력: 없음 (Config에서 설정 읽기)
- 출력: connection 객체
- 특징: 연결 실패 시 재시도 로직

**`execute_query(query, params=None)`**

- 목적: SQL 쿼리 실행
- 입력:
                - `query`: SQL 쿼리 문자열
                - `params`: 쿼리 파라미터 (dict, 선택사항)
- 출력: 쿼리 결과 (list of dict)
- 특징: 안전한 파라미터 바인딩

**`test_connection()`**

- 목적: DB 연결 테스트
- 입력: 없음
- 출력: bool (연결 성공 여부)

#### 2.2.3 구현 구조

```python
"""
PostgreSQL DB 연결 및 쿼리 모듈
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
import logging
from config import Config

logger = logging.getLogger(__name__)

# 연결 풀 (선택사항)
_connection_pool = None

def connect_db():
    """PostgreSQL DB 연결"""
    # 구현 내용

def execute_query(query, params=None):
    """SQL 쿼리 실행"""
    # 구현 내용

def test_connection():
    """DB 연결 테스트"""
    # 구현 내용
```

### 2.3 `prompt_builder.py` (신규 생성)

#### 2.3.1 기능

- DB 데이터를 프롬프트 형식으로 변환
- 프롬프트 템플릿 관리
- 데이터 포맷팅

#### 2.3.2 주요 함수

**`build_prompt_from_data(db_data, template=None, additional_context=None)`**

- 목적: DB 데이터로부터 프롬프트 생성
- 입력:
                - `db_data`: DB 쿼리 결과 (dict)
                - `template`: 프롬프트 템플릿 (str, 선택사항)
                - `additional_context`: 추가 컨텍스트 (str, 선택사항)
- 출력: 완성된 프롬프트 (str)

**`format_db_results(results)`**

- 목적: DB 결과를 읽기 쉬운 형식으로 변환
- 입력: 쿼리 결과 (list of dict)
- 출력: 포맷된 문자열

#### 2.3.3 구현 구조

```python
"""
프롬프트 빌더 모듈
DB 데이터로부터 프롬프트 생성
"""
import json
from typing import List, Dict, Optional

def build_prompt_from_data(db_data, template=None, additional_context=None):
    """DB 데이터로부터 프롬프트 생성"""
    # 구현 내용

def format_db_results(results):
    """DB 결과 포맷팅"""
    # 구현 내용
```

### 2.4 `model_api_client.py` (신규 생성)

#### 2.4.1 기능

- PC의 API 서버와 통신
- 프롬프트 전송 및 응답 수신
- 오류 처리 및 재시도 로직

#### 2.4.2 주요 함수

**`check_server_health()`**

- 목적: PC API 서버 상태 확인
- 입력: 없음
- 출력: bool (서버 정상 여부)

**`send_prompt(prompt, system_prompt=None, max_new_tokens=None, temperature=None)`**

- 목적: 프롬프트를 PC API 서버로 전송
- 입력:
                - `prompt`: 사용자 프롬프트 (str)
                - `system_prompt`: 시스템 프롬프트 (str, 선택사항)
                - `max_new_tokens`: 최대 생성 토큰 수 (int, 선택사항)
                - `temperature`: 생성 온도 (float, 선택사항)
- 출력: (response, stats) 튜플
                - `response`: 생성된 답변 (str)
                - `stats`: 통계 정보 (dict)

#### 2.4.3 구현 구조

```python
"""
PC API 서버 클라이언트 모듈
"""
import requests
import logging
from config import Config

logger = logging.getLogger(__name__)

def check_server_health():
    """PC API 서버 상태 확인"""
    # 구현 내용

def send_prompt(prompt, system_prompt=None, max_new_tokens=None, temperature=None):
    """프롬프트 전송 및 응답 수신"""
    # 구현 내용
```

### 2.5 `db_to_model_pipeline.py` (신규 생성)

#### 2.5.1 기능

- 전체 파이프라인 오케스트레이션
- DB 조회 → 프롬프트 생성 → API 전송 → 결과 처리
- 에러 핸들링 및 로깅

#### 2.5.2 주요 함수

**`run_pipeline(query, template=None, additional_context=None)`**

- 목적: 전체 파이프라인 실행
- 입력:
                - `query`: DB 쿼리 (str)
                - `template`: 프롬프트 템플릿 (str, 선택사항)
                - `additional_context`: 추가 컨텍스트 (str, 선택사항)
- 출력: (response, stats, db_data) 튜플

**`main()`**

- 목적: 명령줄 인터페이스
- 입력: 명령줄 인자
- 출력: 없음 (결과 출력)

#### 2.5.3 구현 구조

```python
"""
DB to Model 파이프라인
DB 조회 → 프롬프트 생성 → 모델 실행 → 결과 반환
"""
import argparse
import logging
from db_connector import execute_query, test_connection
from prompt_builder import build_prompt_from_data
from model_api_client import send_prompt, check_server_health

logger = logging.getLogger(__name__)

def run_pipeline(query, template=None, additional_context=None):
    """전체 파이프라인 실행"""
    # 구현 내용

if __name__ == "__main__":
    # 명령줄 인터페이스
    # 구현 내용
```

## 3. 의존성

### 3.1 추가 패키지

**`requirements.txt`에 추가** (노트북 전용):

```
psycopg2-binary>=2.9.0  # PostgreSQL 연결
python-dotenv>=1.0.0    # 환경 변수 관리 (공통)
requests>=2.31.0        # HTTP 요청 (공통)
```

### 3.2 공통 패키지 활용

- `config.py`: 공통 설정 관리
- `공용_작업_가이드.md`: API 프로토콜 및 데이터 형식 참조

## 4. 설정 파일

### 4.1 `.env` 파일 (노트북용)

```bash
# 모델 설정
MODEL_TYPE=phi4-quantized
MAX_NEW_TOKENS=512
TEMPERATURE=0.7

# DB 설정
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database
DB_USER=your_user
DB_PASSWORD=your_password

# 네트워크 설정
PC_IP=192.168.0.XXX  # PC의 실제 IP 주소
LAPTOP_IP=192.168.0.YYY  # 노트북의 실제 IP 주소

# API 클라이언트 설정
API_TIMEOUT=300  # 초 단위
```

## 5. Docker PostgreSQL 설정

### 5.1 Docker Compose 파일 (선택사항)

**`docker-compose.yml`**:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    container_name: phi4_postgres
    environment:
      POSTGRES_DB: your_database
      POSTGRES_USER: your_user
      POSTGRES_PASSWORD: your_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
```

### 5.2 Docker 명령어

```bash
# Docker 컨테이너 실행
docker run -d \
  --name phi4_postgres \
  -e POSTGRES_DB=your_database \
  -e POSTGRES_USER=your_user \
  -e POSTGRES_PASSWORD=your_password \
  -p 5432:5432 \
  postgres:15

# 또는 docker-compose 사용
docker-compose up -d
```

## 6. 실행 방법

### 6.1 기본 실행

```bash
# 파이프라인 실행 (쿼리 직접 입력)
python db_to_model_pipeline.py --query "SELECT * FROM table LIMIT 10"

# 프롬프트 템플릿 사용
python db_to_model_pipeline.py --query "SELECT * FROM table" --template "다음 데이터를 분석해주세요: {data}"
```

### 6.2 개별 모듈 테스트

```bash
# DB 연결 테스트
python -c "from db_connector import test_connection; test_connection()"

# API 서버 상태 확인
python -c "from model_api_client import check_server_health; check_server_health()"
```

## 7. 테스트

### 7.1 DB 연결 테스트

```bash
# db_connector.py 테스트
python -c "from db_connector import test_connection; print('DB 연결:', test_connection())"
```

### 7.2 API 서버 연결 테스트

```bash
# model_api_client.py 테스트
python -c "from model_api_client import check_server_health; print('서버 상태:', check_server_health())"
```

### 7.3 전체 파이프라인 테스트

```bash
# 간단한 쿼리로 테스트
python db_to_model_pipeline.py --query "SELECT 1 as test"
```

## 8. 로깅

### 8.1 로그 설정

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('laptop_client.log'),
        logging.StreamHandler()
    ]
)
```

### 8.2 로그 내용

- DB 연결 시도 및 결과
- 쿼리 실행 및 결과
- 프롬프트 생성
- API 요청 전송 및 응답
- 오류 발생 시 상세 정보

## 9. 오류 처리

### 9.1 주요 오류 상황

**DB 연결 실패**:

- 연결 정보 확인
- Docker 컨테이너 실행 상태 확인
- 재시도 로직 적용

**API 서버 연결 실패**:

- PC IP 주소 확인
- PC에서 서버 실행 상태 확인
- 네트워크 연결 확인

**쿼리 실행 오류**:

- SQL 문법 확인
- 테이블/컬럼 존재 확인
- 권한 확인

### 9.2 오류 처리 전략

- 각 단계별 try-except 블록
- 상세한 오류 메시지 제공
- 재시도 로직 (선택사항)
- 로깅을 통한 오류 추적

## 10. 보안 고려사항

### 10.1 DB 비밀번호 관리

- `.env` 파일을 Git에 커밋하지 않기
- 환경 변수로 비밀번호 관리
- `.gitignore`에 `.env` 추가

### 10.2 네트워크 보안

- 로컬 네트워크에서만 통신
- 외부 IP 접근 시 추가 보안 조치 필요

## 11. 배포 체크리스트

- [ ] `config.py` 생성
- [ ] `.env` 파일 생성 (노트북용)
- [ ] `db_connector.py` 생성 및 테스트
- [ ] `prompt_builder.py` 생성 및 테스트
- [ ] `model_api_client.py` 생성 및 테스트
- [ ] `db_to_model_pipeline.py` 생성 및 테스트
- [ ] Docker PostgreSQL 설정
- [ ] DB 연결 테스트 완료
- [ ] PC API 서버 연결 테스트 완료
- [ ] 전체 파이프라인 테스트 완료
- [ ] 로깅 설정 확인
- [ ] 오류 처리 확인
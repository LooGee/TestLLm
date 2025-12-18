"""
Phi-4 Model API Server
FastAPI 기반 HTTP API 서버로 노트북으로부터 프롬프트를 수신하고 모델로 처리하여 답변을 반환합니다.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from phi4_server import load_quantized_model, generate_response
import uvicorn
import logging
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(title="Phi-4 Model API Server", version="1.0.0")

# CORS 설정 (노트북에서 접근 허용)
cors_origins = os.getenv("CORS_ORIGINS", "http://192.168.0.0/24").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
_model = None
_tokenizer = None
_model_loaded = False

def load_model():
    """Phi-4 모델 로드 (서버 시작 시 한 번만)
    
    Returns:
        tuple: (model, tokenizer) 튜플
    """
    global _model, _tokenizer, _model_loaded
    
    if _model is not None and _tokenizer is not None:
        logger.info("모델이 이미 메모리에 로드되어 있습니다.")
        return _model, _tokenizer
    
    try:
        logger.info("모델 로딩 시작...")
        model_dir = os.getenv("MODEL_DIR", None)
        if model_dir == "":
            model_dir = None
        
        _model, _tokenizer = load_quantized_model(model_dir)
        _model_loaded = True
        logger.info("모델 로딩 완료")
        return _model, _tokenizer
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}", exc_info=True)
        _model_loaded = False
        raise

# Pydantic 모델
class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: str = None
    max_new_tokens: int = 512
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    response: str
    stats: dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 사전 로드"""
    logger.info("서버 시작 중...")
    try:
        load_model()
        logger.info("서버 시작 완료 - 모델 준비됨")
    except Exception as e:
        logger.error(f"서버 시작 중 모델 로드 실패: {e}")
        logger.warning("서버는 계속 실행되지만 모델이 로드되지 않았습니다.")

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """프롬프트 수신 및 모델 실행
    
    Args:
        request: GenerateRequest 객체 (prompt, system_prompt, max_new_tokens, temperature)
    
    Returns:
        GenerateResponse: 생성된 답변과 통계 정보
    """
    global _model, _tokenizer, _model_loaded
    
    # 프롬프트 검증
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="프롬프트가 필요합니다")
    
    # 모델 로드 확인
    if not _model_loaded or _model is None or _tokenizer is None:
        try:
            load_model()
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise HTTPException(status_code=503, detail=f"모델을 로드할 수 없습니다: {str(e)}")
    
    try:
        logger.info(f"프롬프트 수신: {request.prompt[:100]}...")
        
        # 프롬프트 구성 (시스템 프롬프트가 있으면 추가)
        prompt = request.prompt
        if request.system_prompt:
            prompt = f"{request.system_prompt}\n\n{request.prompt}"
        
        # 모델 실행
        response, stats = generate_response(
            _model,
            _tokenizer,
            prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature
        )
        
        logger.info(f"응답 생성 완료 - 생성 시간: {stats['generation_time']:.2f}초, 토큰 수: {stats['generated_tokens']}")
        
        return GenerateResponse(
            response=response,
            stats=stats
        )
    except Exception as e:
        logger.error(f"모델 실행 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"모델 실행 중 오류 발생: {str(e)}")

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """서버 상태 확인
    
    Returns:
        HealthResponse: 서버 상태 및 모델 로드 상태
    """
    return HealthResponse(
        status="healthy",
        model_loaded=_model_loaded
    )

if __name__ == '__main__':
    # 환경 변수에서 설정 읽기
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    logger.info(f"API 서버 시작: {host}:{port}")
    logger.info(f"자동 리로드: {reload}")
    
    uvicorn.run(app, host=host, port=port, reload=reload)


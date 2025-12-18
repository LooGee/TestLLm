# Phi-4 4BIT 양자화 로컬 실행 프로젝트

Microsoft의 Phi-4 모델을 4BIT 양자화하여 로컬에서 실행하고, Ollama 환경에서도 구동할 수 있도록 구성한 프로젝트입니다.

## 📋 목차

- [요구사항](#요구사항)
- [설치](#설치)
- [사용 방법](#사용-방법)
  - [1. 4BIT 양자화 실행 (bitsandbytes)](#1-4bit-양자화-실행-bitsandbytes)
  - [2. Ollama 실행](#2-ollama-실행)
- [프롬프트 입력 방법](#프롬프트-입력-방법)

## 🔧 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (bitsandbytes 사용 시)
- 최소 16GB RAM 권장
- 약 30GB 디스크 공간 (모델 다운로드)

## 📦 설치

### 설치 방법

1. 저장소 클론 또는 파일 다운로드

2. 가상환경 생성 및 활성화:
```bash
# 가상환경 생성
python -m venv .venv

# Windows - 가상환경 활성화
.venv\Scripts\activate

# Linux/Mac - 가상환경 활성화
source .venv/bin/activate
```

3. 필요한 패키지 설치:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 가상환경 비활성화
```bash
deactivate
```

### 가상환경 확인
가상환경이 활성화되면 명령 프롬프트 앞에 `(.venv)` 또는 `(venv)`가 표시됩니다.

## 🚀 사용 방법

### 1. 4BIT 양자화 실행 (bitsandbytes)

#### 💾 로컬 저장 모델 사용 (가장 빠름) ⭐
양자화된 모델을 로컬에 저장하여 매번 양자화 과정을 건너뛰고 빠르게 로드합니다.

**1단계: 모델 저장 (한 번만 실행)**
```bash
python save_quantized_model.py
```

또는 저장 위치 지정:
```bash
python save_quantized_model.py ./my_models/phi4
```

**2단계: 저장된 모델 사용**
```bash
# 서버 모드 (권장)
python phi4_server.py "프롬프트"

# 대화형 모드
python prompt_input.py

# 저장 위치 지정
python phi4_server.py "프롬프트" --model-dir ./my_models/phi4
```

> **장점:** 
> - 양자화 과정을 한 번만 수행
> - 이후 로딩이 매우 빠름
> - 인터넷 연결 불필요 (저장 후)

#### ⚡ 서버 모드 - 빠른 응답
모델을 한 번 로드한 후 메모리에 유지하여 빠른 응답을 제공합니다.

**대화형 모드:**
```bash
python phi4_server.py
```

**단일 프롬프트 실행:**
```bash
python phi4_server.py "당신의 프롬프트를 여기에 입력하세요"
```

> **장점:** 모델을 한 번만 로드하므로 두 번째 요청부터 매우 빠릅니다.

#### 일반 모드
**대화형 모드:**
```bash
python prompt_input.py
```

**단일 프롬프트 실행:**
```bash
python prompt_input.py "당신의 프롬프트를 여기에 입력하세요"
```

**직접 실행:**
```bash
python phi4_quantized.py "당신의 프롬프트를 여기에 입력하세요"
```

### 2. Ollama 실행

#### Ollama 설치

**Windows:**
1. [Ollama 공식 사이트](https://ollama.com/)에서 Windows 설치 파일 다운로드
2. 설치 파일 실행 및 설치 완료
3. 설치 후 Ollama 앱이 자동으로 실행됩니다
4. 터미널에서 `ollama` 명령어가 작동하는지 확인:
   ```bash
   ollama --version
   ```
   - 작동하지 않으면: Ollama 설치 경로를 PATH 환경 변수에 추가하거나 재시작

**Linux/Mac:**
```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Mac
# Homebrew 사용
brew install ollama
```

**설치 확인:**
```bash
ollama --version
```

**서버 시작:**
- Windows: Ollama 앱이 자동으로 실행됩니다
- Linux/Mac: `ollama serve` 명령어로 서버 시작

#### Python 스크립트로 Ollama API 사용 (권장) ⭐

다른 스크립트들과 동일한 기능을 제공합니다:
- 멀티라인 입력 지원
- 빈 줄 처리
- 스크립트 내부 프롬프트 실행
- 생성 통계 표시 (시간, 토큰 수 등)

**1단계: Ollama 모델 다운로드**
```bash
# Ollama에서 phi4 모델 다운로드
ollama pull phi4
```

**2단계: Python 스크립트 실행**

**대화형 모드:**
```bash
python ollama_api_example.py --interactive
# 또는
python ollama_api_example.py -i
```

**단일 프롬프트 실행 (스크립트 내부 프롬프트 사용):**
```bash
# ollama_api_example.py 파일의 SCRIPT_PROMPT 변수를 수정한 후
python ollama_api_example.py
```

**모델 지정:**
```bash
# phi4 모델 사용 (기본값)
python ollama_api_example.py --interactive --model phi4
```

**Ollama 서버 URL 지정:**
```bash
python ollama_api_example.py --base-url http://localhost:11434
```

#### Ollama CLI 직접 사용

**Ollama 모델 다운로드:**
```bash
ollama pull phi4
```

**대화형 모드:**
```bash
ollama run phi4
```

**단일 프롬프트:**
```bash
ollama run phi4 "당신의 프롬프트를 여기에 입력하세요"
```

#### Modelfile 사용 (고급)

**중요:** Modelfile을 사용하기 전에 먼저 Ollama에서 phi4 모델을 다운로드해야 합니다.

```bash
# 1단계: Ollama에서 phi4 모델 다운로드
ollama pull phi4

# 2단계: Modelfile을 사용하여 커스텀 설정으로 모델 생성
ollama create phi4-quantized -f Modelfile

# 3단계: 생성된 모델 실행
ollama run phi4-quantized
```

> **참고:** 
> - Ollama는 Hugging Face 모델을 직접 사용할 수 없습니다. 먼저 `ollama pull phi4`로 Ollama가 지원하는 phi4 모델을 다운로드해야 합니다.
> - Modelfile의 `FROM phi4`는 Ollama에 다운로드된 phi4 모델을 참조합니다.

## 💬 프롬프트 입력 방법

### 대화형 모드
1. 스크립트를 실행하면 대화형 인터페이스가 시작됩니다.
2. 프롬프트를 입력하고 Enter를 누르면 응답이 생성됩니다.
3. 종료하려면 `quit`, `exit`, 또는 `q`를 입력하세요.

### 단일 프롬프트 모드
명령줄에서 프롬프트를 직접 전달할 수 있습니다:
```bash
python prompt_input.py "Python으로 피보나치 수열을 계산하는 함수를 작성해주세요"
```

## 📝 주요 파일 설명

- `phi4_server.py`: **서버 모드** - 모델을 메모리에 유지하여 빠른 응답 제공 (권장) ⚡
- `phi4_quantized.py`: 4BIT 양자화된 phi-4 모델 실행 스크립트
- `prompt_input.py`: 프롬프트 입력을 받는 메인 대화형 스크립트
- `ollama_api_example.py`: **Ollama API 실행** - phi4 모델을 Ollama로 실행 ⭐
- `device_utils.py`: GPU/CPU 감지 및 디바이스 정보 유틸리티
- `Modelfile`: Ollama용 phi4-quantized 모델 설정 파일
- `requirements.txt`: 필요한 Python 패키지 목록

## ⚙️ 설정 조정

### 생성 파라미터 조정

`phi4_quantized.py`에서 다음 파라미터를 조정할 수 있습니다:

- `max_new_tokens` / `max_tokens`: 생성할 최대 토큰 수 (기본값: 512)
- `temperature`: 생성의 다양성 조절 (0.0-1.0, 기본값: 0.7)
- `top_p`: nucleus sampling 파라미터 (기본값: 0.9)

### 양자화 설정

`phi4_quantized.py`의 `BitsAndBytesConfig`에서 양자화 방식을 조정할 수 있습니다:
- `load_in_4bit`: 4BIT 양자화 활성화
- `bnb_4bit_quant_type`: 양자화 타입 ("nf4" 또는 "fp4")
- `bnb_4bit_use_double_quant`: 이중 양자화 사용 여부

## 🔍 문제 해결

### CUDA 오류 / CPU 전용 PyTorch 설치됨
**증상**: `PyTorch 버전: X.X.X+cpu` 또는 `CUDA 사용 가능: False`

**원인**: 가상환경에 CPU 전용 PyTorch가 설치되어 있습니다.

**해결 방법**:

**해결 방법**:
   ```bash
   # 기존 PyTorch 제거
   pip uninstall torch torchvision torchaudio -y
   
   # CUDA 11.8 지원 PyTorch 설치
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # 또는 CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **확인**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
   ```

**참고**: 
- NVIDIA GPU와 CUDA 드라이버가 설치되어 있어야 합니다.
- `nvidia-smi` 명령어로 GPU 상태를 확인하세요.
- CUDA 버전에 맞는 PyTorch를 설치해야 합니다.

### 메모리 부족
- `max_new_tokens` 값을 줄여보세요.
- 배치 크기를 조정하세요.

### 모델 다운로드 실패
- 인터넷 연결을 확인하세요.
- Hugging Face 토큰이 필요한 경우 설정하세요:
```bash
huggingface-cli login
```

### 모델 로딩이 느린 경우
- **서버 모드 사용**: `phi4_server.py`를 사용하면 모델을 한 번만 로드하고 메모리에 유지합니다.
- **캐시 확인**: Hugging Face 캐시 디렉토리가 올바르게 설정되어 있는지 확인하세요.
- **첫 실행**: 첫 실행 시 모델을 다운로드하므로 시간이 걸립니다. 이후에는 캐시에서 로드됩니다.
- **메모리**: 충분한 RAM과 VRAM이 있는지 확인하세요.

### 로컬 저장 모델 사용법
모델은 `./models/phi4-quantized` 디렉토리에 자동으로 저장됩니다.

**저장된 모델 사용**:
```bash
# 자동 감지 (기본 경로에 저장된 경우)
python phi4_server.py "프롬프트"

# 저장 위치 지정
python phi4_server.py "프롬프트" --model-dir ./models/phi4-quantized
```

### PowerShell PSReadLine 오류 (Windows)
**증상**: `System.ArgumentOutOfRangeException: 값은 0보다 크거나 같아야 하며...` 오류가 발생합니다.

**원인**: PowerShell의 PSReadLine 모듈 버그로, 긴 명령어 히스토리를 탐색할 때 콘솔 버퍼 범위를 벗어나는 경우 발생합니다.

**해결 방법**:

1. **PSReadLine 모듈 업데이트** (권장):
   ```powershell
   # PowerShell을 관리자 권한으로 실행 후
   Install-Module PSReadLine -Force -SkipPublisherCheck
   ```

2. **PowerShell 프로필 수정**:
   ```powershell
   # 프로필 파일 열기
   notepad $PROFILE
   
   # 다음 내용 추가
   Set-PSReadLineOption -PredictionSource None
   Set-PSReadLineOption -HistoryNoDuplicates
   ```

3. **임시 해결책 - CMD 사용**:
   - PowerShell 대신 명령 프롬프트(CMD)를 사용하세요.

4. **PowerShell 재시작**:
   - PowerShell 창을 닫고 다시 열어보세요.

**참고**: 이 오류는 프로젝트 코드와 무관하며, PowerShell 자체의 문제입니다. 프로젝트 실행에는 영향을 주지 않습니다.

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. Phi-4 모델도 MIT 라이선스로 배포됩니다.

## 🙏 참고 자료

- [Phi-4 모델 카드](https://huggingface.co/microsoft/phi-4)
- [Ollama 문서](https://ollama.com/)


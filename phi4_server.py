"""
Phi-4 모델 서버 모드
모델을 한 번 로드한 후 계속 메모리에 유지하여 빠른 응답을 제공합니다.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys
import os
import time
from datetime import datetime
from device_utils import check_and_setup_device, get_device_info

# 전역 변수로 모델과 토크나이저 저장
_model = None
_tokenizer = None

# 스크립트 내부에 작성할 프롬프트
# 이 변수를 수정하여 실행할 프롬프트를 지정하세요
SCRIPT_PROMPT = """
당신의 프롬프트를 여기에 작성하세요.
예: "Python으로 피보나치 수열을 계산하는 함수를 작성해주세요"
"""

def load_quantized_model(model_dir=None):
    """4BIT 양자화된 phi-4 모델을 로드합니다.
    
    Args:
        model_dir: 로컬에 저장된 모델 디렉토리 경로 (None이면 기본 경로 또는 Hugging Face에서 로드)
    """
    global _model, _tokenizer
    
    if _model is not None and _tokenizer is not None:
        print("모델이 이미 메모리에 로드되어 있습니다.")
        return _model, _tokenizer
    
    # 기본 로컬 모델 경로
    default_local_dir = "./models/phi4-quantized"
    
    # 모델 디렉토리 결정
    if model_dir is None:
        # 명령줄 인자 확인
        import sys
        if len(sys.argv) > 1 and "--model-dir" in sys.argv:
            idx = sys.argv.index("--model-dir")
            if idx + 1 < len(sys.argv):
                model_dir = sys.argv[idx + 1]
        # 기본 로컬 경로 확인
        elif os.path.exists(default_local_dir) and os.path.isdir(default_local_dir):
            model_dir = default_local_dir
    
    model_name = "microsoft/phi-4"
    
    # GPU/CPU 확인 및 설정
    device, device_name, is_gpu = check_and_setup_device()
    
    # 4BIT 양자화 설정
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    start_time = time.time()
    
    # 로컬 모델이 있으면 사용
    if model_dir and os.path.exists(model_dir):
        print(f"로컬 저장된 모델 로딩 중: {model_dir}")
        print("(양자화된 모델을 빠르게 로드합니다)")
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True
        )
        
        # pad_token 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 양자화된 모델 로드
        # GPU가 사용 가능하면 명시적으로 cuda 사용
        if torch.cuda.is_available():
            print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
            print("GPU에 모델을 로드합니다...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"로컬 모델 로딩 완료! (소요 시간: {elapsed_time:.2f}초)")
        
        # 모델 디바이스 정보 확인
        device_info = get_device_info(model)
        print(f"\n모델 실행 디바이스: {device_info['device_name']} ({device_info['device']})")
        if device_info['is_gpu']:
            print("[OK] GPU 모드로 실행 중")
        else:
            print("[WARNING] CPU 모드로 실행 중 (GPU 사용 권장)")
        
        # 전역 변수에 저장
        _model = model
        _tokenizer = tokenizer
        
        return model, tokenizer
    
    # 로컬 모델이 없으면 Hugging Face에서 로드
    print(f"모델 로딩 중: {model_name}")
    print("4BIT 양자화 설정 적용 중...")
    print("(처음 로딩 시 시간이 걸릴 수 있습니다)")
    print("(로컬 저장 모델을 사용하려면: python save_quantized_model.py)")
    
    # Hugging Face 캐시 디렉토리 설정 (선택사항)
    cache_dir = os.getenv("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    # pad_token 설정 (없는 경우 eos_token 사용)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 양자화된 모델 로드 (최적화 옵션 추가)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir
    )
    
    elapsed_time = time.time() - start_time
    print(f"모델 로딩 완료! (소요 시간: {elapsed_time:.2f}초)")
    
    # 모델 디바이스 정보 확인
    device_info = get_device_info(model)
    print(f"\n모델 실행 디바이스: {device_info['device_name']} ({device_info['device']})")
    if device_info['is_gpu']:
        print("✅ GPU 모드로 실행 중")
    else:
        print("⚠️  CPU 모드로 실행 중 (GPU 사용 권장)")
    
    # 전역 변수에 저장
    _model = model
    _tokenizer = tokenizer
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7):
    """프롬프트에 대한 응답을 생성합니다.
    
    Returns:
        tuple: (response, stats_dict)
            - response: 생성된 응답 텍스트
            - stats_dict: 통계 정보 (생성 시간, 토큰 수 등)
    """
    # Chat 템플릿 형식으로 변환
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 토크나이저의 chat 템플릿 적용
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 토크나이징
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    input_token_count = inputs['input_ids'].shape[1]
    
    # 생성 시작 시간
    generation_start = time.time()
    
    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 생성 완료 시간
    generation_time = time.time() - generation_start
    
    # 디코딩
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 생성된 토큰 수
    generated_token_count = len(generated_tokens)
    total_token_count = outputs[0].shape[0]
    
    # 통계 정보
    stats = {
        "generation_time": generation_time,
        "input_tokens": input_token_count,
        "generated_tokens": generated_token_count,
        "total_tokens": total_token_count,
        "tokens_per_second": generated_token_count / generation_time if generation_time > 0 else 0
    }
    
    return response.strip(), stats

def interactive_chat(model_dir=None):
    """대화형 채팅 인터페이스 (서버 모드)"""
    global _model, _tokenizer
    
    print("=" * 50)
    print("Phi-4 4BIT 양자화 모델 - 서버 모드")
    print("=" * 50)
    print("모델을 한 번 로드한 후 메모리에 유지합니다.")
    print("=" * 50)
    print()
    
    # 모델 로드 (한 번만)
    model, tokenizer = load_quantized_model(model_dir)
    
    print("\n" + "=" * 50)
    print("대화를 시작하세요. 종료하려면 'quit', 'exit', 또는 'q'를 입력하세요.")
    print("=" * 50)
    print()
    
    conversation_history = []
    
    while True:
        try:
            # 멀티라인 입력 받기
            print("사용자: ", end="", flush=True)
            lines = []
            
            while True:
                try:
                    line = input()
                    # 빈 줄도 포함 (입력의 일부로 간주)
                    lines.append(line)
                    # Ctrl+D (EOF) 또는 Ctrl+C로 입력 종료
                except EOFError:
                    # Ctrl+D로 입력 종료
                    break
                except KeyboardInterrupt:
                    # Ctrl+C로 취소
                    print("\n입력이 취소되었습니다.\n")
                    lines = []
                    break
            
            user_input = "\n".join(lines).strip()
            
            # 빈 입력 처리
            if not user_input:
                continue
            
            # 종료 명령어 확인
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n대화를 종료합니다.")
                break
            
            # 대화 기록에 추가
            conversation_history.append({"role": "user", "content": user_input})
            
            print("\n응답 생성 중...")
            
            # 응답 생성
            response, stats = generate_response(
                model, 
                tokenizer, 
                user_input,
                max_new_tokens=512,
                temperature=0.7
            )
            
            # 응답 출력
            print(f"\nPhi-4: {response}")
            print("\n📊 생성 통계:")
            print(f"  생성 시간: {stats['generation_time']:.2f}초")
            print(f"  입력 토큰 수: {stats['input_tokens']}")
            print(f"  생성된 토큰 수: {stats['generated_tokens']}")
            print(f"  총 토큰 수: {stats['total_tokens']}")
            print(f"  생성 속도: {stats['tokens_per_second']:.2f} 토큰/초\n")
            
            # 대화 기록에 응답 추가
            conversation_history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\n\n대화를 종료합니다.")
            break
        except Exception as e:
            print(f"\n오류 발생: {e}")
            print("계속 진행합니다...\n")

def single_prompt(model_dir=None):
    """단일 프롬프트 실행 - 스크립트 내부에 작성된 프롬프트 사용"""
    # 스크립트 내부 프롬프트 사용
    prompt = SCRIPT_PROMPT.strip()
    
    if not prompt or prompt == "당신의 프롬프트를 여기에 작성하세요.\n예: \"Python으로 피보나치 수열을 계산하는 함수를 작성해주세요\"":
        print("=" * 50)
        print("[WARNING] 프롬프트가 설정되지 않았습니다.")
        print("=" * 50)
        print("phi4_server.py 파일의 SCRIPT_PROMPT 변수를 수정하여")
        print("실행할 프롬프트를 작성하세요.")
        print("=" * 50)
        sys.exit(1)
    
    print("=" * 50)
    print("Phi-4 4BIT 양자화 모델 - 서버 모드")
    print("=" * 50)
    
    # 모델 로드
    model, tokenizer = load_quantized_model(model_dir)
    
    print(f"\n프롬프트: {prompt}\n")
    print("응답 생성 중...\n")
    
    response, stats = generate_response(model, tokenizer, prompt)
    
    print("=" * 50)
    print("응답:")
    print("=" * 50)
    print(response)
    print("=" * 50)
    print("\n📊 생성 통계:")
    print(f"  생성 시간: {stats['generation_time']:.2f}초")
    print(f"  입력 토큰 수: {stats['input_tokens']}")
    print(f"  생성된 토큰 수: {stats['generated_tokens']}")
    print(f"  총 토큰 수: {stats['total_tokens']}")
    print(f"  생성 속도: {stats['tokens_per_second']:.2f} 토큰/초")
    print("=" * 50)
    print()
    print("모델이 메모리에 유지되어 있습니다.")
    print("다시 실행하면 빠르게 응답할 수 있습니다.")

def run_api_server(port=8000, host="0.0.0.0"):
    """API 서버 모드로 실행
    
    Args:
        port: 포트 번호 (기본값: 8000)
        host: 서버 주소 (기본값: '0.0.0.0')
    """
    try:
        from model_api_server import app
        import uvicorn
        
        print("=" * 50)
        print("Phi-4 API 서버 모드로 실행")
        print("=" * 50)
        print(f"서버 주소: {host}:{port}")
        print("=" * 50)
        
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        print("=" * 50)
        print("오류: model_api_server 모듈을 찾을 수 없습니다.")
        print("=" * 50)
        print("model_api_server.py 파일이 같은 디렉토리에 있는지 확인하세요.")
        sys.exit(1)
    except Exception as e:
        print(f"API 서버 실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phi-4 서버 모드 실행")
    parser.add_argument("--interactive", "-i", action="store_true", help="대화형 모드로 실행")
    parser.add_argument("--model-dir", type=str, help="로컬 저장된 모델 디렉토리 경로")
    parser.add_argument("--api-server", action="store_true", help="API 서버 모드로 실행")
    parser.add_argument("--port", type=int, default=8000, help="API 서버 포트 (기본값: 8000)")
    
    args = parser.parse_args()
    
    # API 서버 모드
    if args.api_server:
        run_api_server(port=args.port)
    # --interactive 플래그가 있거나 인자가 없으면 대화형 모드
    elif args.interactive or len(sys.argv) == 1:
        interactive_chat(args.model_dir)
    else:
        # 단일 프롬프트 모드 (스크립트 내부 프롬프트 사용)
        print("=" * 50)
        print("Phi-4 4BIT 양자화 모델 - 서버 모드")
        print("=" * 50)
        
        model, tokenizer = load_quantized_model(args.model_dir)
        single_prompt(args.model_dir)


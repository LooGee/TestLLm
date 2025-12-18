"""
Phi-4 4BIT 양자화 실행 스크립트
bitsandbytes를 사용하여 4BIT 양자화된 phi-4 모델을 로드하고 실행합니다.
로컬에 저장된 양자화 모델이 있으면 우선 사용합니다.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys
import os
import argparse
import time
from device_utils import check_and_setup_device, get_device_info

def load_quantized_model(model_dir=None):
    """4BIT 양자화된 phi-4 모델을 로드합니다.
    
    Args:
        model_dir: 로컬에 저장된 모델 디렉토리 경로 (None이면 기본 경로 또는 Hugging Face에서 로드)
    """
    # GPU/CPU 확인 및 설정
    device, device_name, is_gpu = check_and_setup_device()
    
    # 기본 로컬 모델 경로
    default_local_dir = "./models/phi4-quantized"
    
    # 모델 디렉토리 결정
    if model_dir is None:
        # 명령줄 인자 확인
        if len(sys.argv) > 1 and sys.argv[1].startswith("--model-dir"):
            if "=" in sys.argv[1]:
                model_dir = sys.argv[1].split("=", 1)[1]
            elif len(sys.argv) > 2:
                model_dir = sys.argv[2]
        # 기본 로컬 경로 확인
        elif os.path.exists(default_local_dir) and os.path.isdir(default_local_dir):
            model_dir = default_local_dir
    
    model_name = "microsoft/phi-4"
    
    # 4BIT 양자화 설정
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
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
        
        print("로컬 모델 로딩 완료!")
        
        # 모델 디바이스 정보 확인
        device_info = get_device_info(model)
        print(f"\n모델 실행 디바이스: {device_info['device_name']} ({device_info['device']})")
        if device_info['is_gpu']:
            print("✅ GPU 모드로 실행 중")
        else:
            print("⚠️  CPU 모드로 실행 중 (GPU 사용 권장)")
        
        return model, tokenizer
    
    # 로컬 모델이 없으면 Hugging Face에서 로드
    print(f"모델 로딩 중: {model_name}")
    print("4BIT 양자화 설정 적용 중...")
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
    # GPU가 사용 가능하면 명시적으로 cuda 사용
    if torch.cuda.is_available():
        print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        print("GPU에 모델을 로드합니다...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir
    )
    
    print("모델 로딩 완료!")
    
    # 모델 디바이스 정보 확인
    device_info = get_device_info(model)
    print(f"\n모델 실행 디바이스: {device_info['device_name']} ({device_info['device']})")
    if device_info['is_gpu']:
        print("✅ GPU 모드로 실행 중")
    else:
        print("⚠️  CPU 모드로 실행 중 (GPU 사용 권장)")
    
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
    # 입력 길이만큼 제외하고 새로 생성된 부분만 디코딩
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

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Phi-4 4BIT 양자화 모델 실행")
    parser.add_argument("prompt", nargs="?", help="입력 프롬프트")
    parser.add_argument("--model-dir", type=str, help="로컬 저장된 모델 디렉토리 경로")
    
    args = parser.parse_args()
    
    if not args.prompt:
        print("사용법: python phi4_quantized.py <프롬프트> [--model-dir <경로>]")
        print("또는: python prompt_input.py 를 사용하여 대화형으로 실행하세요.")
        sys.exit(1)
    
    prompt = args.prompt
    
    print("=" * 50)
    print("Phi-4 4BIT 양자화 모델 실행")
    print("=" * 50)
    
    # 모델 로드
    model, tokenizer = load_quantized_model()
    
    print(f"\n프롬프트: {prompt}\n")
    print("응답 생성 중...\n")
    
    # 응답 생성
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

if __name__ == "__main__":
    main()


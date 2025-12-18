"""
Ollama API를 사용하여 모델과 상호작용하는 예제 스크립트
Ollama가 실행 중이어야 합니다.
다른 스크립트들과 동일한 기능을 제공합니다.
"""
import requests
import json
import sys
import argparse
import time
import re

# 스크립트 내부에 작성할 프롬프트
# 이 변수를 수정하여 실행할 프롬프트를 지정하세요
SCRIPT_PROMPT = """
당신의 프롬프트를 여기에 작성하세요.
예: "Python으로 피보나치 수열을 계산하는 함수를 작성해주세요"
"""

def extract_json_from_response(response_text):
    """
    응답 텍스트에서 JSON 객체만 추출합니다.
    마크다운 코드 블록(```json ... ```)이나 설명 텍스트를 제거하고 순수 JSON만 반환합니다.
    
    Args:
        response_text: 모델 응답 텍스트
    
    Returns:
        str: 추출된 JSON 문자열 (없으면 원본 반환)
    """
    if not response_text:
        return response_text
    
    # 1. 마크다운 코드 블록에서 JSON 추출 (```json ... ``` 또는 ``` ... ```)
    json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_block_pattern, response_text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        # 유효한 JSON인지 확인
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass
    
    # 2. 첫 번째 { 부터 시작하여 중괄호 매칭으로 완전한 JSON 객체 추출
    start_idx = response_text.find('{')
    if start_idx != -1:
        brace_count = 0
        end_idx = start_idx
        
        for i in range(start_idx, len(response_text)):
            if response_text[i] == '{':
                brace_count += 1
            elif response_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if brace_count == 0:
            json_str = response_text[start_idx:end_idx].strip()
            # 유효한 JSON인지 확인
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass
    
    # 3. JSON이 없으면 원본 반환
    return response_text

def check_ollama_installed():
    """
    Ollama가 시스템에 설치되어 있는지 확인합니다.
    
    Returns:
        bool: Ollama가 설치되어 있으면 True
    """
    import subprocess
    import shutil
    
    # ollama 명령어가 PATH에 있는지 확인
    if shutil.which("ollama"):
        return True
    
    # Windows에서 일반적인 설치 경로 확인
    import os
    if os.name == 'nt':  # Windows
        common_paths = [
            os.path.expanduser("~\\AppData\\Local\\Programs\\Ollama\\ollama.exe"),
            "C:\\Program Files\\Ollama\\ollama.exe",
        ]
        for path in common_paths:
            if os.path.exists(path):
                return True
    
    return False

def check_ollama_server(base_url="http://localhost:11434"):
    """
    Ollama 서버가 실행 중인지 확인하고 사용 가능한 모델 목록을 반환합니다.
    
    Args:
        base_url: Ollama 서버 URL
    
    Returns:
        tuple: (is_available, models_list)
            - is_available: 서버가 실행 중이면 True
            - models_list: 사용 가능한 모델 목록
    """
    # 먼저 Ollama가 설치되어 있는지 확인
    if not check_ollama_installed():
        return False, []
    
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            result = response.json()
            models = [model.get("name", "") for model in result.get("models", [])]
            return True, models
        return False, []
    except:
        return False, []

def check_model_exists(model, base_url="http://localhost:11434"):
    """
    특정 모델이 Ollama에 설치되어 있는지 확인합니다.
    
    Args:
        model: 확인할 모델 이름
        base_url: Ollama 서버 URL
    
    Returns:
        bool: 모델이 존재하면 True
    """
    is_available, models = check_ollama_server(base_url)
    if not is_available:
        return False
    
    # 모델 이름 매칭 (정확한 이름 또는 접두사)
    for available_model in models:
        if model == available_model or available_model.startswith(model + ":"):
            return True
    return False

def query_ollama(prompt, model="phi4-quantized", base_url="http://localhost:11434", system_prompt=None):
    """
    Ollama API를 통해 모델에 쿼리를 보냅니다.
    
    Args:
        prompt: 입력 프롬프트
        model: 사용할 모델 이름 (기본값: phi4-quantized)
        base_url: Ollama 서버 URL
        system_prompt: 시스템 프롬프트 (선택사항)
    
    Returns:
        tuple: (response, stats_dict)
            - response: 생성된 응답 텍스트
            - stats_dict: 통계 정보 (생성 시간, 토큰 수 등)
    """
    # Ollama 설치 확인
    if not check_ollama_installed():
        print("=" * 50)
        print("⚠️  Ollama가 설치되어 있지 않습니다")
        print("=" * 50)
        print("Ollama 설치 방법:")
        print("1. Windows: https://ollama.com/ 에서 다운로드 및 설치")
        print("2. Linux: curl -fsSL https://ollama.com/install.sh | sh")
        print("3. Mac: brew install ollama")
        print("\n설치 후 Ollama 앱을 실행하거나 'ollama serve' 명령어로 서버를 시작하세요.")
        print("=" * 50)
        return None, None
    
    # 서버 연결 확인
    is_available, models = check_ollama_server(base_url)
    if not is_available:
        print("=" * 50)
        print("⚠️  Ollama 서버에 연결할 수 없습니다")
        print("=" * 50)
        print(f"서버 URL: {base_url}")
        print("\n확인 사항:")
        print("1. Ollama 서버가 실행 중인지 확인")
        print("   - Windows: Ollama 앱이 실행 중인지 확인 (시작 메뉴에서 'Ollama' 검색)")
        print("   - Linux/Mac: 'ollama serve' 명령어로 서버 시작")
        print("2. 서버 URL이 올바른지 확인 (기본값: http://localhost:11434)")
        print("3. 방화벽이 포트 11434를 차단하지 않는지 확인")
        print("=" * 50)
        return None, None
    
    # 모델 존재 확인
    if not check_model_exists(model, base_url):
        print("=" * 50)
        print(f"⚠️  모델 '{model}'을(를) 찾을 수 없습니다")
        print("=" * 50)
        print(f"사용 가능한 모델 목록:")
        if models:
            for m in models:
                print(f"  - {m}")
        else:
            print("  (모델이 설치되어 있지 않습니다)")
        print(f"\n모델을 다운로드하려면:")
        print(f"  ollama pull {model}")
        print("=" * 50)
        return None, None
    
    url = f"{base_url}/api/chat"
    
    # messages 리스트 구성
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # prompt가 문자열이면 user 메시지로 추가
    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    else:
        # prompt가 이미 messages 리스트인 경우 (향후 확장성)
        messages.extend(prompt)
    
    data = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    
    try:
        generation_start = time.time()
        response = requests.post(url, json=data, timeout=300)
        response.raise_for_status()
        result = response.json()
        generation_time = time.time() - generation_start
        
        # /api/chat 응답 형식: {"message": {"role": "assistant", "content": "..."}}
        message = result.get("message", {})
        response_text = message.get("content", "")
        
        # JSON만 추출 (마크다운 코드 블록이나 설명 텍스트 제거)
        response_text = extract_json_from_response(response_text)
        
        # 통계 정보 추출 (/api/chat 응답 형식)
        stats = {
            "generation_time": generation_time,
            "total_duration": result.get("total_duration", 0) / 1e9 if result.get("total_duration") else 0,  # 나노초를 초로 변환
            "load_duration": result.get("load_duration", 0) / 1e9 if result.get("load_duration") else 0,
            "prompt_eval_count": result.get("prompt_eval_count", 0),
            "eval_count": result.get("eval_count", 0),
            "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
            "tokens_per_second": result.get("eval_count", 0) / generation_time if generation_time > 0 and result.get("eval_count") else 0
        }
        
        return response_text, stats
    except requests.exceptions.ConnectionError as e:
        print("=" * 50)
        print("⚠️  Ollama 서버 연결 오류")
        print("=" * 50)
        print(f"서버 URL: {base_url}")
        print(f"오류: {e}")
        print("\nOllama 서버가 실행 중인지 확인하세요:")
        print("1. Ollama 앱을 실행하세요")
        print("2. 또는 터미널에서 'ollama serve' 명령어를 실행하세요")
        print("=" * 50)
        return None, None
    except requests.exceptions.RequestException as e:
        print(f"API 요청 오류: {e}")
        return None, None

def interactive_chat(model="phi4-quantized", base_url="http://localhost:11434", system_prompt=None):
    """대화형 채팅 인터페이스"""
    print("=" * 50)
    print(f"Ollama API - {model} 모델")
    print("=" * 50)
    
    # Ollama 설치 확인
    if not check_ollama_installed():
        print("=" * 50)
        print("⚠️  Ollama가 설치되어 있지 않습니다")
        print("=" * 50)
        print("Ollama 설치 방법:")
        print("1. Windows: https://ollama.com/ 에서 다운로드 및 설치")
        print("2. Linux: curl -fsSL https://ollama.com/install.sh | sh")
        print("3. Mac: brew install ollama")
        print("\n설치 후 Ollama 앱을 실행하거나 'ollama serve' 명령어로 서버를 시작하세요.")
        print("=" * 50)
        return
    
    # 서버 연결 확인
    print("Ollama 서버 연결 확인 중...")
    is_available, models = check_ollama_server(base_url)
    if not is_available:
        print("⚠️  Ollama 서버에 연결할 수 없습니다.")
        print(f"서버 URL: {base_url}")
        print("\nOllama 서버를 시작하세요:")
        print("1. Windows: 시작 메뉴에서 'Ollama' 앱을 실행하세요")
        print("2. Linux/Mac: 터미널에서 'ollama serve' 명령어를 실행하세요")
        print("\n서버가 실행되면 다시 시도하세요.")
        return
    else:
        print("✅ Ollama 서버 연결 성공!")
        if models:
            print(f"사용 가능한 모델: {', '.join(models[:5])}" + (f" 외 {len(models)-5}개" if len(models) > 5 else ""))
        
        # 모델 존재 확인
        if not check_model_exists(model, base_url):
            print(f"\n⚠️  모델 '{model}'을(를) 찾을 수 없습니다.")
            print(f"사용 가능한 모델 목록:")
            for m in models:
                print(f"  - {m}")
            print(f"\n모델을 다운로드하려면:")
            print(f"  ollama pull {model}")
            return
    
    print("\n대화를 시작하세요. 종료하려면 'quit', 'exit', 또는 'q'를 입력하세요.")
    print("멀티라인 입력: 여러 줄 입력 후 Ctrl+D (Windows: Ctrl+Z 후 Enter)로 완료")
    if system_prompt:
        print(f"시스템 프롬프트: {system_prompt[:50]}...")
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
            response, stats = query_ollama(user_input, model=model, base_url=base_url, system_prompt=system_prompt)
            
            if response and stats:
                print(f"\n{model}: {response}")
                print("\n📊 생성 통계:")
                print(f"  생성 시간: {stats['generation_time']:.2f}초")
                print(f"  프롬프트 토큰 수: {stats['prompt_eval_count']}")
                print(f"  생성된 토큰 수: {stats['eval_count']}")
                print(f"  총 토큰 수: {stats['total_tokens']}")
                print(f"  생성 속도: {stats['tokens_per_second']:.2f} 토큰/초")
                print()
                
                # 대화 기록에 응답 추가
                conversation_history.append({"role": "assistant", "content": response})
            else:
                print("\n응답을 받을 수 없습니다. Ollama 서버가 실행 중인지 확인하세요.\n")
            
        except KeyboardInterrupt:
            print("\n\n대화를 종료합니다.")
            break
        except Exception as e:
            print(f"\n오류 발생: {e}")
            print("계속 진행합니다...\n")

def single_prompt(model="phi4-quantized", base_url="http://localhost:11434", system_prompt=None):
    """단일 프롬프트 실행 - 스크립트 내부에 작성된 프롬프트 사용"""
    # 스크립트 내부 프롬프트 사용
    prompt = SCRIPT_PROMPT.strip()
    
    if not prompt or prompt == "당신의 프롬프트를 여기에 작성하세요.\n예: \"Python으로 피보나치 수열을 계산하는 함수를 작성해주세요\"":
        print("=" * 50)
        print("⚠️  프롬프트가 설정되지 않았습니다.")
        print("=" * 50)
        print("ollama_api_example.py 파일의 SCRIPT_PROMPT 변수를 수정하여")
        print("실행할 프롬프트를 작성하세요.")
        print("=" * 50)
        sys.exit(1)
    
    print("=" * 50)
    print(f"Ollama API - {model} 모델")
    print("=" * 50)
    
    # Ollama 설치 확인
    if not check_ollama_installed():
        print("=" * 50)
        print("⚠️  Ollama가 설치되어 있지 않습니다")
        print("=" * 50)
        print("Ollama 설치 방법:")
        print("1. Windows: https://ollama.com/ 에서 다운로드 및 설치")
        print("2. Linux: curl -fsSL https://ollama.com/install.sh | sh")
        print("3. Mac: brew install ollama")
        print("\n설치 후 Ollama 앱을 실행하거나 'ollama serve' 명령어로 서버를 시작하세요.")
        print("=" * 50)
        sys.exit(1)
    
    # 서버 연결 확인
    print("Ollama 서버 연결 확인 중...")
    is_available, models = check_ollama_server(base_url)
    if not is_available:
        print("⚠️  Ollama 서버에 연결할 수 없습니다.")
        print(f"서버 URL: {base_url}")
        print("\nOllama 서버를 시작하세요:")
        print("1. Windows: 시작 메뉴에서 'Ollama' 앱을 실행하세요")
        print("2. Linux/Mac: 터미널에서 'ollama serve' 명령어를 실행하세요")
        sys.exit(1)
    else:
        print("✅ Ollama 서버 연결 성공!")
        if models:
            print(f"사용 가능한 모델: {', '.join(models[:5])}" + (f" 외 {len(models)-5}개" if len(models) > 5 else ""))
        
        # 모델 존재 확인
        if not check_model_exists(model, base_url):
            print(f"\n⚠️  모델 '{model}'을(를) 찾을 수 없습니다.")
            print(f"사용 가능한 모델 목록:")
            for m in models:
                print(f"  - {m}")
            print(f"\n모델을 다운로드하려면:")
            print(f"  ollama pull {model}")
            sys.exit(1)
    
    print(f"\n프롬프트: {prompt}\n")
    if system_prompt:
        print(f"시스템 프롬프트: {system_prompt}\n")
    print("응답 생성 중...\n")
    
    response, stats = query_ollama(prompt, model=model, base_url=base_url, system_prompt=system_prompt)
    
    if response and stats:
        print("=" * 50)
        print("응답:")
        print("=" * 50)
        print(response)
        print("=" * 50)
        print("\n📊 생성 통계:")
        print(f"  생성 시간: {stats['generation_time']:.2f}초")
        print(f"  프롬프트 토큰 수: {stats['prompt_eval_count']}")
        print(f"  생성된 토큰 수: {stats['eval_count']}")
        print(f"  총 토큰 수: {stats['total_tokens']}")
        print(f"  생성 속도: {stats['tokens_per_second']:.2f} 토큰/초")
        print("=" * 50)
    else:
        print("응답을 받을 수 없습니다. Ollama 서버가 실행 중인지 확인하세요.")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Ollama API 실행")
    parser.add_argument("--interactive", "-i", action="store_true", help="대화형 모드로 실행")
    parser.add_argument("--model", "-m", type=str, default="phi4-quantized", help="사용할 모델 이름 (기본값: phi4-quantized)")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434", help="Ollama 서버 URL")
    parser.add_argument("--system-prompt", "-s", type=str, default=None, help="시스템 프롬프트 설정")
    
    args = parser.parse_args()
    
    # --interactive 플래그가 있거나 인자가 없으면 대화형 모드
    if args.interactive or len(sys.argv) == 1:
        interactive_chat(model=args.model, base_url=args.base_url, system_prompt=args.system_prompt)
    else:
        # 단일 프롬프트 모드 (스크립트 내부 프롬프트 사용)
        single_prompt(model=args.model, base_url=args.base_url, system_prompt=args.system_prompt)

if __name__ == "__main__":
    # requests 라이브러리 확인
    try:
        import requests
    except ImportError:
        print("requests 라이브러리가 필요합니다. 다음 명령어로 설치하세요:")
        print("pip install requests")
        sys.exit(1)
    
    main()


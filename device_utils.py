"""
디바이스 확인 및 GPU 설정 유틸리티
"""
import torch

def check_and_setup_device():
    """
    GPU/CPU 확인 및 GPU를 기본값으로 설정합니다.
    
    Returns:
        tuple: (device, device_name, is_gpu)
    """
    # CUDA 사용 가능 여부 확인
    cuda_available = torch.cuda.is_available()
    
    # 디버깅 정보
    if not cuda_available:
        print(f"[DEBUG] torch.cuda.is_available() = {cuda_available}")
        print(f"[DEBUG] PyTorch 버전: {torch.__version__}")
        try:
            print(f"[DEBUG] CUDA 버전: {torch.version.cuda}")
        except:
            print("[DEBUG] CUDA 버전 정보를 가져올 수 없습니다")
    
    if cuda_available:
        try:
            device = torch.device("cuda:0")
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            is_gpu = True
            
            print("=" * 50)
            print("[OK] GPU 감지됨")
            print("=" * 50)
            print(f"디바이스: {device}")
            print(f"GPU 이름: {device_name}")
            print(f"사용 가능한 GPU 개수: {device_count}")
            
            # GPU 메모리 정보
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            print(f"GPU 메모리: {memory_allocated:.2f}GB 할당됨 / {memory_reserved:.2f}GB 예약됨 / {memory_total:.2f}GB 전체")
            print("=" * 50)
        except Exception as e:
            # GPU 정보를 가져오는 중 오류 발생
            print("=" * 50)
            print("[WARNING] GPU 감지 중 오류 발생")
            print("=" * 50)
            print(f"오류: {e}")
            print("CUDA는 사용 가능하지만 GPU 정보를 가져올 수 없습니다.")
            print("=" * 50)
            device = torch.device("cuda:0")
            device_name = "CUDA (정보 없음)"
            is_gpu = True
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        is_gpu = False
        
        print("=" * 50)
        print("[WARNING] GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
        print("=" * 50)
        print("디바이스: CPU")
        print("\nGPU 사용을 위한 확인 사항:")
        print("1. NVIDIA GPU가 설치되어 있는지 확인")
        print("2. CUDA가 설치되어 있는지 확인: nvidia-smi")
        print("3. PyTorch가 CUDA를 지원하는 버전인지 확인")
        print("4. torch.cuda.is_available() 결과 확인")
        print("=" * 50)
    
    return device, device_name, is_gpu

def get_device_info(model):
    """
    모델이 사용 중인 디바이스 정보를 반환합니다.
    bitsandbytes 양자화 모델을 포함한 다양한 모델 구조를 지원합니다.
    
    Args:
        model: PyTorch 모델
    
    Returns:
        dict: 디바이스 정보
    """
    device = None
    device_name = "CPU"
    is_gpu = False
    
    # 방법 1: hf_device_map 확인 (device_map="auto" 사용 시, 가장 신뢰할 수 있음)
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # device_map이 딕셔너리인 경우
        devices = list(model.hf_device_map.values())
        if devices:
            first_device = devices[0]
            if isinstance(first_device, torch.device):
                device = first_device
            elif isinstance(first_device, str):
                # "cuda:0", "cuda", "cpu" 등의 문자열 처리
                if "cuda" in first_device.lower():
                    if ":" in first_device:
                        device_index = int(first_device.split(":")[1])
                        device = torch.device(f"cuda:{device_index}")
                    else:
                        device = torch.device("cuda:0")
                else:
                    device = torch.device(first_device)
            elif isinstance(first_device, int):
                device = torch.device(f"cuda:{first_device}")
    
    # 방법 2: 모델의 device 속성 확인
    if device is None and hasattr(model, 'device'):
        device = model.device
    
    # 방법 3: 모델 파라미터의 디바이스 확인 (양자화 모델 포함)
    if device is None:
        try:
            # 양자화된 모델의 경우 여러 경로로 시도
            if hasattr(model, 'model'):
                # model.model 구조 확인
                if hasattr(model.model, 'embed_tokens'):
                    device = next(model.model.embed_tokens.parameters()).device
                elif hasattr(model.model, 'layers') and len(model.model.layers) > 0:
                    # 첫 번째 레이어의 파라미터 확인
                    device = next(model.model.layers[0].parameters()).device
                elif hasattr(model.model, 'lm_head'):
                    device = next(model.model.lm_head.parameters()).device
            elif hasattr(model, 'base_model'):
                # base_model 구조 확인
                if hasattr(model.base_model, 'embed_tokens'):
                    device = next(model.base_model.embed_tokens.parameters()).device
                elif hasattr(model.base_model, 'layers') and len(model.base_model.layers) > 0:
                    device = next(model.base_model.layers[0].parameters()).device
            else:
                # 일반적인 방법: 첫 번째 파라미터 확인
                device = next(model.parameters()).device
        except (StopIteration, AttributeError, RuntimeError):
            # 파라미터를 찾을 수 없는 경우
            pass
    
    # 방법 4: CUDA 사용 가능 여부로 기본값 설정
    if device is None:
        if torch.cuda.is_available():
            # GPU가 사용 가능하면 cuda:0 사용
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    
    # 디바이스 타입 확인
    is_gpu = device.type == "cuda"
    
    # GPU 이름 가져오기
    if is_gpu:
        try:
            device_index = device.index if device.index is not None else 0
            device_name = torch.cuda.get_device_name(device_index)
        except (RuntimeError, AttributeError):
            # GPU 이름을 가져올 수 없는 경우
            device_name = f"CUDA Device {device_index}"
    else:
        device_name = "CPU"
    
    return {
        "device": device,
        "device_name": device_name,
        "is_gpu": is_gpu
    }


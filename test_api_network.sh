#!/bin/bash
# 노트북에서 네트워크 테스트 스크립트
# Mac Terminal에서 실행

# PC의 IP 주소를 입력받거나 환경 변수에서 읽기
PC_IP=${PC_IP:-"192.168.0.XXX"}
PORT=8000
BASE_URL="http://${PC_IP}:${PORT}"

echo "=================================================="
echo "Phi-4 API 서버 네트워크 테스트"
echo "=================================================="
echo ""

# PC IP 확인
if [ "$PC_IP" = "192.168.0.XXX" ]; then
    echo "⚠️  PC IP 주소를 설정하세요:"
    echo "   export PC_IP=192.168.0.100"
    echo "   또는 스크립트를 수정하여 PC_IP 변수를 설정하세요"
    echo ""
    read -p "PC의 IP 주소를 입력하세요: " PC_IP
    BASE_URL="http://${PC_IP}:${PORT}"
fi

echo "테스트 대상: $BASE_URL"
echo ""

# 1. Health Check 테스트
echo "[1/3] Health Check 테스트..."
if curl -s -f "${BASE_URL}/api/health" > /dev/null; then
    HEALTH_RESPONSE=$(curl -s "${BASE_URL}/api/health")
    echo "✅ Health Check 성공!"
    echo "$HEALTH_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$HEALTH_RESPONSE"
    echo ""
else
    echo "❌ Health Check 실패"
    echo "   서버가 실행 중인지, PC IP 주소가 올바른지 확인하세요"
    exit 1
fi

# 2. Generate 엔드포인트 테스트
echo "[2/3] Generate 엔드포인트 테스트..."
GENERATE_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "안녕하세요. 간단히 자기소개를 해주세요.",
        "max_new_tokens": 100,
        "temperature": 0.7
    }')

if [ $? -eq 0 ]; then
    echo "✅ Generate 요청 성공!"
    echo "$GENERATE_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$GENERATE_RESPONSE"
    echo ""
else
    echo "❌ Generate 요청 실패"
    exit 1
fi

# 3. 완료 메시지
echo "[3/3] 테스트 완료"
echo "✅ 모든 테스트 완료!"
echo ""
echo "FastAPI 자동 생성 문서를 확인하려면 브라우저에서 다음 URL을 열어보세요:"
echo "   - Swagger UI: ${BASE_URL}/docs"
echo "   - ReDoc: ${BASE_URL}/redoc"
echo ""


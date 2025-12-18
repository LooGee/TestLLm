# 로컬 API 서버 테스트 스크립트
# PowerShell에서 실행

Write-Host "=" * 50
Write-Host "Phi-4 API 서버 로컬 테스트"
Write-Host "=" * 50
Write-Host ""

# 서버 URL
$baseUrl = "http://localhost:8000"

# 1. Health Check 테스트
Write-Host "[1/3] Health Check 테스트..." -ForegroundColor Cyan
try {
    $healthResponse = Invoke-RestMethod -Uri "$baseUrl/api/health" -Method GET
    Write-Host "✅ Health Check 성공!" -ForegroundColor Green
    Write-Host "   상태: $($healthResponse.status)" -ForegroundColor White
    Write-Host "   모델 로드: $($healthResponse.model_loaded)" -ForegroundColor White
    Write-Host ""
} catch {
    Write-Host "❌ Health Check 실패: $_" -ForegroundColor Red
    Write-Host "   서버가 실행 중인지 확인하세요: python model_api_server.py" -ForegroundColor Yellow
    exit 1
}

# 2. Generate 엔드포인트 테스트
Write-Host "[2/3] Generate 엔드포인트 테스트..." -ForegroundColor Cyan
try {
    $body = @{
        prompt = "안녕하세요. 간단히 자기소개를 해주세요."
        max_new_tokens = 100
        temperature = 0.7
    } | ConvertTo-Json

    $generateResponse = Invoke-RestMethod -Uri "$baseUrl/api/generate" `
        -Method POST `
        -Body $body `
        -ContentType "application/json"

    Write-Host "✅ Generate 요청 성공!" -ForegroundColor Green
    Write-Host "   응답: $($generateResponse.response.Substring(0, [Math]::Min(100, $generateResponse.response.Length)))..." -ForegroundColor White
    Write-Host "   생성 시간: $($generateResponse.stats.generation_time)초" -ForegroundColor White
    Write-Host "   생성된 토큰 수: $($generateResponse.stats.generated_tokens)" -ForegroundColor White
    Write-Host "   토큰/초: $([Math]::Round($generateResponse.stats.tokens_per_second, 2))" -ForegroundColor White
    Write-Host ""
} catch {
    Write-Host "❌ Generate 요청 실패: $_" -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "   응답: $responseBody" -ForegroundColor Yellow
    }
    exit 1
}

# 3. FastAPI 문서 확인 안내
Write-Host "[3/3] FastAPI 자동 문서 확인" -ForegroundColor Cyan
Write-Host "✅ 모든 테스트 완료!" -ForegroundColor Green
Write-Host ""
Write-Host "FastAPI 자동 생성 문서를 확인하려면 브라우저에서 다음 URL을 열어보세요:" -ForegroundColor Yellow
Write-Host "   - Swagger UI: $baseUrl/docs" -ForegroundColor Cyan
Write-Host "   - ReDoc: $baseUrl/redoc" -ForegroundColor Cyan
Write-Host ""


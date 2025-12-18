# Windows 방화벽 설정 스크립트
# API 서버 포트(8000)에 대한 로컬 네트워크(192.168.0.0/24) 인바운드 규칙 추가

# 관리자 권한 확인
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "오류: 이 스크립트는 관리자 권한으로 실행해야 합니다." -ForegroundColor Red
    Write-Host "PowerShell을 관리자 권한으로 실행한 후 다시 시도하세요." -ForegroundColor Yellow
    exit 1
}

Write-Host "=" * 50
Write-Host "Windows 방화벽 설정"
Write-Host "=" * 50

# 기존 규칙 확인
$existingRule = Get-NetFirewallRule -DisplayName "Model-API-Server" -ErrorAction SilentlyContinue

if ($existingRule) {
    Write-Host "기존 방화벽 규칙이 발견되었습니다." -ForegroundColor Yellow
    $response = Read-Host "기존 규칙을 삭제하고 새로 만들까요? (Y/N)"
    if ($response -eq "Y" -or $response -eq "y") {
        Remove-NetFirewallRule -DisplayName "Model-API-Server"
        Write-Host "기존 규칙 삭제 완료" -ForegroundColor Green
    } else {
        Write-Host "작업을 취소했습니다." -ForegroundColor Yellow
        exit 0
    }
}

# 새 방화벽 규칙 생성
try {
    New-NetFirewallRule -DisplayName "Model-API-Server" `
        -Direction Inbound `
        -LocalPort 8000 `
        -Protocol TCP `
        -RemoteAddress 192.168.0.0/24 `
        -Action Allow `
        -Description "Phi-4 Model API Server - 로컬 네트워크만 허용"
    
    Write-Host "방화벽 규칙 생성 완료!" -ForegroundColor Green
    Write-Host "포트 8000이 로컬 네트워크(192.168.0.0/24)에서 접근 가능합니다." -ForegroundColor Green
} catch {
    Write-Host "방화벽 규칙 생성 중 오류 발생: $_" -ForegroundColor Red
    exit 1
}

Write-Host "=" * 50


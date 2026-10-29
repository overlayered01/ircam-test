# Python 3.12 가상환경 생성 + 의존성 설치
# pyorbbecsdk2 는 Python 3.13 이하용 휠만 제공하므로 3.12 를 사용한다.
# py 런처에 3.12 가 없으면 uv 로 만든다 (uv 가 3.12 를 자동으로 받아 온다).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    $hasPy312 = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.12 -c "pass" 2>$null
        $hasPy312 = ($LASTEXITCODE -eq 0)
    }
    if ($hasPy312) {
        py -3.12 -m venv .venv
    } elseif (Get-Command uv -ErrorAction SilentlyContinue) {
        uv venv --python 3.12 .venv
    } else {
        throw "Python 3.12 (py -3.12) 도 uv 도 없습니다. 둘 중 하나를 설치하세요."
    }
}

# uv 로 만든 venv 에는 pip 이 없다 → uv pip 사용
.\.venv\Scripts\python -m pip --version 2>$null
if ($LASTEXITCODE -eq 0) {
    .\.venv\Scripts\python -m pip install --upgrade pip
    .\.venv\Scripts\python -m pip install -r requirements.txt
} else {
    uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
}
.\.venv\Scripts\python -c "import pyorbbecsdk, cv2; print('pyorbbecsdk', pyorbbecsdk.get_version(), '/ opencv', cv2.__version__)"

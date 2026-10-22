# Python 3.12 가상환경 생성 + 의존성 설치
# pyorbbecsdk2 는 Python 3.13 이하용 휠만 제공하므로 3.12 를 사용한다.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    py -3.12 -m venv .venv
}
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -c "import pyorbbecsdk, cv2; print('pyorbbecsdk', pyorbbecsdk.get_version(), '/ opencv', cv2.__version__)"

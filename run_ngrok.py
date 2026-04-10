"""HybridPdM - ngrok 터널링으로 Streamlit 앱 외부 공개.

사용법:
  # 방법 1: 환경변수로 토큰 전달
  set NGROK_TOKEN=your_token_here   (Windows)
  python run_ngrok.py

  # 방법 2: 인자로 직접 전달
  python run_ngrok.py --token your_token_here

  # 방법 3: .env 파일 사용 (NGROK_TOKEN=... 한 줄 작성)
  python run_ngrok.py

ngrok 토큰 발급: https://dashboard.ngrok.com/get-started/your-authtoken
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def get_token(cli_token: str | None) -> str:
    """토큰 우선순위: CLI 인자 > 환경변수 > .env 파일."""
    if cli_token:
        return cli_token

    if token := os.environ.get("NGROK_TOKEN"):
        return token

    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("NGROK_TOKEN="):
                return line.split("=", 1)[1].strip()

    return ""


def find_free_port(start: int = 8501, end: int = 8600) -> int:
    """start~end 범위에서 사용 가능한 포트를 반환."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"포트 {start}~{end} 범위에서 사용 가능한 포트를 찾을 수 없습니다.")


def main():
    parser = argparse.ArgumentParser(description="HybridPdM ngrok 배포")
    parser.add_argument("--token",  default=None, help="ngrok authtoken")
    parser.add_argument("--port",   type=int, default=None, help="Streamlit 포트 (미지정 시 자동 탐색)")
    parser.add_argument("--region", default="jp", help="ngrok 리전 (기본 jp, 한국 인접)")
    args = parser.parse_args()

    # ── 토큰 확인 ──────────────────────────────────────────────────
    token = get_token(args.token)
    if not token:
        print(
            "[오류] ngrok authtoken이 필요합니다.\n"
            "  1) https://dashboard.ngrok.com/get-started/your-authtoken 에서 발급\n"
            "  2) 실행: python run_ngrok.py --token <YOUR_TOKEN>\n"
            "     또는: set NGROK_TOKEN=<YOUR_TOKEN>  후  python run_ngrok.py"
        )
        sys.exit(1)

    try:
        from pyngrok import conf, ngrok
    except ImportError:
        print("[오류] pyngrok이 설치되지 않았습니다.\n  pip install pyngrok")
        sys.exit(1)

    # ── 포트 결정 ──────────────────────────────────────────────────
    if args.port:
        port = args.port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) == 0:
                print(f"[경고] 포트 {port}가 이미 사용 중입니다. 다른 포트를 자동 탐색합니다.")
                port = find_free_port(port + 1)
    else:
        port = find_free_port()

    print(f"[포트] {port} 사용")

    # ── ngrok 설정 ─────────────────────────────────────────────────
    conf.get_default().region = args.region
    ngrok.set_auth_token(token)

    # ── Streamlit 서버 백그라운드 실행 ─────────────────────────────
    app_path = Path(__file__).parent / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ]

    print(f"[1/3] Streamlit 시작 중... (포트 {port})")
    stderr_log = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr_log)

    # Streamlit 기동 대기 (최대 10초)
    for _ in range(10):
        time.sleep(1)
        if proc.poll() is not None:
            stderr_log.flush()
            log_content = Path(stderr_log.name).read_text(encoding="utf-8", errors="ignore")
            print(
                f"[오류] Streamlit이 시작되지 않았습니다.\n"
                f"--- 오류 로그 ---\n{log_content or '(로그 없음)'}\n"
                "의존성 확인: pip install -r requirements.txt"
            )
            sys.exit(1)
        # 포트가 열렸으면 준비 완료
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) == 0:
                break
    else:
        print(f"[경고] Streamlit 응답 대기 시간 초과. 계속 진행합니다.")

    # ── ngrok 터널 생성 ────────────────────────────────────────────
    print("[2/3] ngrok 터널 생성 중...")
    try:
        tunnel = ngrok.connect(port, "http")
    except Exception as e:
        print(f"[오류] ngrok 터널 생성 실패: {e}")
        proc.terminate()
        sys.exit(1)

    public_url = tunnel.public_url
    print(
        f"\n[3/3] 배포 완료!\n"
        f"  ✅ 공개 URL : {public_url}\n"
        f"  🏠 로컬 URL : http://localhost:{port}\n"
        f"\n  Ctrl+C 로 종료\n"
    )

    # ── 종료 대기 ──────────────────────────────────────────────────
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[종료] 터널과 서버를 닫습니다...")
        ngrok.disconnect(public_url)
        ngrok.kill()
        proc.terminate()
        print("종료 완료.")


if __name__ == "__main__":
    main()

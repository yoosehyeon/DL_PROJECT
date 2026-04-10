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
import subprocess
import sys
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


def main():
    parser = argparse.ArgumentParser(description="HybridPdM ngrok 배포")
    parser.add_argument("--token",  default=None, help="ngrok authtoken")
    parser.add_argument("--port",   type=int, default=8501, help="Streamlit 포트 (기본 8501)")
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

    # ── ngrok 설정 ─────────────────────────────────────────────────
    conf.get_default().region = args.region
    ngrok.set_auth_token(token)

    # ── Streamlit 서버 백그라운드 실행 ─────────────────────────────
    app_path = Path(__file__).parent / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ]
    print(f"[1/3] Streamlit 시작 중... (포트 {args.port})")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Streamlit 기동 대기
    time.sleep(4)
    if proc.poll() is not None:
        print("[오류] Streamlit 실행에 실패했습니다. app.py 경로와 의존성을 확인하세요.")
        sys.exit(1)

    # ── ngrok 터널 생성 ────────────────────────────────────────────
    print("[2/3] ngrok 터널 생성 중...")
    try:
        tunnel = ngrok.connect(args.port, "http")
    except Exception as e:
        print(f"[오류] ngrok 터널 생성 실패: {e}")
        proc.terminate()
        sys.exit(1)

    public_url = tunnel.public_url
    print(
        f"\n[3/3] 배포 완료!\n"
        f"  ✅ 공개 URL : {public_url}\n"
        f"  🏠 로컬 URL : http://localhost:{args.port}\n"
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

"""과적합 진단 — best_epoch 시점의 train/val + test 성능 종합 분석.

EarlyStopping 로직 보정:
  - 저장된 모델 = best_epoch 가중치 (마지막 epoch가 아님)
  - 따라서 train/val gap도 best_epoch 시점으로 비교
  - 절대 loss 값이 작을 때 gap% 과대평가되는 함정 보정 위해 test 성능 함께 검증

사용: python _diag_overfit.py [report_path1] [report_path2] ...
인자 없으면 artifacts/reports/ 의 최신 1건 자동 선택
"""
import json
import sys
from pathlib import Path


def diagnose(reports):
    all_results = {}
    for fp in reports:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            name = entry.get("name", "?")
            if entry.get("status") != "ok":
                continue
            td = entry.get("train", {})
            history = td.get("history", [])
            ev = entry.get("eval", {})
            if not history:
                # GBDT/CatBoost는 history 없음 → eval만 표시
                all_results[name] = {
                    "best_epoch": None, "stopped_at": None,
                    "history": [], "eval": ev,
                }
                continue
            all_results[name] = {
                "best_epoch": td.get("best_epoch"),
                "stopped_at": td.get("stopped_at"),
                "history": history,
                "eval": ev,
            }

    print("=" * 115)
    print(" 과적합 진단 — best_epoch 시점의 train/val + test 성능 검증")
    print("=" * 115)
    print()
    print(f"{'모델':24s} {'best_ep':>8s} {'tr@best':>10s} {'val@best':>10s} {'gap':>9s}  {'test 지표':<24s} 진단")
    print("-" * 115)

    for name, info in all_results.items():
        hist = info["history"]
        ev = info["eval"]

        # test 지표 문자열 + 양호 여부
        if "f1" in ev:
            test_str = f"F1={ev['f1']:.4f}"
            test_good = ev["f1"] > 0.6
        elif "test_f1" in ev:
            test_str = f"F1={ev['test_f1']:.4f}"
            test_good = ev["test_f1"] > 0.6
        elif "rmse" in ev:
            test_str = f"RMSE={ev['rmse']:.2f} R²={ev.get('r2', 0):.3f}"
            test_good = ev.get("r2", -1) > 0.7
        elif "accuracy" in ev:
            test_str = f"Acc={ev['accuracy']:.4f}"
            test_good = ev["accuracy"] > 0.9
        else:
            test_str = "?"
            test_good = False

        # history 없는 모델 (GBDT/CatBoost)
        if not hist:
            print(f"{name:24s} {'-':>8} {'-':>10} {'-':>10} {'-':>9}  {test_str:<24s} GBDT/CatBoost (자체 early-stop)")
            continue

        best_ep = info["best_epoch"]
        if best_ep is None or best_ep < 1:
            best_idx = -1
        else:
            best_idx = max(0, min(best_ep - 1, len(hist) - 1))
        h_best = hist[best_idx]
        tr_at_best = h_best.get("train_loss")
        va_at_best = h_best.get("val_loss")
        if tr_at_best is None or va_at_best is None:
            print(f"{name:24s} (history에 loss 없음)")
            continue
        gap_at_best_pct = (va_at_best - tr_at_best) / max(abs(tr_at_best), 1e-9) * 100
        abs_gap = abs(va_at_best - tr_at_best)

        # 종합 진단
        if va_at_best < tr_at_best * 1.1:
            verdict = "정상 — val≈train (일반화 우수)"
        elif gap_at_best_pct < 50:
            verdict = "정상 — gap 작음 (<50%)"
        elif gap_at_best_pct < 150:
            if test_good:
                verdict = "OK — gap 있으나 test 우수 → 실질 과적합 아님"
            else:
                verdict = "경계 — gap 신호 + test 약함, 검토 필요"
        else:
            if test_good and abs_gap < 0.5:
                verdict = "OK — 절대 loss 작아 gap% 과대평가 (test 우수)"
            elif test_good:
                verdict = "주의 — gap 큼 but test 우수, 모니터링 권장"
            else:
                verdict = "주의 — 강한 과적합 + test 약함"

        best_ep_str = str(best_ep) if best_ep else "?"
        print(f"{name:24s} {best_ep_str:>8} {tr_at_best:>10.5f} {va_at_best:>10.5f} "
              f"{gap_at_best_pct:>+8.1f}%  {test_str:<24s} {verdict}")

    print()
    print("=" * 115)
    print("참고:")
    print("  - EarlyStopping이 best_state를 복원하므로 저장된 모델 = best_epoch 가중치")
    print("  - 절대 loss가 작을 때 gap% 수치는 과대평가 → test 성능이 진짜 판정 기준")
    print("  - val_loss > train_loss는 정상 (val에 학습 신호 없음, 약간 차이는 자연스러움)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        reports = sys.argv[1:]
    else:
        report_dir = Path("artifacts/reports")
        reports = sorted(report_dir.glob("pipeline_report_*.json"),
                         key=lambda p: p.stat().st_mtime, reverse=True)[:1]
        reports = [str(p) for p in reports]
    print(f"분석 대상 리포트: {reports}\n")
    diagnose(reports)

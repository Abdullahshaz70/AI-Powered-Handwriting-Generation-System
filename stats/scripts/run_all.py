"""
run_all.py
==========
Master runner — executes all evaluation scripts in the correct order
and prints a final summary of all numbers.

Usage:
    python stats/scripts/run_all.py              # run everything
    python stats/scripts/run_all.py --skip-ocr  # skip the slow OCR step
    python stats/scripts/run_all.py --only 02   # run only one script
"""

import os, sys, argparse, subprocess, time, textwrap

_HERE        = os.path.dirname(os.path.abspath(__file__))
STATS_DIR    = os.path.dirname(_HERE)
PROJECT_ROOT = os.path.dirname(STATS_DIR)
OUT_DIR      = os.path.join(STATS_DIR, "outputs")

SCRIPTS = {
    "02":      os.path.join(_HERE, "eval_02_classifier.py"),
    "05":      os.path.join(_HERE, "eval_05_charnet.py"),
    "quality": os.path.join(_HERE, "eval_image_quality.py"),
    "ocr":     os.path.join(_HERE, "eval_ocr_benchmark.py"),
}

DESCRIPTIONS = {
    "02":      "Approach 02 MultiTaskCNN  — Classifier metrics (accuracy, F1, confusion, t-SNE)",
    "05":      "Approach 05 CharNet       — Bézier MSE, endpoint deviation",
    "quality": "Image Quality             — SSIM, sharpness, ink coverage, distinctiveness",
    "ocr":     "OCR Cross-Model Benchmark — EasyOCR accuracy/F1 across all generators",
}


def run_script(key, script_path):
    print(f"\n{'='*65}")
    print(f"  Running: {DESCRIPTIONS[key]}")
    print(f"{'='*65}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ⚠  Script exited with code {result.returncode}  ({elapsed:.1f}s)")
    else:
        print(f"\n  ✓  Done  ({elapsed:.1f}s)")
    return result.returncode == 0


def print_summary():
    """Read all summary CSVs and print a human-readable table."""
    import glob
    try:
        import pandas as pd
    except ImportError:
        return

    print(f"\n{'='*65}")
    print("  FINAL SUMMARY")
    print(f"{'='*65}\n")

    # Approach 02
    f = os.path.join(OUT_DIR, "approach_02", "summary.csv")
    if os.path.isfile(f):
        df = pd.read_csv(f)
        r  = df.iloc[0]
        print(f"Approach 02 — MultiTaskCNN Classifier")
        print(f"  Character Accuracy (Top-1): {r.get('char_acc_top1','?')}%")
        print(f"  Character Accuracy (Top-5): {r.get('char_acc_top5','?')}%")
        print(f"  Writer Accuracy:            {r.get('writer_acc','?')}%\n")

    # Approach 05
    f = os.path.join(OUT_DIR, "approach_05", "summary.csv")
    if os.path.isfile(f):
        df = pd.read_csv(f)
        r  = df.iloc[0]
        print(f"Approach 05 — CharNet (MLP Bézier Regressor)")
        print(f"  Overall Bézier MSE:    {r.get('overall_mse','?')}")
        print(f"  Endpoint Deviation:    {r.get('endpoint_dev','?')}\n")

    # Image quality
    f = os.path.join(OUT_DIR, "image_quality", "approach_summary.csv")
    if os.path.isfile(f):
        df = pd.read_csv(f)
        print("Image Quality Metrics (generators only):")
        cols = ["approach", "avg_ssim", "avg_sharpness", "avg_distinctiveness"]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False))
        print()

    # OCR
    f = os.path.join(OUT_DIR, "ocr", "ocr_summary.csv")
    if os.path.isfile(f):
        df = pd.read_csv(f)
        print("OCR Cross-Model Benchmark:")
        cols = ["approach", "accuracy", "macro_f1", "correct", "total"]
        cols = [c for c in cols if c in df.columns]
        # Format as percentages
        if "accuracy" in df.columns:
            df["accuracy"] = (df["accuracy"] * 100).round(1).astype(str) + "%"
        if "macro_f1" in df.columns:
            df["macro_f1"] = (df["macro_f1"] * 100).round(1).astype(str) + "%"
        print(df[cols].to_string(index=False))
        print()

    print(f"All outputs saved to: {OUT_DIR}")
    print("\nOutput structure:")
    for root, dirs, files in os.walk(OUT_DIR):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        level = root.replace(OUT_DIR, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = "  " * (level + 1)
        for f in sorted(files):
            print(f"{subindent}{f}")


def main():
    ap = argparse.ArgumentParser(
        description="Run all stats evaluation scripts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python stats/scripts/run_all.py
              python stats/scripts/run_all.py --skip-ocr
              python stats/scripts/run_all.py --only 02
              python stats/scripts/run_all.py --only quality
        """)
    )
    ap.add_argument("--skip-ocr", action="store_true",
                    help="Skip the OCR benchmark (slow, requires easyocr)")
    ap.add_argument("--only", choices=list(SCRIPTS.keys()), default=None,
                    help="Run only one script by key (02 / 05 / quality / ocr)")
    args = ap.parse_args()

    print(f"\nAI Handwriting Generation — Stats Runner")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Outputs   -> : {OUT_DIR}\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.only:
        run_script(args.only, SCRIPTS[args.only])
    else:
        run_order = ["02", "05", "quality"]
        if not args.skip_ocr:
            run_order.append("ocr")
        else:
            print("  (OCR benchmark skipped — use without --skip-ocr to enable)")

        results = {}
        for key in run_order:
            results[key] = run_script(key, SCRIPTS[key])

        print(f"\n{'='*65}")
        print("  SCRIPT RESULTS")
        print(f"{'='*65}")
        for key, ok in results.items():
            status = "✓ OK" if ok else "✗ FAILED"
            print(f"  [{status}] {DESCRIPTIONS[key]}")

    print_summary()


if __name__ == "__main__":
    main()

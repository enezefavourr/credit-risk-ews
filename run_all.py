"""Run the whole pipeline in order.

  python run_all.py            # uses data/raw if the Kaggle files are there, else data/sample
  python run_all.py --sample   # (re)generate the synthetic sample first
"""
import subprocess
import sys
import time

STEPS = [
    ("Credit models: scorecard vs RF vs XGBoost, SHAP, cut-offs", "src.credit.train"),
    ("Delinquency, roll-rates, vintages, early warning", "src.credit.dpd_ews"),
    ("Fraud & anomaly detection", "src.fraud.train_fraud"),
    ("Monitoring replay + health report", "src.monitoring.monitor"),
]

if __name__ == "__main__":
    if "--sample" in sys.argv:
        STEPS.insert(0, ("Generate synthetic sample data", "src.make_sample_data"))
    for title, mod in STEPS:
        print(f"\n=== {title} ===", flush=True)
        t = time.time()
        subprocess.run([sys.executable, "-m", mod], check=True)
        print(f"--- done in {time.time() - t:.0f}s")
    print("\nAll steps complete. Start the API with:  uvicorn api.main:app --reload")

"""Run the reproducible analysis stages from one portable command."""

from pathlib import Path
import argparse
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"

SCENARIO = [
    "scenario_alignment.py",
    "summarize_scenario_alignment.py",
    "scenario_conclusion_stability.py",
    "query_budget_frontier.py",
    "report_query_budget_frontier.py",
]
OPERATIONAL = [
    "operational_batch_validation.py",
    "batch_placebo_tail_response.py",
    "operational_stress_sweep.py",
    "direct_response_gain_validation.py",
    "dispatch_wave_sensitivity.py",
    "batch_anchor_sensitivity.py",
    "operational_final_audit.py",
    "reviewer_critique_audit.py",
    "finalize_operational_validation.py",
]
AUDIT = ["verify_release.py", "generate_manifest.py"]


def run(names):
    for name in names:
        print(f"\n>>> {name}", flush=True)
        subprocess.run([sys.executable, str(CODE / name)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["verify", "operational", "full"])
    args = parser.parse_args()
    if args.stage == "verify":
        run(["verify_release.py"])
    elif args.stage == "operational":
        run(OPERATIONAL + AUDIT)
    else:
        run(SCENARIO + OPERATIONAL + AUDIT)


if __name__ == "__main__":
    main()

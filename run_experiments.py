"""Run the reproducible cloud/MEC/QoS comparison and export CSV summaries."""
from __future__ import annotations
import argparse
import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.simulator import NetworkSimulation

SCENARIOS = (
    "cloud_baseline",
    "mec_qos",
    "mec_qos_latency_aware",
)

def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def maybe_plot(summary_rows, outdir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping charts. Install with: pip install matplotlib")
        return
    metrics = [
        ("mean_latency_s", "Mean completion latency (s)", "latency_comparison.png"),
        ("throughput_jobs_s", "Throughput (completed jobs/s)", "throughput_comparison.png"),
        ("drop_ratio", "Drop ratio", "drop_ratio_comparison.png"),
    ]
    for key, ylabel, filename in metrics:
        plt.figure(figsize=(9, 5))
        clients = sorted({int(r["clients"]) for r in summary_rows})
        for scenario in SCENARIOS:
            values = []
            for n in clients:
                match = next((r for r in summary_rows
                              if r["scenario"] == scenario and int(r["clients"]) == n), None)
                values.append(float(match[key]) if match else 0.0)
            plt.plot(clients, values, marker="o", label=scenario.replace("_", " "))
        plt.xlabel("Number of simulated clients")
        plt.ylabel(ylabel)
        plt.title(ylabel + " by scenario")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, filename), dpi=160)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", nargs="+", type=int, default=[10, 25, 50, 100])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--queue-limit", type=int, default=100)
    args = parser.parse_args()
    if any(n < 1 for n in args.clients) or args.rounds < 1 or args.queue_limit < 1:
        parser.error("clients, rounds, and queue-limit must be positive integers")

    outdir = os.path.join(ROOT, "results")
    os.makedirs(outdir, exist_ok=True)
    raw_rows = []
    for clients in args.clients:
        for scenario in SCENARIOS:
            sim = NetworkSimulation(scenario, clients, args.rounds, args.seed,
                                    queue_limit=args.queue_limit)
            for m in sim.run():
                raw_rows.append(m.__dict__)
            print(f"Completed scenario={scenario}, clients={clients}, rounds={args.rounds}")

    fields = list(raw_rows[0].keys()) if raw_rows else []
    write_csv(os.path.join(outdir, "experiment_results.csv"), raw_rows, fields)
    grouped = {}
    for row in raw_rows:
        grouped.setdefault((row["scenario"], row["clients"]), []).append(row)
    summary = []
    for (scenario, clients), rows in sorted(grouped.items()):
        summary.append({
            "scenario": scenario,
            "clients": clients,
            "rounds": len(rows),
            "mean_latency_s": statistics.mean(r["mean_latency_s"] for r in rows),
            "mean_p95_latency_s": statistics.mean(r["p95_latency_s"] for r in rows),
            "mean_queue_delay_s": statistics.mean(r["mean_queue_delay_s"] for r in rows),
            "throughput_jobs_s": statistics.mean(r["throughput_jobs_s"] for r in rows),
            "drop_ratio": statistics.mean(r["drop_ratio"] for r in rows),
            "mean_simulated_duration_s": statistics.mean(r["simulated_duration_s"] for r in rows),
        })
    summary_fields = list(summary[0].keys()) if summary else []
    write_csv(os.path.join(outdir, "summary.csv"), summary, summary_fields)
    maybe_plot(summary, outdir)
    print(f"\nSaved results to: {outdir}")
    print("IMPORTANT: These are outputs of the stated abstract simulation, not real-world network measurements.")

if __name__ == "__main__":
    main()

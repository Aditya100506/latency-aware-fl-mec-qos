# Latency-Aware QoS Optimization for Federated Learning in MEC-Enabled 5G/6G Networks

A reproducible Python simulation comparing:
1. **Cloud baseline** — clients send model-update traffic through a shared cloud path.
2. **MEC + QoS** — updates are routed to edge servers and scheduled with weighted deficit round robin (DRR).
3. **MEC + QoS + latency-aware allocation** — edge assignment and scheduling prioritize latency-sensitive updates while retaining throughput-oriented service.

> **Scope note:** This repository implements a controlled, event-driven network simulation for FL communication workloads. It does not implement a production 5G/6G stack or ns-3 radio/channel model. The network values are configurable simulation assumptions, not measurements from a real operator network. The script does not train a neural network on MNIST; MNIST is the intended benchmark for a future/companion FL model-training experiment. Do not describe the generated network metrics as MNIST model-accuracy results.

## Requirements
- Python 3.9+
- No third-party packages are required for the simulation.

## Run
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
python -m pip install --upgrade pip
python src/run_experiments.py
```

Optional configuration:
```bash
python src/run_experiments.py --clients 10 25 50 100 --rounds 20 --seed 42
```

Outputs are written to `results/`:
- `experiment_results.csv` — per-scenario/per-round metrics
- `summary.csv` — aggregated metrics by scenario and client count
- `latency_comparison.png` — mean latency comparison (requires matplotlib; otherwise skipped)
- `throughput_comparison.png` — throughput comparison (requires matplotlib; otherwise skipped)

## Scenarios and metrics
- Cloud baseline
- MEC + QoS (weighted DRR)
- MEC + QoS + latency-aware allocation
- Mean and 95th-percentile completion latency
- Throughput (completed update jobs / simulated second)
- Packet/job drop ratio
- Average queueing delay
- Total simulated communication time

## Method
Each FL round creates a configurable batch of client model-update jobs. Jobs have a simulated payload size and a QoS class. The discrete-event simulator models a shared or edge-specific service capacity, propagation delay, queueing, and finite queue capacity. The QoS scheduler uses weighted deficit round robin; the latency-aware scenario also assigns clients to edge servers based on estimated path latency and load. The simulator is deterministic for a fixed seed.

## Dataset
MNIST: http://yann.lecun.com/exdb/mnist/

The current script evaluates **network-side communication behavior**, not image classification. If the paper reports ML accuracy, training loss, or convergence, add and run a separate federated-learning training implementation and document its configuration before reporting those metrics.

## Reproducibility
Record the command, Python version, seed, and any changed parameters when reporting results. The supplied defaults are illustrative simulation assumptions. Run multiple seeds for uncertainty analysis before drawing general conclusions.

## Citation / code availability
After publishing this repository, replace the placeholder in the paper with the public GitHub URL. Do not use a placeholder or claim code availability until the repository is actually public.

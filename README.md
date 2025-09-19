## Multi‑Agent AML Detection (DIA_COMP4105)

This repository implements a cooperative, two‑agent reinforcement learning system for Anti‑Money Laundering (AML) detection over transaction streams. A base environment computes rewards from agent decisions and a gamified wrapper adds evolving difficulty. Training coordinates both agents end‑to‑end with logging and checkpointing.

### Repository structure

```
agents/
  base_agent.py                 # Epsilon‑greedy Q‑learning/SARSA base
  transaction_monitoring_agent.py  # TMA: per‑transaction features
  network_analysis_agent.py        # NAA: sliding‑window network features
envs/
  standardenv.py                # Core laundering environment (rewards, step)
  Gamified.py                   # Difficulty scheduling, dynamic bonuses/penalties
root_dir/
  main.py                       # Entry point to train both agents
  train.py                      # Orchestration loop, logging, checkpoints
  logs/                         # TensorBoard event files
  checkpoints/                  # Saved Q‑tables and metadata
exp/
  random_search.py              # Simple hyperparameter search driver
utils/
  training_utils.py             # Checkpoint manager, TensorBoard writer
processed_amlsim_final (1).csv  # Example dataset (local copy)
```

### How the agents connect and work together

- **Base policy and learning**
  - Both agents subclass `agents.base_agent.BaseAgent` which maintains a `q_table` and supports two algorithms: **Q‑learning** and **SARSA** with epsilon‑greedy action selection.
  - Actions are binary `{0, 1}` and interpreted as the confidence/decision to flag a transaction/window.

- **Transaction Monitoring Agent (TMA)** — `agents.transaction_monitoring_agent.TransactionMonitoringAgent`
  - Operates on a single transaction at the current step.
  - Extracts a state vector from fields like `AmountReceived`, `AmountPaid`, and one‑hot prefixed features `ReceivingCurrency_*`, `PaymentCurrency_*`, `PaymentFormat_*`.

- **Network Analysis Agent (NAA)** — `agents.network_analysis_agent.NetworkAnalysisAgent`
  - Operates on a sliding window ending at the current step.
  - Aggregates precomputed network/graph features across the window, e.g., `from_degree`, `to_degree`, `sender_clustering`, `receiver_clustering`, `from_betweenness`, `to_betweenness` (mean‑pooled).

- **Decision fusion and learning loop** (`root_dir/train.py`)
  - At each timestep: TMA chooses `action1` on current transaction; NAA chooses `action2` on the current window.
  - The environment consumes these as inputs and computes a reward; training fetches the next state and both agents update their `q_table` using either Q‑learning or SARSA (with next actions when SARSA).
  - Final decision used for metrics is `final_decision = max(decision1, decision2)` (flag if either flags). Disagreements are tracked and influence reward via confidence.
  - Epsilon decays over episodes; learning rates can decay via a simple scheduler.

### How the environment works

- **Core environment** — `envs.standardenv.LaunderingEnv`
  - Holds the transaction dataframe, current step, sliding window, and cumulative/episode‑level stats.
  - `step(...)` accepts either the agents or direct numeric inputs for TMA/NAA; in training we pass the numeric actions and the environment invokes `choose_action(...)` for logging consistency.
  - Reward shaping (`_compute_reward`):
    - Rewards agreement and correctness (true positives/negatives), caps false positive penalties, penalizes false negatives more strongly, and provides a teamwork bonus for correct joint flags.
    - On disagreement, picks the more confident decision and rewards/penalizes accordingly; maintains streak bonuses for sustained true negatives; tracks flagged accounts/clusters to shape future rewards and discourage false flags.
  - Provides `episode_summary()` and `summary()` computing Precision/Recall/F1 and confusion matrix.

- **Gamified wrapper** — `envs.Gamified.GamifiedEnv`
  - Adds time‑based difficulty scheduling and evolving fraud patterns.
  - Decreases reward after a warmup (harder environment), and adds early bonuses for quickly catching large transactions (`AmountReceived` above a learned percentile threshold).
  - Maintains `time_step`, `difficulty_factor`, and evolving `active_fraud_patterns`.

### Data expectations

Your dataframe (CSV → `pandas.read_csv`) should include at minimum:

- Supervision/meta: `IsLaundering` (0/1), `PatternType`, `FromBankAccount`, `ClusterID`
- TMA features: `AmountReceived`, `AmountPaid`, and one‑hot prefixed keys: `ReceivingCurrency_*`, `PaymentCurrency_*`, `PaymentFormat_*`
- NAA features (precomputed per transaction): `from_degree`, `to_degree`, `sender_clustering`, `receiver_clustering`, `from_betweenness`, `to_betweenness`

The repository includes an example CSV `processed_amlsim_final (1).csv`. You can supply your own as long as it matches the feature expectations above.

### Dependencies

- Python 3.10+
- Packages:
  - `numpy`, `pandas`, `matplotlib`
  - `torch` (for `torch.utils.tensorboard` writer)
  - `tensorboard` (to visualize logs)

Install (Windows, Command Prompt):

```bat
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install numpy pandas matplotlib torch tensorboard
```

### Configure the data path

- The entrypoint `root_dir/main.py` reads from a hardcoded `csv_path`. Options:
  - Replace `csv_path` with your dataset location, or
  - Move/copy your CSV to the repo and point to it, e.g. `processed_amlsim_final (1).csv`.

Example minimal edit in `root_dir/main.py`:

```python
csv_path = r"E:\DIA_COMP4105\processed_amlsim_final (1).csv"
```

### Run training

From the repository root:

```bat
.venv\Scripts\activate
python root_dir\main.py
```

Artifacts:
- Checkpoints (Q‑tables and metadata): `root_dir/checkpoints/`
- TensorBoard logs: `root_dir/logs/`

Launch TensorBoard:

```bat
tensorboard --logdir root_dir\logs --port 6006
```

### Recreate this setup from scratch

1) Clone/copy this repo to your machine.
2) Create a Python 3.10+ virtual environment and install the dependencies above.
3) Prepare your dataset CSV with the feature columns listed in “Data expectations”.
4) Edit `root_dir/main.py` to set `csv_path` to your dataset location.
5) Run `python root_dir/main.py`. Monitor progress in the console and via TensorBoard.

Optional: run a quick hyperparameter search

```bat
python exp\random_search.py
```

The script constructs new agents per trial with random `alpha`, `epsilon`, and `gamma`, runs short training, and writes `random_search_results.csv` in the project root.

### Key design notes

- Agents act on complementary views: TMA (local transaction), NAA (temporal/window network summary). Fusion favors recall by flagging when either agent flags.
- Reward shaping encourages agreement when correct, measured restraint on false positives, and consistency streaks; it also leverages historical flags of accounts/clusters.
- The gamified layer simulates non‑stationarity to stress generalization and timely detection.

### Troubleshooting

- If you see `ModuleNotFoundError: torch`, install PyTorch appropriate for your system: see the official site for the correct wheel, then re‑run `pip install torch`.
- If your CSV lacks expected columns, adapt `TransactionMonitoringAgent.extract_state` and/or `NetworkAnalysisAgent.extract_state` to your schema.
- If plots don’t show, ensure a GUI backend is available or rely on the saved `reward_per_episode.png` and TensorBoard.



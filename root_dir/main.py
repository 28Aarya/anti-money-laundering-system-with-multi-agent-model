# main.py
import sys
import os
import pandas as pd
import cProfile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from envs.Gamified import GamifiedEnv
from agents.transaction_monitoring_agent import TransactionMonitoringAgent
from agents.network_analysis_agent import NetworkAnalysisAgent
from train import train


csv_path = r"E:\AML_DIA\data\processed_amlsim_final (1).csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at: {csv_path}")

df = pd.read_csv(csv_path)

# Initialize environment
env = GamifiedEnv(df)


ALPHA =  0.053913
GAMMA = 0.649693
EPSILON = 0.05
epsilon_decay = 0.995  
min_epsilon = 0.01
ALGORITHMT_1= "Q-learning"
ALGORITHMT_2= "SARSA"

agent1 = TransactionMonitoringAgent(alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, algorithm=ALGORITHMT_2)
agent2 = NetworkAnalysisAgent(alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON, algorithm=ALGORITHMT_2)

# Begin training
# --- Training hyperparameters ---
NUM_EPISODES = 30
EARLY_STOP_PATIENCE = 5
LR_DECAY = 0.99
LR_DECAY_FREQ = 7
VALIDATION_FREQ = 7
QTABLE_PRINT_FREQ = 7
EVAL_INTERVAL = 6

# --- Run training ---
train(
    env=env,
    agent1=agent1,
    agent2=agent2,
    num_episodes=NUM_EPISODES,
    log_dir="logs",
    checkpoint_dir="checkpoints",
    eval_interval=EVAL_INTERVAL,
    early_stop_patience=EARLY_STOP_PATIENCE,
    lr_decay=LR_DECAY,
    lr_decay_freq=LR_DECAY_FREQ,
    epsilon_decay=epsilon_decay,
    min_epsilon=min_epsilon,
    plot_results=True,
    validation_freq=VALIDATION_FREQ,
    qtable_print_freq=QTABLE_PRINT_FREQ
)

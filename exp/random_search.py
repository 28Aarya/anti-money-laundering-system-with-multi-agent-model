import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import pandas as pd
from utils.training_utils import CheckpointManager, setup_tensorboard
from envs.standardenv import LaunderingEnv
from agents.transaction_monitoring_agent import TransactionMonitoringAgent
from agents.network_analysis_agent import NetworkAnalysisAgent
from agents.base_agent import BaseAgent
from envs.Gamified import GamifiedEnv
from root_dir.train import train

def random_search(
    df,
    n_trials=6,
    alpha_range=(0.01, 0.9),
    epsilon_range=(0.01, 0.8),
    gamma_range=(0.2, 0.99),
    num_episodes=16,
    early_stop_patience=3,
    **train_kwargs
):
    results = []
    for trial in range(n_trials):
        alpha = random.uniform(*alpha_range)
        epsilon = random.uniform(*epsilon_range)
        gamma = random.uniform(*gamma_range)
        print(f"\n=== Trial {trial+1}/{n_trials}: alpha={alpha:.4f}, epsilon={epsilon:.4f}, gamma={gamma:.4f} ===")
        env = GamifiedEnv(df)
        agent1 = TransactionMonitoringAgent(alpha=alpha, gamma=gamma, epsilon=epsilon, algorithm=train_kwargs.get('algorithm', 'SARSA'))
        agent2 = NetworkAnalysisAgent(alpha=alpha, gamma=gamma, epsilon=epsilon, algorithm=train_kwargs.get('algorithm', 'SARSA'))
        best_reward = train(
            env=env,
            agent1=agent1,
            agent2=agent2,
            num_episodes=num_episodes,
            early_stop_patience=early_stop_patience,
            plot_results=False,  # Don't plot for each trial
            **train_kwargs
        )
        results.append({'alpha': alpha, 'epsilon': epsilon, 'gamma': gamma, 'best_reward': best_reward})
    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv('random_search_results.csv', index=False)
    print("\nRandom search complete. Results saved to random_search_results.csv.")
    print(df_results.sort_values('best_reward', ascending=False).head())

if __name__ == "__main__":
    csv_path = r"E:/AML_DIA/data/processed_amlsim_final (1).csv"
    df = pd.read_csv(csv_path)
    random_search(df, n_trials=6, num_episodes=16, early_stop_patience=3)  # Adjust as needed

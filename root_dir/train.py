import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.training_utils import CheckpointManager, setup_tensorboard
from envs.standardenv import LaunderingEnv
from agents.transaction_monitoring_agent import TransactionMonitoringAgent
from agents.network_analysis_agent import NetworkAnalysisAgent
from agents.base_agent import BaseAgent

import matplotlib.pyplot as plt

def train(
    env, agent1, agent2,
    num_episodes=100,
    log_dir="logs",
    checkpoint_dir="checkpoints",
    eval_fn=None,
    eval_interval=100,
    early_stop_patience=20,
    lr_decay=0.99,
    lr_decay_freq=20,
    epsilon_decay=0.995,
    min_epsilon=0.05,
    validation_freq=20,
    qtable_print_freq=10,
    plot_results=True
):
    """
    Train RL agents in the laundering environment.

    Args:
        env: RL environment
        agent1: TransactionMonitoringAgent
        agent2: NetworkAnalysisAgent
        num_episodes (int): Number of training episodes
        log_dir (str): Directory for tensorboard logs
        checkpoint_dir (str): Directory for saving checkpoints
        eval_fn (callable): Optional evaluation function
        eval_interval (int): Save/evaluate every N episodes
        early_stop_patience (int): Stop if reward doesn't improve this many episodes
        lr_decay (float): Alpha decay factor (learning rate scheduler)
        lr_decay_freq (int): Frequency (episodes) to decay alpha
        validation_freq (int): Run validation (epsilon=0) every N episodes
        qtable_print_freq (int): Print Q-table stats every N episodes
    """

    writer = setup_tensorboard(log_dir)
    checkpoint_mgr = CheckpointManager(checkpoint_dir)
    best_reward = float('-inf')
    best_reward_episode = 0
    no_improve_count = 0
    initial_alpha1 = agent1.alpha if hasattr(agent1, 'alpha') else None
    initial_alpha2 = agent2.alpha if hasattr(agent2, 'alpha') else None

    episode_rewards = []
    episode_f1s = []
    episode_precisions = []
    episode_recalls = []
    episode_accuracies = []

    # Gather hyperparameters for checkpoint saving
    hyperparams = {
        'num_episodes': num_episodes,
        'early_stop_patience': early_stop_patience,
        'lr_decay': lr_decay,
        'lr_decay_freq': lr_decay_freq,
        'validation_freq': validation_freq,
        'qtable_print_freq': qtable_print_freq,
        'log_dir': log_dir,
        'checkpoint_dir': checkpoint_dir,
        'eval_interval': eval_interval,
        # Agent 1 hyperparameters
        'agent1_alpha': getattr(agent1, 'alpha', None),
        'agent1_gamma': getattr(agent1, 'gamma', None),
        'agent1_epsilon': getattr(agent1, 'epsilon', None),
        'agent1_algorithm': getattr(agent1, 'algorithm', None),
        # Agent 2 hyperparameters
        'agent2_alpha': getattr(agent2, 'alpha', None),
        'agent2_gamma': getattr(agent2, 'gamma', None),
        'agent2_epsilon': getattr(agent2, 'epsilon', None),
        'agent2_algorithm': getattr(agent2, 'algorithm', None),
    }

    for episode in range(num_episodes):
        txn_input = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            # For TMA: single transaction
            tma_features = txn_input
            # For NAA: window
            naa_window = env.get_current_window()

            action1 = agent1.choose_action(tma_features)
            action2 = agent2.choose_action(naa_window)

            # Step the environment
            next_state, reward, done, info = env.step(
                tma_agent=agent1,
                naa_agent=agent2,
                tma_input=action1,
                naa_input=action2
            )


            # Prepare next state for TMA and NAA
            txn_input = next_state
            next_tma_features = txn_input if not done else None
            next_naa_window = env.get_current_window() if not done else None
            next_action1 = agent1.choose_action(next_tma_features) if next_tma_features else None
            next_action2 = agent2.choose_action(next_naa_window) if next_naa_window else None

            agent1.update(tma_features, action1, reward, next_tma_features, next_action1)
            agent2.update(naa_window, action2, reward, next_naa_window, next_action2)

            total_reward += reward
            step += 1

        writer.add_scalar('Train/TotalReward', total_reward, episode)

        episode_summary = env.episode_summary()
        episode_rewards.append(episode_summary['Reward'])
        episode_f1s.append(episode_summary['F1'])
        episode_precisions.append(episode_summary['Precision'])
        episode_recalls.append(episode_summary['Recall'])
        # Calculate accuracy for this episode
        TP = episode_summary['ConfusionMatrix']['TP']
        TN = episode_summary['ConfusionMatrix']['TN']
        FP = episode_summary['ConfusionMatrix']['FP']
        FN = episode_summary['ConfusionMatrix']['FN']
        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-5)
        episode_accuracies.append(accuracy)

        print(f"\nEpisode {episode+1} Summary:")
        print(f"  Total Reward: {episode_summary['Reward']}")
        print(f"  True Positives:   {TP}")
        print(f"  False Positives:  {FP}")
        print(f"  True Negatives:   {TN}")
        print(f"  False Negatives:  {FN}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1: {episode_summary['F1']}, Precision: {episode_summary['Precision']}, Recall: {episode_summary['Recall']}")

        # Early stopping: check for reward stagnation
        if total_reward > best_reward:
            best_reward = total_reward
            best_reward_episode = episode
            checkpoint_mgr.save(agent1, agent2, episode+1, total_reward, extra_info={
                "best": True,
                "hyperparameters": hyperparams
            })
            no_improve_count = 0
        else:
            no_improve_count += 1
        if early_stop_patience and no_improve_count >= early_stop_patience:
            print(f"Early stopping at episode {episode+1} (no reward improvement for {early_stop_patience} episodes)")
            break

        # Learning rate decay (LR scheduler)
        if lr_decay and lr_decay_freq and (episode+1) % lr_decay_freq == 0:
            if hasattr(agent1, 'alpha') and initial_alpha1 is not None:
                agent1.alpha *= lr_decay
                print(f"  [LR Scheduler] Agent1 alpha decayed to {agent1.alpha:.5f}")
            if hasattr(agent2, 'alpha') and initial_alpha2 is not None:
                agent2.alpha *= lr_decay
                print(f"  [LR Scheduler] Agent2 alpha decayed to {agent2.alpha:.5f}")
                
        # Epsilon decay (exploration rate) for training
        if hasattr(agent1, 'epsilon'):
            agent1.epsilon = max(agent1.epsilon * epsilon_decay, min_epsilon)
        if hasattr(agent2, 'epsilon'):
            agent2.epsilon = max(agent2.epsilon * epsilon_decay, min_epsilon)

        # Validation episode (exploration-free)
        if validation_freq and (episode+1) % validation_freq == 0:
            print("\n[Validation] Running exploration-free episode...")
            old_eps1, old_eps2 = getattr(agent1, 'epsilon', None), getattr(agent2, 'epsilon', None)
            if hasattr(agent1, 'epsilon'):
                agent1.epsilon = 0.0
            if hasattr(agent2, 'epsilon'):
                agent2.epsilon = 0.0
            val_txn_input = env.reset()
            val_done = False
            val_total_reward = 0
            while not val_done:
                tma_features = val_txn_input
                naa_window = env.get_current_window()
                val_action1 = agent1.choose_action(tma_features)
                val_action2 = agent2.choose_action(naa_window)
                val_next_state, val_reward, val_done, val_info = env.step(
                    tma_agent=agent1, naa_agent=agent2, tma_input=val_action1, naa_input=val_action2)
                val_txn_input = val_next_state
                val_total_reward += val_reward
            val_summary = env.episode_summary()
            print(f"[Validation] Reward: {val_total_reward}, F1: {val_summary['F1']}, Precision: {val_summary['Precision']}, Recall: {val_summary['Recall']}")
            if hasattr(agent1, 'epsilon') and old_eps1 is not None:
                agent1.epsilon = old_eps1
            if hasattr(agent2, 'epsilon') and old_eps2 is not None:
                agent2.epsilon = old_eps2

        # Q-table stats print (every N episodes)
        if qtable_print_freq and episode % qtable_print_freq == 0:
            tma_states = len(agent1.q_table)
            naa_states = len(agent2.q_table)
            tma_zero_q = sum(1 for q_values in agent1.q_table.values() if all(q == 0 for q in q_values))
            naa_zero_q = sum(1 for q_values in agent2.q_table.values() if all(q == 0 for q in q_values))
            print(f"  TMA Q-table: {tma_states} states, {tma_zero_q} with all-zero Q-values")
            print(f"  NAA Q-table: {naa_states} states, {naa_zero_q} with all-zero Q-values")

        print("  ---\n")
        print(env.episode_summary())

    # --- Plot results at end of training ---
    if plot_results:
        import numpy as np
        episodes = list(range(1, len(episode_rewards)+1))
        # Moving average reward
        def moving_average(x, w=10):
            x = np.array(x)
            if len(x) < w:
                return x
            return np.convolve(x, np.ones(w)/w, mode='valid')
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, episode_rewards, label='Reward')
        plt.plot(episodes[len(episodes)-len(moving_average(episode_rewards)):], moving_average(episode_rewards), label='Moving Avg Reward (window=3)', linestyle='--')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward per Episode')
        plt.legend()
        plt.tight_layout()
        plt.savefig('reward_per_episode.png')
        plt.show()

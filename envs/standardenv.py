class LaunderingEnv:
    def _init_episode_stats(self):
        # Per-episode confusion matrix
        return {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'disagreements': 0, 'caught_sus_labels': 0, 'total_steps': 0}
    
    def get_current_window(self):
        start = max(0, self.current_step - self.window_size)
        window = self.data.iloc[start:self.current_step + 1].to_dict(orient='records')
        for txn in window:
            txn.pop('IsLaundering', None)
        return window

    def __init__(self, transaction_data, window_size=20, reward_mode='confidence'):
        # Load precomputed graph features and initialize variables
        self.data = transaction_data
        self.total_steps = len(self.data)
        self.window_size = window_size
        self.reward_mode = reward_mode
        self.tn_streak = 0
        self.episode_rewards = []

        self.stats = {
            'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'disagreements': 0, 
            'total_steps': 0, 'caught_sus_labels': 0, 
            'Disagreement_TMA_Wins': 0, 'Disagreement_NAA_Wins': 0  
        }
        self.episode_stats = self._init_episode_stats()
        self.current_step = 0
        self.cumulative_reward = 0.0

        # Tracking flagged accounts and clusters
        self.flagged_accounts = set()  
        self.flagged_clusters = set()  

    def reset(self):
        self.current_step = 0
        self.tn_streak = 0
        self.cumulative_reward = 0.0
        self.episode_stats = self._init_episode_stats()  
        self.episode_rewards = []
        txn = self.data.iloc[self.current_step]
        txn_dict = txn.to_dict()
        txn_dict.pop('IsLaundering', None)  
        return txn_dict

    
    def _compute_reward(self, txn, label, decision1, decision2, tma_confidence, naa_confidence):
        reward = 0.0

        # --- Agreement-based rewards ---
        if decision1 == decision2:
            final_decision = decision1
            if final_decision == 1 and label == 1:
                reward += 3.5
            elif final_decision == 0 and label == 0:
                reward += 4.0
            elif final_decision == 1 and label == 0:
                reward -= 1.5  # Cap FP penalty
            elif final_decision == 0 and label == 1:
                reward -= 2.0

            # Teamwork bonus
            if final_decision == 1 and label == 1:
                if reward < 6.0:
                    reward = 6.0
        else:
            self.episode_stats['disagreements'] += 1
            if tma_confidence >= naa_confidence:
                final_decision = decision1
                chosen_conf = tma_confidence
            else:
                final_decision = decision2
                chosen_conf = naa_confidence

            if final_decision == label:
                reward += 2.0 if chosen_conf > 0.5 else 0.2
            else:
                reward -= 1.0 if chosen_conf > 0.5 else 0.3
                
        if final_decision == 0 and label == 0:
            self.tn_streak += 1
            if self.tn_streak >= 5:
                reward += 3.0
            else:
                self.tn_streak = 0


        pattern_type = txn.get('PatternType', 'Not Laundering')
        if label == 0 and pattern_type != 'Not Laundering' and decision1 == decision2 == 1:
            reward += 4.0    
            self.stats['caught_sus_labels'] += 1

        from_acct = txn['FromBankAccount']
        cluster = txn['ClusterID']
        if decision1 == decision2 == 1 and label == 1:
            if from_acct not in self.flagged_accounts:
                reward += 0.2
                self.flagged_accounts.add(from_acct)
            if cluster not in self.flagged_clusters:
                reward += 0.2
                self.flagged_clusters.add(cluster)
            if from_acct in self.flagged_accounts and cluster in self.flagged_clusters:
                reward += 1.5

        # --- False flag penalties ---
        if label == 0:
            if from_acct in self.flagged_accounts:
                reward -= 0.25
            if cluster in self.flagged_clusters:
                reward -= 0.25

        self.episode_rewards.append(reward)
        return reward
    
    def step(self, tma_agent=None, naa_agent=None, tma_input=None, naa_input=None):
        if self.current_step >= self.total_steps:
            return None, 0.0, True, {"reason": "Dataset exhausted"}

        txn = self.data.iloc[self.current_step]
        label = txn['IsLaundering']
        txn_dict = txn.drop(['IsLaundering', 'PatternType']).to_dict()

        # Create input window for NAA 
        start = max(0, self.current_step - self.window_size)
        window = self.data.iloc[start:self.current_step + 1].to_dict(orient='records')

        # TMA Decision 
        if tma_agent is not None and hasattr(tma_agent, "choose_action"):
            tma_confidence = tma_agent.choose_action(txn_dict)
        elif tma_input is not None:
            tma_confidence = tma_input
        else:
            raise ValueError("Either tma_agent or tma_input must be provided.")

        # NAA Decision 
        if naa_agent is not None and hasattr(naa_agent, "choose_action"):
            naa_confidence = naa_agent.choose_action(window)
        elif naa_input is not None:
            naa_confidence = naa_input
        else:
            raise ValueError("Either naa_agent or naa_input must be provided.")


        decision1 = int(tma_confidence >= 0.5)
        decision2 = int(naa_confidence >= 0.5)
        final_decision = max(decision1, decision2)

        if decision1 == 1:
            self.flagged_accounts.add(txn['FromBankAccount'])
        if decision2 == 1:
            self.flagged_clusters.add(txn['ClusterID'])

        reward = self._compute_reward(txn, label, decision1, decision2, tma_confidence, naa_confidence)
        info = {
            "disagreement": decision1 != decision2,
            "label": int(label),
            "final_decision": final_decision,
        }


        
        # final_decision: 1 = flagged as laundering, 0 = not flagged
        # label: 1 = laundering, 0 = not laundering
        if final_decision == 1 and label == 1:
            self.stats['TP'] += 1  # True Positive
            self.episode_stats['TP'] += 1
        elif final_decision == 0 and label == 0:
            self.stats['TN'] += 1  # True Negative
            self.episode_stats['TN'] += 1
        elif final_decision == 1 and label == 0:
            self.stats['FP'] += 1  # False Positive
            self.episode_stats['FP'] += 1
        elif final_decision == 0 and label == 1:
            self.stats['FN'] += 1  # False Negative
            self.episode_stats['FN'] += 1

        self.cumulative_reward += reward
        self.current_step += 1
        self.stats['total_steps'] += 1
        self.episode_stats['total_steps'] += 1

        done = self.current_step >= self.total_steps
        return txn_dict, reward, done, info
    

    def summary(self):
        # Cumulative stats for the whole run
        precision = self.stats['TP'] / (self.stats['TP'] + self.stats['FP'] + 1e-5)
        recall = self.stats['TP'] / (self.stats['TP'] + self.stats['FN'] + 1e-5)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-5)
        return {
            "Reward": self.cumulative_reward,
            "Steps": self.stats['total_steps'],
            "Disagreements": self.stats['disagreements'],
            "Caught Sus Labels": self.stats['caught_sus_labels'],
            "ConfusionMatrix": {
                "TP": self.stats['TP'], "TN": self.stats['TN'],
                "FP": self.stats['FP'], "FN": self.stats['FN']
            },
            "F1": round(f1, 4), "Precision": round(precision, 4), "Recall": round(recall, 4)
        }

    def episode_summary(self):
        # Per-episode summary and reset
        precision = self.episode_stats['TP'] / (self.episode_stats['TP'] + self.episode_stats['FP'] + 1e-5)
        recall = self.episode_stats['TP'] / (self.episode_stats['TP'] + self.episode_stats['FN'] + 1e-5)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-5)
        summary = {
            "Reward": self.cumulative_reward,
            "Steps": self.episode_stats['total_steps'],
            "Disagreements": self.episode_stats['disagreements'],
            "Caught Sus Labels": self.episode_stats['caught_sus_labels'],
            "ConfusionMatrix": {
                "TP": self.episode_stats['TP'], "TN": self.episode_stats['TN'],
                "FP": self.episode_stats['FP'], "FN": self.episode_stats['FN']
            },
            "F1": round(f1, 4), "Precision": round(precision, 4), "Recall": round(recall, 4)
        }
        return summary

import os
import pickle
import datetime

class CheckpointManager:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, agent1, agent2, episode, total_reward, extra_info=None):
        checkpoint = {
            "agent1_q_table": agent1.q_table,
            "agent2_q_table": agent2.q_table,
            "episode": episode,
            "total_reward": total_reward,
            "extra_info": extra_info
        }
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_ep{episode}_{timestamp}.pkl"
        path = os.path.join(self.checkpoint_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)
        return path

    def load(self, path, agent1, agent2):
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        agent1.q_table = checkpoint["agent1_q_table"]
        agent2.q_table = checkpoint["agent2_q_table"]
        return checkpoint

def setup_tensorboard(log_dir="logs"):
    from torch.utils.tensorboard import SummaryWriter
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)

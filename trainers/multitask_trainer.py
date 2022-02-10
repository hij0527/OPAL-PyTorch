import torch

from memory.replay_buffer import ReplayBuffer
from trainers.base_trainer import BaseTrainer


class MultitaskTrainer(BaseTrainer):
    def __init__(
        self,
        logger,
        phase,
        tag='',
        print_freq=100,
        log_freq=100,
        save_freq=10,
    ):
        super().__init__(
            logger,
            phase,
            tag,
            print_freq,
            log_freq,
            save_freq,
        )

    def train(
        self,
        model,
        env,
        device,
        longstep_len,
        train_episodes,
        update_freq,
        updates_per_step,
        batch_size,
        eval_freq,
        eval_episodes,
    ):
        pass

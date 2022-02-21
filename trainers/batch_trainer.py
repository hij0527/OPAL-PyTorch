from torch.utils.data import DataLoader

from memory.replay_buffer import FixedReplayBuffer
from trainers.base_trainer import BaseTrainer, BaseRLTrainer, BaseHRLTrainer


class BatchTrainer(BaseTrainer):
    def train(
        self,
        buffer,
        num_epochs=100,
        batch_size=256,
        num_workers=6,
        batch_preproc={},
        train_param_schedule={},
        **train_params,
    ):
        data_loader = DataLoader(buffer, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        schedule_step_size = 1. / (num_epochs - 1) if num_epochs > 1 else 0.
        train_step = 0

        self.eval_and_log(0, buffer=buffer, batch_size=batch_size, num_workers=num_workers,
                          batch_preproc=batch_preproc)

        for epoch in range(1, num_epochs + 1):
            print('Epoch {}'.format(epoch))
            self.reset_timer()
            epoch_loss, total_num_data = 0., 0

            for k, (v_init, v_max) in train_param_schedule.items():
                train_params[k] = v_init + (v_max - v_init) * (epoch - 1) * schedule_step_size

            for i, batch in enumerate(data_loader):
                samples = self.preproc_batch(batch, batch_preproc)
                num_data = next(iter(samples.values())).shape[0]
                simple_buffer = FixedReplayBuffer(samples)
                loss, sublosses = self.model.update(simple_buffer, batch_size, 1, **train_params)
                epoch_loss += loss * num_data
                total_num_data += num_data
                train_step += 1

                self.print_losses(train_step, loss, sublosses)
                self.log_losses(train_step, loss, sublosses)

            epoch_loss /= total_num_data
            self.print_losses(epoch, epoch_loss, force=True, is_epoch=True)
            self.log_losses(epoch, epoch_loss, force=True, is_epoch=True)
            self.eval_and_log(epoch, buffer=buffer, batch_size=batch_size, num_workers=num_workers,
                              batch_preproc=batch_preproc)
            self.save_model(epoch)

        self.save_model(epoch=num_epochs, force=True)

    def eval_model(self, buffer, batch_size, num_workers, batch_preproc):
        data_loader = DataLoader(buffer, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        total_num_data = 0
        sum_results = {}
        for i, batch in enumerate(data_loader):
            samples = self.preproc_batch(batch, batch_preproc)
            num_data = next(iter(samples.values())).shape[0]
            results = self.model.evaluate(samples)
            if not sum_results:
                sum_results = {k: 0 for k in results}
            for k in sum_results:
                sum_results[k] += results[k] * num_data
            total_num_data += num_data

        if not total_num_data:
            return {}
        return {k: v / total_num_data for k, v in sum_results.items()}

    def preproc_batch(self, batch, batch_preproc):
        return {k: (batch_preproc[k](v) if k in batch_preproc else v).to(self.device)
            for k, v in batch.items()}


class BatchRLTrainer(BaseRLTrainer, BatchTrainer):
    pass


class BatchHRLTrainer(BatchRLTrainer, BaseHRLTrainer):
    def preproc_batch(self, batch, batch_preproc):
        samples = super().preproc_batch(batch, batch_preproc)
        samples['actions'] = samples['latents']
        return samples

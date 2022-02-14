from torch.utils.data import DataLoader

from trainers.base_trainer import BaseTrainer


class BatchTrainer(BaseTrainer):
    def __init__(
        self,
        logger,
        tag='',
        print_freq=100,
        log_freq=100,
        save_freq=10,
    ):
        super().__init__(
            logger,
            tag,
            print_freq,
            log_freq,
            save_freq,
        )

    def train(
        self,
        model,
        buffer,
        data_keys,
        device,
        num_epochs=100,
        batch_size=256,
        num_workers=6,
        param_schedule={},
        batch_preproc={},
        **train_kwargs
    ):
        data_loader = DataLoader(buffer, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        fn_preproc = {k: (lambda x: x) for k in data_keys}
        fn_preproc.update(batch_preproc)
        schedule_step_size = 1. / (num_epochs - 1) if num_epochs > 1 else 0.
        train_step = 0

        for epoch in range(1, num_epochs + 1):
            print('Epoch {}'.format(epoch))
            self.reset_timer()
            epoch_loss, num_data = 0., 0

            for k, (v_init, v_max) in param_schedule.items():
                train_kwargs[k] = v_init + (v_max - v_init) * (epoch - 1) * schedule_step_size

            for i, batch in enumerate(data_loader):
                batch_items = [fn_preproc[k](batch[k]).to(device) for k in data_keys]
                loss, sublosses = model.update(batch_items, **train_kwargs)
                epoch_loss += loss * batch_items[0].shape[0]
                num_data += batch_items[0].shape[0]
                train_step += 1

                self.print_losses(train_step, loss, sublosses)
                self.log_losses(train_step, loss, sublosses)

            epoch_loss /= num_data
            self.print_losses(epoch, epoch_loss, is_epoch=True)
            self.log_losses(epoch, epoch_loss, is_epoch=True)
            self.save_model(model, epoch)

        self.save_model(model, num_epochs, force=True)

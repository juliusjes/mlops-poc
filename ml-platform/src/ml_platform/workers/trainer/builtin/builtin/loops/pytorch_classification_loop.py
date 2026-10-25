import importlib

import numpy as np
import torch
from ml_platform.workers.trainer.plugin_api.context import TrainingContext
from ml_platform.workers.trainer.plugin_api.training_loop import (
    TrainingLoop,
    TrainingResult,
)
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
from torch.utils.data import DataLoader

from builtin.utils import EarlyStopping


class PyTorchClassificationLoop(TrainingLoop):

    def __init__(self, seed=0, **kwargs):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.generator = torch.Generator()

        self.torch_nn = importlib.import_module("torch.nn")
        self.torch_optim = importlib.import_module("torch.optim")

    def train(self, ctx: TrainingContext):

        model = ctx.model   

        sampling_seed = ctx.training_config.sampling_seed
        tr_samples = ctx.training_config.n_training_samples

        if tr_samples == -1:
            ts_samples = -1
        else: 
            ts_samples = (ctx.training_config.test_size / ctx.training_config.train_size) * ctx.training_config.n_training_samples
            ts_samples = int(ts_samples)

        train_data = ctx.source.get_torch_dataset(split="train", num_splits=3, seed=sampling_seed, max_samples=tr_samples)
        test_data = ctx.source.get_torch_dataset(split="test", num_splits=3, seed=sampling_seed, max_samples=ts_samples)
        
        train_loader = DataLoader( train_data, **ctx.training_config.train_loader_kwargs)
        test_loader = DataLoader( test_data, **ctx.training_config.test_loader_kwargs)

        optimizer_class = getattr(self.torch_optim, ctx.training_config.optimizer)
        optimizer = optimizer_class( model.parameters(), **ctx.training_config.optimizer_kwargs )
        if ctx.checkpoint:
            optimizer.load_state_dict(ctx.checkpoint["optimizer_state_dict"])

            # put the optimizer state on the same device as model
            md = next(model.parameters()).device
            print(md)
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(md)

        loss_class = getattr( self.torch_nn,  ctx.training_config.loss_function)
        criterion = loss_class( **ctx.training_config.loss_function_kwargs )

        epochs = ctx.training_config.epochs

        device = ctx.training_config.device

        model.to(device=device)

        sk = ctx.source.dsd.sample_column[0]
        tk = ctx.source.dsd.target_column[0]

        # TODO: log datasets
        # train_df = self.dataset.df.iloc[self.train_dataset.indices]
        # test_df = self.dataset.df.iloc[self.test_dataset.indices]
        # tags = {"lakefs_commit": self.dataset.commit}
        # self.mlflow_service.log_training_input( train_df, self.dataset.dataset_path, additional_tags=tags)
        # self.mlflow_service.log_testing_input( test_df, self.dataset.dataset_path, additional_tags=tags)

        early_stopping = EarlyStopping(patience=10, min_delta=1e-3)
        stopped_early = False

        for epoch in range(epochs):

            # train
            model.train()
            train_loss = 0.0
            tr_count = 0
            for batch in train_loader:
                X = batch[sk].to(device=device)
                y = batch[tk].to(device=device)
                optimizer.zero_grad()
                pred = model(X)

                loss = criterion(pred["logits"], y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X.size(0)
                tr_count += X.size(0)

            train_loss /= tr_count

            # test
            model.eval()
            test_loss = 0.0
            ts_count = 0
            with torch.no_grad():

                for batch in test_loader:
                    X = batch[sk].to(device=device)
                    y = batch[tk].to(device=device)
                    preds = model(X)
                    loss = criterion(preds["logits"], y)
                    test_loss += loss.item() * X.size(0)
                    ts_count += X.size(0)

            test_loss /= ts_count

            print( f"Epoch {epoch + 1}/{epochs} | " f"train_loss={train_loss:.4f} | " f"test_loss={test_loss:.4f}")

            if early_stopping.step(test_loss, model):
                print("Stopping early")
                stopped_early = True
                break

             # TODO: move these outside the training loop
            ctx.mlf_logger.log_metric( "train_loss", train_loss, step=epoch, )
            ctx.mlf_logger.log_metric( "test_loss", test_loss, step=epoch, )

        model.load_state_dict(early_stopping.best_state)

        ctx.mlf_logger.log_params({
            "stopped_early": str(stopped_early),
            "trained_epochs": epoch + 1,
            "optimizer": ctx.training_config.optimizer,
            "loss_function": ctx.training_config.loss_function,
            **ctx.training_config.optimizer_kwargs,
            # **{f"train_{k}": v for k, v in self.trcfg.train_loader_kwargs.items()},
            # **{f"test_{k}": v for k, v in self.trcfg.test_loader_kwargs.items()},
        })

        model.eval() # enforce eval mode
        sample = batch # should work with iterable dataset
        x = sample[sk].detach().cpu().to(torch.float32)

        signature = ModelSignature(
            inputs=Schema([
                TensorSpec(
                    np.dtype(np.float32),
                    (-1, *x.shape[1:])
                )
            ])
        )

        model_info = ctx.mlf_logger.log_model(
            model,
            input_example=x[:2],
            step=epoch + 1,
            signature=signature,
            device=ctx.training_config.register_device
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        } 
        ctx.mlf_logger.log_checkpoint(
            checkpoint,
            ctx.training_config,
            )

        return TrainingResult(model_info=model_info)
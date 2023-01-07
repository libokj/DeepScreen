from typing import Any, Optional

from torch import nn, optim, squeeze, cat
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class DTI(LightningModule):
    """
        Drug Target Interaction Prediction
    """
    def __init__(
            self,
            binarization: bool,
            optimizer: optim.Optimizer,
            drug_encoder: Optional[nn.Module] = None,
            protein_encoder: Optional[nn.Module] = None,
            classifier: Optional[nn.Module] = None,
            model: Optional[nn.Module] = None,
            ):
        super().__init__()
        if model:
            # allows access to init params with 'self.hparams' attribute and ensures init params will be stored in ckpt
            self.save_hyperparameters(logger=False, ignore=["model"])
            self.model = model
        else:
            # allows access to init params with 'self.hparams' attribute and ensures init params will be stored in ckpt
            self.save_hyperparameters(logger=False, ignore=["drug_encoder", "protein_encoder", "classifier"])
            self.drug_encoder = drug_encoder
            self.protein_encoder = protein_encoder
            self.classifier = classifier

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        # use separate metric instances for train, val and test step to ensure a proper reduction over the epoch
        if binarization:
            self.train_acc = Accuracy()
            self.val_acc = Accuracy()
            self.test_acc = Accuracy()
            # for logging best-so-far validation accuracy
            self.val_acc_best = MaxMetric()

    def forward(self, v_d, v_p):
        # drug encoding and protein encoding
        v_d = self.drug_encoder(v_d.float())
        v_p = self.protein_encoder(v_p.float())

        # concatenate and classify
        v_f = self.classifier(cat((v_d, v_p), 1))

        return v_f

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        if self.hparams.binarization:
            self.val_acc_best.reset()

    def step(self, batch: Any):
        # common step for training/validation/test
        v_d, v_p, y = batch

        score = self.forward(v_d, v_p)

        if self.hparams.binarization:
            criterion = nn.BCELoss()
            score = nn.Sigmoid(score)
            n = squeeze(score, dim=1)
        else:
            criterion = nn.MSELoss()
            n = squeeze(score, dim=1)

        loss = criterion(n.float(), y.float())

        return loss, n, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.train_loss(loss)
        # log train metrics
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=False)

        if self.hparams.binarization:
            acc = self.train_acc(preds, targets)
            self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: list[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        if self.hparams.binarization:
            self.train_acc.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.val_loss(loss)
        # log val metrics
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=False)

        if self.hparams.binarization:
            acc = self.train_acc(preds, targets)
            self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: list[Any]):
        if self.hparams.binarization:
            acc = self.val_acc.compute()  # get val accuracy from current epoch
            self.val_acc_best.update(acc)
            self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
            self.val_acc.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.test_loss(loss)
        # log test metrics
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.hparams.binarization:
            acc = self.train_acc(preds, targets)
            self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: list[Any]):
        if self.hparams.binarization:
            self.test_acc.reset()

    def predict_step(self, batch: Any, batch_idx: int):
        v_d, v_p = batch
        score = self.forward(v_d, v_p)
        return {"preds": score}

    def configure_optimizers(self):
        return {
            "optimizer": self.hparams.optimizer(params=self.parameters()),
        }


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "dti.yaml")
    _ = hydra.utils.instantiate(cfg)

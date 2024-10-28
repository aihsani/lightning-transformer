import lightning as L
import torch
import torch.nn.functional as F
from util import generate_and_print_sample
from metrics import evaluate_model


class TrainingModule(L.LightningModule):

    def __init__(self, model, learning_rate, weight_decay, tokenizer, test_string, eval_num_batches=1, eval_freq=5):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.tokenizer = tokenizer
        self._test_string = test_string
        self.eval_freq = eval_freq
        self.eval_iter = eval_num_batches

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input_batch, target_batch = batch
        logits = self(input_batch)
        loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_batch, target_batch = batch
        logits = self(input_batch)
        loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        self.log("val_loss", loss, prog_bar=True)

        if self.global_step % self.eval_freq == 0:
            train_loss, val_loss = evaluate_model(
                self.model,
                self.train_dataloader(),
                self.val_dataloader(),
                self.device,
                self.eval_iter,
            )
            self.log("cumul_train_loss", train_loss, prog_bar=True)
            self.log("cumul_val_loss", val_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer

    def on_train_epoch_end(self):
        generate_and_print_sample(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            start_context=self._test_string,
        )

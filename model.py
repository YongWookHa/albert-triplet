import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel

import utils

class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters(ignore='hparams')

        self.tokenizer = AutoTokenizer.from_pretrained('AI-Growth-Lab/PatentSBERTa')
        """ Define Layers """
        
        self.net = AutoModel.from_pretrained("AI-Growth-Lab/PatentSBERTa")

        self.metric = torch.nn.TripletMarginLoss()

    def forward(self, input):
        """
        input:
            [B, ]
        return:
            [B, ]
        """
        encoded_input = self.tokenizer(list(input), return_tensors='pt', padding='max_length', truncation=True)
        model_output = self.net(**encoded_input.to(self.device))
        return model_output[0][:, 0]  # cls-pooling

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams['optimizer'])
        optimizer = optimizer(self.parameters(), lr=float(self.hparams['lr']))

        if not self.hparams['scheduler']:
            return optimizer
        elif hasattr(torch.optim.lr_scheduler, self.hparams['scheduler']):
            scheduler = getattr(torch.optim.lr_scheduler, self.hparams['scheduler'])
        elif hasattr(utils, self.hparams['scheduler']):
            scheduler = getattr(utils, self.hparams['scheduler'])
        else:
            raise ModuleNotFoundError

        scheduler = {
            'scheduler': scheduler(optimizer, **self.hparams['scheduler_param']),
            'interval': self.hparams['scheduler_interval'],
            'name': "learning_rate"
            }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_nb):
        q, pos, neg = zip(*batch)
        
        q_emb = self(q)
        pos_emb = self(pos)
        neg_emb = self(neg)
        loss = self.cal_loss(q_emb, pos_emb, neg_emb)
        
        self.log('train_loss', loss, batch_size=len(batch))
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        q, pos, neg = zip(*batch)

        q_emb = self(q)
        pos_emb = self(pos)
        neg_emb = self(neg)
        loss = self.cal_loss(q_emb, pos_emb, neg_emb)

        self.log('val_loss', loss, batch_size=len(batch))

        return {'val_loss':loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print(f"\nEpoch {self.current_epoch} | avg_loss:{avg_loss}\n")

    def cal_loss(self, query, positive, negative):
        """
        Define how to calculate loss

        logits:
            [B, ]
        targets:
            [B, ]
        """
        loss = self.metric(query, positive, negative)

        return loss

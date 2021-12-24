import torch
import torch.nn as nn
import time
import os


class Trainer:
    def __init__(self, model, optimizer, loader_train, patience=None,
                 scheduler=None, loader_train_eval=None, loader_val=None, cuda=None, logger=None, save_dir=None,
                 max_epochs=None, min_epochs=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loader_train = loader_train
        self.patience = patience
        self.scheduler = scheduler
        self.loader_train_eval = loader_train_eval
        self.loader_val = loader_val
        self.cuda = cuda
        self.logger = logger
        self.save_dir = save_dir
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs

    def train_epoch(self):
        """while true:
                grab batch
                zero out gradients
                y hat
                Loss (pass in desired loss fctn)
                backward prop
                update params"""
        self.model.train()
        total_loss = 0
        loss = nn.MSELoss()
        for batch_num, data in enumerate(self.loader_train):
            inputs = data[0].float()
            targets = data[1].float()
            if self.cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            curr_loss = loss(outputs, targets)
            curr_loss.backward()
            self.optimizer.step()
            total_loss += curr_loss
        return total_loss / len(self.loader_train)

    def compute_loss(self, dat_loader):
        self.model.eval()
        device = torch.device(
            "cuda:0" if self.cuda else "cpu"
        )
        total_loss = 0
        for batch_num, data in enumerate(self.loader_train):
            inputs = data[0].float()
            targets = data[1].float()
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = self.model(inputs)
            loss = nn.MSELoss()
            total_loss += loss(outputs, targets)
        return total_loss / len(dat_loader)

    def train(self):
        epoch = 1
        loss_val_best = 100
        num_epochs_in = 0
        # best_epoch = 1
        while True:
            # iterate SGD
            t0 = time.time()
            loss_train = self.train_epoch()
            print("Epoch trained")
            loss_train_eval = self.compute_loss(self.loader_train_eval)
            loss_val = self.compute_loss(self.loader_val)
            time_epoch = time.time() - t0
            self.logger.add_entry({'loss_train': loss_train.item(),
                                   'loss_train_eval': loss_train_eval.item(),
                                   'loss_val': loss_val.item()})

            # save logger info
            if self.save_dir:
                self.logger.append(os.path.join(self.save_dir, 'log.txt'))

            change_loss_val = ((loss_val - loss_val_best) / loss_val_best) * 100

            print("Epoch: " + str(epoch) + "/ Train: " + str(loss_train_eval.item()) + "/ Valid: " + str(loss_val.item()) +
                  "/ Diff Valid: " + str(change_loss_val.item()) + "/ Diff Valid-Train: " +
                  str(((loss_val - loss_train_eval)/loss_train_eval * 100).item()) + "/ Time: " + str(time_epoch) + "\n")

            # if new loss val is less than previous best
            if change_loss_val < -5:
                num_epochs_in = 0
                # best_epoch = epoch
                loss_val_best = loss_val
                if self.save_dir:
                    print('Loss Improved. Saving model')
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model.dat'))

            num_epochs_in += 1
            # patience, break if necessary
            if epoch > self.min_epochs and (epoch > self.max_epochs or num_epochs_in > self.patience):
                if num_epochs_in > self.patience:
                    print("Out of patience, breaking")
                break

            epoch += 1

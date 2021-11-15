import torch
import time
import os

class Trainer():
    def __init__(self, model, loss, optimizer, loader_train, patience=None,
                 scheduler=None, loader_train_eval=None, loader_val=None, cuda=None, logger=None, save_dir=None,
                 max_epochs=None, min_epochs=None):
        super().__init__()
        self.model = model
        self.loss = loss
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
        for batch_num, data in enumerate(self.loader_train):
            inputs = data[0]
            targets = data[0]
            #check if gpu
            if self.cuda:
                inputs = inputs.cuda()
                outputs = targets.cuda()
                self.loss = self.loss.cuda()
            #FIXME check this line
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)
            total_loss = loss.data.item()
        return total_loss/ len(dat_loader) #FIXME

    def train(self):
        epoch = 1
        loss_val_best = 100
        num_epochs_in = 0
        best_epoch = 1
        while True:
            t0 = time.time()
            loss_train = self.train_epoch()
            loss_train_eval = self.compute_loss(self.loader.train_eval)
            loss_val = self.compute_loss(self.loader_val)
            t1 = time.time()-t0
            #FIXME add logger
            #FIXME learning rate scheduler/ learning rate decay
            """
            display results using variables
            if self.save_dir
                log if given a directory to save out to
            """
            change_loss_val = ((loss_val - loss_val_best)/loss_val_best)*100
            # if new loss val is less than previous best
            if change_loss_val < 0:
                num_epochs_in = 0
                best_epoch = epoch
                loss_val_best = loss_val
                if self.save_dir:
                    print('Loss Improved. Saving model')
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model.dat'))

            num_epochs_in += 1
            # patience, break if necessary
            if epoch > self.min_epochs and (epoch > self.max_epochs or num_epochs_in > self.patience):
                break

            epoch += 1

    def compute_loss(self, dat_loader):
        self.model.eval()
        device = torch.device(
            "cuda:0" if self.cuda else "cpu"
        )
        total_loss = 0
        for batch_num, data in enumerate(self.loader_train):
            inputs = data[0]
            targets = data[0]
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)
            total_loss = loss.data.item()
        return total_loss / len(dat_loader)

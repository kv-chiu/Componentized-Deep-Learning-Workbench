import torch

class myOptimizer(torch.optim.SGD):
    def __init__(self, config, params):
        super().__init__(params, lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
        self.config = config
        self.lr = config.lr
        self.momentum = 0.9
        self.weight_decay = config.weight_decay

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.lr = self.config.lr * (0.1 ** (epoch // 30))
        for param_group in self.param_groups:
            param_group['lr'] = self.lr
        return self.lr
    

class myScheduler(torch.optim.lr_scheduler.StepLR):
    def __init__(self, config, optimizer):
        super().__init__(optimizer, step_size=config.step_size, gamma=config.gamma)
        self.config = config
        self.step_size = config.step_size
        self.gamma = config.gamma

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.lr = self.optimizer.lr * (0.1 ** (epoch // 30))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr
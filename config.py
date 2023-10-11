import torch
from dataloader import myDataset
from model import myModel
from criterion import myCriterion
from optimizer import myOptimizer, myScheduler


class myConfig:
    comment = 'first_try'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 文件路径
    data_dir = '/media/ders/mazhiming/scripts/ykq/dataset/eyes_standard/stage2/'
    train_input_path = data_dir + 'processed_train.csv'
    train_label_path = data_dir + 'processed_multi_train_label.csv'
    val_input_path = data_dir + 'processed_train.csv'
    val_label_path = data_dir + 'processed_multi_train_label.csv'
    test_input_path = data_dir + 'processed_test.csv'
    test_label_path = data_dir + 'processed_multi_test_label.csv'

    # 模型配置
    class ModelConfig:
        snps = 7853
        num_classes = 8
        in_channels = 4
        kernel_sizes = {
            'conv1': 4,
            'pool1': 8,
        }

    # 优化器配置
    class OptConfig:
        lr = 3e-3
        weight_decay = 0.01

    # 学习率调整器配置
    class SchedulerConfig:
        step_size = 25
        gamma = 0.9

    # 损失函数配置
    class LossConfig:
        smoothing = 0.05
        dim = -1
        num_classes = 8

    @classmethod
    def create_dataset(cls, input_path, label_path, mode, has_header=False, has_index=False):
        return myDataset(
            input_path=input_path,
            label_path=label_path,
            mode=mode,
            has_header=has_header,
            has_index=has_index
        )

    @classmethod
    def create_model(cls):
        model = myModel(cls.ModelConfig)
        return model.to(cls.device)

    @classmethod
    def create_optimizer(cls, model):
        return myOptimizer(
            config=cls.OptConfig,
            params=model.parameters()
        )
    
    @classmethod
    def create_scheduler(cls, optimizer):
        return myScheduler(
            config=cls.SchedulerConfig,
            optimizer=optimizer
        )
    
    @classmethod
    def create_loss_func(cls):
        return myCriterion(
            config=cls.LossConfig
        )

    def __init__(self):
        # 数据集
        self.train_dataset = self.create_dataset(self.train_input_path, self.train_label_path, 'train')
        self.train_batch_size = 2048
        self.val_dataset = self.create_dataset(self.val_input_path, self.val_label_path, 'val')
        self.val_batch_size = 512
        self.test_dataset = self.create_dataset(self.test_input_path, self.test_label_path, 'test')
        self.test_batch_size = 512

        # 模型
        self.model = self.create_model()

        # 损失函数
        self.loss_func = self.create_loss_func()

        # 优化器
        self.optimizer = self.create_optimizer(self.model)

        # 学习率调整器
        self.scheduler = self.create_scheduler(self.optimizer)

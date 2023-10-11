import torch
from sklearn.metrics import confusion_matrix

from utils import calculate_metrics, writer_add, MetricsTracker


def train_one_epoch(i, writer, config):
    print("--------第{}轮训练开始---------".format(i))

    tracker = MetricsTracker()

    train_loader = torch.utils.data.DataLoader(
        dataset=config.train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )

    config.model.train()

    for data in train_loader:
        X_data, y_data = data
        # X_data, y_data = X_data.to(config.device), y_data.to(config.device)
        output = config.model(X_data)
        loss = config.loss_func(output, y_data)
        
        config.optimizer.zero_grad()
        loss.backward()
        config.optimizer.step()
        config.scheduler.step()

        pred = output.argmax(axis=1)
        y_data, pred = y_data.cpu().numpy(), pred.cpu().numpy()
        matrix = confusion_matrix(y_data, pred)
        acc, precision, recall, F1, G_mean, TN, FN, TP, FP = calculate_metrics(matrix)

        tracker.update(loss, acc, precision, recall, F1, G_mean, TN, FN, TP, FP)

    lr = config.optimizer.param_groups[0]['lr']
    m_dict = tracker.get_epoch_metrics()

    writer_add(writer, m_dict['avg_loss'], m_dict['avg_accuracy'], m_dict['avg_precision'], m_dict['avg_recall'], m_dict['avg_f1'], m_dict['avg_g_mean'], lr, i, y_data, pred, m_dict['total_TN'], m_dict['total_FN'], m_dict['total_TP'], m_dict['total_FP'], mode='train')

    print("--------第{}轮训练结束---------".format(i))


def val_one_epoch(i, writer, config):
    print("--------第{}轮验证开始---------".format(i))

    tracker = MetricsTracker()

    val_loader = torch.utils.data.DataLoader(
        dataset=config.val_dataset,
        batch_size=config.val_batch_size,
        shuffle=True
    )

    config.model.eval()

    for data in val_loader:
        X_data, y_data = data
        # X_data, y_data = X_data.to(config.device), y_data.to(config.device)
        output = config.model(X_data)
        loss = config.loss_func(output, y_data)

        pred = output.argmax(axis=1)
        y_data, pred = y_data.cpu().numpy(), pred.cpu().numpy()
        matrix = confusion_matrix(y_data, pred)
        acc, precision, recall, F1, G_mean, TN, FN, TP, FP = calculate_metrics(matrix)

        tracker.update(loss, acc, precision, recall, F1, G_mean, TN, FN, TP, FP)

    lr = config.optimizer.param_groups[0]['lr']
    m_dict = tracker.get_epoch_metrics()

    writer_add(writer, m_dict['avg_loss'], m_dict['avg_accuracy'], m_dict['avg_precision'], m_dict['avg_recall'], m_dict['avg_f1'], m_dict['avg_g_mean'], lr, i, y_data, pred, m_dict['total_TN'], m_dict['total_FN'], m_dict['total_TP'], m_dict['total_FP'], mode='val')

    print("--------第{}轮验证结束---------".format(i))


def test_one_epoch(i, writer, config):
    print("--------第{}轮测试开始---------".format(i))

    tracker = MetricsTracker()

    test_loader = torch.utils.data.DataLoader(
        dataset=config.test_dataset,
        batch_size=config.test_batch_size,
        shuffle=True
    )

    config.model.eval()

    for data in test_loader:
        X_data, y_data = data
        # X_data, y_data = X_data.to(config.device), y_data.to(config.device)
        output = config.model(X_data)
        loss = config.loss_func(output, y_data)

        pred = output.argmax(axis=1)
        y_data, pred = y_data.cpu().numpy(), pred.cpu().numpy()
        matrix = confusion_matrix(y_data, pred)
        acc, precision, recall, F1, G_mean, TN, FN, TP, FP = calculate_metrics(matrix)

        tracker.update(loss, acc, precision, recall, F1, G_mean, TN, FN, TP, FP)

    lr = config.optimizer.param_groups[0]['lr']
    m_dict = tracker.get_epoch_metrics()

    writer_add(writer, m_dict['avg_loss'], m_dict['avg_accuracy'], m_dict['avg_precision'], m_dict['avg_recall'], m_dict['avg_f1'], m_dict['avg_g_mean'], lr, i, y_data, pred, m_dict['total_TN'], m_dict['total_FN'], m_dict['total_TP'], m_dict['total_FP'], mode='test')

    print("--------第{}轮测试结束---------".format(i))

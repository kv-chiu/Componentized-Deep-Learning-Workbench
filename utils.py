import math
import numpy as np
from typing import Tuple
import torch
import random

def calculate_metrics(matrix) -> Tuple[float, float, float, float, float, float, float, float, float]:
    TN = matrix[0][0]
    FN = matrix[1][0]
    TP = matrix[1][1]
    FP = matrix[0][1]

    FDR = TP / (TP + FN)
    FAR = FP / (FP + TN)
    P = TN / (TN + FP)
    G_mean = math.sqrt(FDR * P)

    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)

    if np.isnan(G_mean):
        G_mean = 0.0
    if np.isnan(acc):
        acc = 0.0
    if np.isnan(precision):
        precision = 0.0
    if np.isnan(recall):
        recall = 0.0
    if np.isnan(F1):
        F1 = 0.0

    return acc, precision, recall, F1, G_mean, TN, FN, TP, FP


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)


def writer_add(writer,
                loss,
                acc,
                precision,
                recall, F1,
                G_mean, lr,
                i,
                y_data,
                pred,
                TN, FN, TP, FP,
                mode='train'
                ):
    # 如果loss.item()存在，则写入，否则写入loss
    if hasattr(loss, 'item'):
        writer.add_scalar(f'Loss/{mode}', loss.item(), i)
    else:
        writer.add_scalar(f'Loss/{mode}', loss, i)
    writer.add_scalar(f'Accuracy/{mode}', acc, i)
    writer.add_scalar(f'Precision/{mode}', precision, i)
    writer.add_scalar(f'Recall/{mode}', recall, i)
    writer.add_scalar(f'F1/{mode}', F1, i)
    writer.add_scalar(f'G_mean/{mode}', G_mean, i)
    writer.add_scalar('Learning rate', lr, i)

    writer.add_scalar(f'Result/Truth/{mode}', len(y_data), i)
    writer.add_scalar(f'Result/Pred/{mode}', len(pred), i)
    writer.add_scalar(f'Result/TP/{mode}', TP, i)
    writer.add_scalar(f'Result/TN/{mode}', TN, i)
    writer.add_scalar(f'Result/FP/{mode}', FP, i)
    writer.add_scalar(f'Result/FN/{mode}', FN, i)


class MetricsTracker:
    def __init__(self):
        self.losses = []  # 用于存储每个批次的损失
        self.accuracies = []  # 用于存储每个批次的准确率
        self.precisions = []  # 用于存储每个批次的精确度
        self.recalls = []  # 用于存储每个批次的召回率
        self.f1_scores = []  # 用于存储每个批次的F1分数
        self.g_means = []  # 用于存储每个批次的G-mean
        self.TNs = []  # 用于存储每个批次的真负例
        self.FNs = []  # 用于存储每个批次的假负例
        self.TPs = []  # 用于存储每个批次的真正例
        self.FPs = []  # 用于存储每个批次的假正例

    def update(self, loss, acc, precision, recall, f1, g_mean, TN, FN, TP, FP):
        self.losses.append(loss)
        self.accuracies.append(acc)
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1_scores.append(f1)
        self.g_means.append(g_mean)
        self.TNs.append(TN)
        self.FNs.append(FN)
        self.TPs.append(TP)
        self.FPs.append(FP)

    def get_epoch_metrics(self):
        avg_loss = sum(self.losses) / len(self.losses)
        avg_accuracy = sum(self.accuracies) / len(self.accuracies)
        avg_precision = sum(self.precisions) / len(self.precisions)
        avg_recall = sum(self.recalls) / len(self.recalls)
        avg_f1 = sum(self.f1_scores) / len(self.f1_scores)
        avg_g_mean = sum(self.g_means) / len(self.g_means)
        total_TN = sum(self.TNs)
        total_FN = sum(self.FNs)
        total_TP = sum(self.TPs)
        total_FP = sum(self.FPs)

        return {
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'avg_g_mean': avg_g_mean,
            'total_TN': total_TN,
            'total_FN': total_FN,
            'total_TP': total_TP,
            'total_FP': total_FP
        }

    def reset(self):
        self.losses = []
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.g_means = []
        self.TNs = []
        self.FNs = []
        self.TPs = []
        self.FPs = []

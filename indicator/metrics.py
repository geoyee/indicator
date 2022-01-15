import numpy as np
import paddle
import sklearn.metrics as skmetrics


def cal_areas(pred, label, num_classes, ignore_index=255):
    """
    Calculate intersect, prediction and label area

    Args:
        pred (Tensor): The prediction by model.
        label (Tensor): The ground truth of image.
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.

    Returns:
        dict (Tensor): Include:
            The prediction area on all class.
            The ground truth area on all class.
            The TP area on all class.
            The TN area on all class.
            The FP area on all class.
            The FN area on all class.
    """
    if len(pred.shape) == 4:
        pred = paddle.squeeze(pred, axis=1)
    if len(label.shape) == 4:
        label = paddle.squeeze(label, axis=1)
    if not pred.shape == label.shape:
        raise ValueError('Shape of `pred` and `label should be equal, '
                         'but there are {} and {}.'.format(
                             pred.shape, label.shape))
    pred_area = []
    label_area = []
    TP_area = []  # is intersect
    FP_area = []
    FN_area = []
    TN_area = []
    mask = label != ignore_index
    for i in range(num_classes):
        pred_i = paddle.logical_and(pred == i, mask)
        label_i = label == i
        TP_i = paddle.logical_and(pred_i, label_i)
        FP_i = paddle.logical_and(paddle.logical_not(TP_i), pred_i)
        FN_i = paddle.logical_and(paddle.logical_not(TP_i), label_i)
        TN_i = paddle.logical_not(paddle.logical_or(pred_i, label_i))
        pred_area.append(paddle.sum(paddle.cast(pred_i, "int32")))
        label_area.append(paddle.sum(paddle.cast(label_i, "int32")))
        TP_area.append(paddle.sum(paddle.cast(TP_i, "int32")))
        FP_area.append(paddle.sum(paddle.cast(FP_i, "int32")))
        FN_area.append(paddle.sum(paddle.cast(FN_i, "int32")))
        TN_area.append(paddle.sum(paddle.cast(TN_i, "int32")))
    areas_result = {
        "pred_area": paddle.concat(pred_area),
        "label_area": paddle.concat(label_area),
        "confusion_matrix": {
            "TP_area": paddle.concat(TP_area),
            "FP_area": paddle.concat(FP_area),
            "FN_area": paddle.concat(FN_area),
            "TN_area": paddle.concat(TN_area)
        }
    }
    return areas_result


def cal_quality_indexrates(TP, FP, FN, TN, eps=1e-12):
    """
    Calculate quality indexrates based on confusion matrix
    Args:
        TP, FP, FN, TN (Tensor)
    Returns:
        dict (Tensor): Include:
            Accuracy.
            Precison.
            Recall.
            False Alarm.
            Missing Alarm.
            F1-Sorce.
    """
    TP = TP.numpy()
    FP = FP.numpy()
    FN = FN.numpy()
    TN = TN.numpy()
    accuracy = []
    precison = []
    recall = []
    false_alarm = []
    missing_alarm = []
    F1 = []
    for i in range(len(TP)):
        accuracy_i = (TP[i] + TN[i]) / (TP[i] + FP[i] + FN[i] + TN[i] + eps)
        precison_i = TP[i] / (TP[i] + FP[i] + eps)
        recall_i = TP[i] / (TP[i] + FN[i] + eps)
        false_alarm_i = 1 - precison_i
        missing_alarm_i = 1 - recall_i
        F1_i = (2 * precison_i * recall_i) / (precison_i + recall_i + eps)
        accuracy.append(accuracy_i)
        precison.append(precison_i)
        recall.append(recall_i)
        false_alarm.append(false_alarm_i)
        missing_alarm.append(missing_alarm_i)
        F1.append(F1_i)
    rates_result = {
        "accuracy": np.mean(accuracy),
        "precison": np.mean(precison),
        "recall": np.mean(recall),
        "false_alarm": np.mean(false_alarm),
        "missing_alarm": np.mean(missing_alarm),
        "F1": np.mean(F1)
    }
    return rates_result


def cal_auc_roc(logits, label, num_classes, ignore_index=None):
    """
    Calculate area under the roc curve
    Args:
        logits (Tensor): The prediction by model on testset, of shape (N,C,H,W) .
        label (Tensor): The ground truth of image.   (N,1,H,W)
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.
    Returns:
        auc_roc(float): The area under roc curve
    """
    if ignore_index or len(np.unique(label)) > num_classes:
        raise RuntimeError('labels with ignore_index is not supported yet.')
    if len(label.shape) != 4:
        raise ValueError(
            'The shape of label is not 4 dimension as (N, C, H, W), it is {}'.
            format(label.shape))
    if len(logits.shape) != 4:
        raise ValueError(
            'The shape of logits is not 4 dimension as (N, C, H, W), it is {}'.
            format(logits.shape))
    N, C, H, W = logits.shape
    logits = np.transpose(logits, (1, 0, 2, 3))
    logits = logits.reshape([C, N * H * W]).transpose([1, 0])
    label = np.transpose(label, (1, 0, 2, 3))
    label = label.reshape([1, N * H * W]).squeeze()
    if not logits.shape[0] == label.shape[0]:
        raise ValueError('length of `logit` and `label` should be equal, '
                         'but they are {} and {}.'.format(
                             logits.shape[0], label.shape[0]))
    if num_classes == 2:
        auc = skmetrics.roc_auc_score(label, logits[:, 1])
    else:
        auc = skmetrics.roc_auc_score(label, logits, multi_class='ovr')
    return auc


def cal_mean_iou(intersect_area, pred_area, label_area):
    """
    Calculate iou.
    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.
    Returns:
        np.ndarray: iou on all classes.
        float: mean iou of all classes.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    union = pred_area + label_area - intersect_area
    class_iou = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            iou = 0
        else:
            iou = intersect_area[i] / union[i]
        class_iou.append(iou)
    miou = np.mean(class_iou)
    return np.array(class_iou), miou


def cal_kappa(intersect_area, pred_area, label_area):
    """
    Calculate kappa coefficient
    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.
    Returns:
        float: kappa coefficient.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    total_area = np.sum(label_area)
    po = np.sum(intersect_area) / total_area
    pe = np.sum(pred_area * label_area) / (total_area * total_area)
    kappa = (po - pe) / (1 - pe)
    return kappa


def cal_dice(intersect_area, pred_area, label_area):
    """
    Calculate DICE
    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.
    Returns:
        np.ndarray: DICE on all classes.
        float: mean DICE of all classes.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    union = pred_area + label_area
    class_dice = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            dice = 0
        else:
            dice = (2 * intersect_area[i]) / union[i]
        class_dice.append(dice)
    mdice = np.mean(class_dice)
    return np.array(class_dice), mdice
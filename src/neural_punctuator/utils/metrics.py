import logging
import sys
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score
import numpy as np

from neural_punctuator.utils.visualize import plot_confusion_matrix


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-9s %(message)s'))

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

if (log.hasHandlers()):
    log.handlers.clear()
    
log.addHandler(handler)


def get_eval_metrics(targets, preds, config):
    # TODO: get the desired metric list from config-frozen.yaml
    """
    Calculates metrics on validation data
    """
    metrics = {}

    preds = np.exp(preds)
    preds = preds.reshape(-1, config.model.num_classes)
    targets = targets.reshape(-1)
    pred_index = preds.argmax(-1)

    p_class, count_p = np.unique(targets,return_counts=True)
    class_dict = dict(zip(p_class,count_p))
    non_empty_weights = dict()
    
    tot_count = sum(class_dict.values())-class_dict[0] #aside from empty
    
    for i,j in zip(config.data.output_labels,class_dict.keys()):
        if i.lower() != 'empty':
            non_empty_weights[i] = class_dict[j]/tot_count
            
    # One-hot encode targets
    # metric_targets = np.zeros((targets.size, config.model.num_classes))
    # metric_targets[np.arange(targets.size), targets] = 1

    cls_report, cls_report_print = get_classification_report(targets, pred_index, config)
    # print(cls_report_print)
    metrics['cls_report'] = cls_report

    ## ignore empty class for macro and weighted averages
    sum_precision, sum_recall, sum_f1_score = 0, 0, 0
    weight_precision, weight_recall, weight_f1_score = 0, 0, 0
    for i in config.data.output_labels:
        if i.lower() != 'empty':
            sum_precision += cls_report[i]['precision']
            sum_recall    += cls_report[i]['recall']
            sum_f1_score  += cls_report[i]['f1-score']
            
            weight_precision += cls_report[i]['precision']*non_empty_weights[i]
            weight_recall    += cls_report[i]['recall']*non_empty_weights[i]
            weight_f1_score  += cls_report[i]['f1-score']*non_empty_weights[i]
            
            print(f"{i:10}:\t{cls_report[i]['precision']:1.3f}\t{cls_report[i]['recall']:1.3f}\t{cls_report[i]['f1-score']:1.3f}")

    macro_precision = sum_precision/(config.model.num_classes-1) #precision_score(targets, pred_index, average='macro')
    log.info(f'Macro precision is: {macro_precision:1.3f}')
    metrics['precision'] = macro_precision

    macro_recall = sum_recall/(config.model.num_classes-1) #recall_score(targets, pred_index, average='macro')
    log.info(f'Macro recall is {macro_recall:1.3f}')
    metrics['recall'] = macro_recall

    macro_f1_score = sum_f1_score/(config.model.num_classes-1) #f1_score(targets, pred_index, average='macro')
    log.info(f'Macro f-score is {macro_f1_score:1.3f}')
    metrics['f_score'] = macro_f1_score


    log.info(f'Weighted precision is {weight_precision:1.3f}')
    log.info(f'Weighted recall is {weight_recall:1.3f}')
    log.info(f'Weighted f-score is {weight_f1_score:1.3f}')
    #Since masking might yield zero for a certain class in extreme circumstances
    #auc_score = roc_auc_score(targets, preds, average='macro', multi_class='ovo')
    #log.info(f'AUC is: {auc_score}')
    #metrics['auc'] = auc_score


    conf_mx = get_confusion_mx(targets, pred_index)
    if config.trainer.show_confusion_matrix:
        plot_confusion_matrix(conf_mx, config.data.output_labels)

    return metrics


def get_classification_report(target, preds, config):
    report = classification_report(target, preds, output_dict=True, target_names=config.data.output_labels)
    report_print = classification_report(target, preds, digits=3, target_names=config.data.output_labels)
    return report, report_print


def get_confusion_mx(target, preds):
    return confusion_matrix(target[1:], preds[1:])


def get_total_grad_norm(parameters, norm_type=2):
    total_norm = 0
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    return total_norm ** (1. / norm_type)
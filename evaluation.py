# -*- coding: utf-8 -*-
# Author: Hoan Bui Dang
# Python: 3.6

"""
Home-made tools for model evaluation.
"""

import pandas as pd
import numpy as np


def confusion(predicted,fact):
    count_total = len(predicted)
    
    true_pos = sum((predicted == 1) & (fact == 1)).astype(int)
    true_neg = sum((predicted == 0) & (fact == 0)).astype(int)
    false_pos = sum((predicted == 1) & (fact == 0)).astype(int)
    false_neg = sum((predicted == 0) & (fact == 1)).astype(int)
    
    accuracy = (true_pos + true_neg)/count_total
    pos_recall = true_pos / (true_pos + false_neg)
    pos_precision = true_pos / (true_pos + false_pos)
    neg_recall = true_neg / (true_neg + false_pos)
    neg_precision = true_neg / (true_neg + false_neg)
    f1_score = 2/((1/pos_recall) + (1/pos_precision))
    
    print('Confusion matrix:')
    print('%7d %7d' % (true_pos, false_pos))
    print('%7d %7d' % (false_neg, true_neg))
    print()
    print('Accuracy      : %.4f' % accuracy)
    print('Pos recall    : %.4f' % pos_recall)
    print('Pos precision : %.4f' % pos_precision)
    print('Neg recall    : %.4f' % neg_recall)
    print('Neg precision : %.4f' % neg_precision)
    print('F1 score      : %.4f' % f1_score)
    
def KS_chart(score, target):
    """ aka AR chart """
    
    if any((target!=0) & (target!=1)):
        print('Target for gain_chart must contain only 0 and 1.')
        return
    
    df = pd.DataFrame({'score': score, 'target':target})
    df = df.sort_values(by='score')
    L = len(df)
    target_count = sum(df.target)
    
    x = np.linspace(0,1,1001)
    y = np.array([])
    z = np.array([])
    for i in x.tolist():
        partial = df.iloc[:int(L*i)]
        y = np.append(y, sum(partial.target)/target_count)
        z = np.append(z, (len(partial)-sum(partial.target))/(L-target_count))
    diff = y-z
    ind_max = np.argmax(diff)
    print('KS: ',np.max(diff))
    print('Score at max:', score[ind_max])
    return np.column_stack((x,y,z))



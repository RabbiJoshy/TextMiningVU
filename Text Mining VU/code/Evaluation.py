import pandas as pd
from collections import Counter
import numpy as np

def EvaluatePredictionsI_NEG(Predictions, Gold):
    """

    :param Predictions:
    :param Gold:
    :return:
    """

    Total = len(Gold)
    labcount = Counter(Gold)
    fcontribution = []
    rcontribution= []
    preccontribution = []
    macrof = []
    macror = []
    macrop = []

    ClassInfo = []
    for label in set(Gold):
        contribution = labcount[label]/Total

        confdict = {'TP': 0.0001, 'TN': 0.000001, 'FP': 0.1, 'FN': 0.1}
        scoredict = {'Precision': 0, 'F_Score': 0, 'Recall': 0}

        for i in range(Total):
            if Predictions[i] == Gold[i]:
                if Predictions[i] == label:
                    confdict['TP'] += 1
                else:
                    confdict['TN'] += 1

            else:
                if Predictions[i] == label:
                    confdict['FP'] += 1
                else:
                    confdict['FN'] += 1

        #Failures = (confdict['FP'] + confdict['FN'], confdict['FP'], confdict['FN'])
        #Accuracy = (confdict['TP'] + confdict['TN']) / Total
        scoredict['Precision'] = round(confdict['TP'] / (confdict['TP'] + confdict['FP']), 4)
        scoredict['Recall'] = round(confdict['TP'] / (confdict['TP'] + confdict['FN']),4)
        scoredict['F_Score'] = round((2 * (scoredict['Precision'] * scoredict['Recall'])) / (scoredict['Precision'] + scoredict['Recall']), 4)
        fcontribution.append((contribution * scoredict['F_Score']))
        rcontribution.append((contribution * scoredict['Recall']))
        preccontribution.append((contribution * scoredict['Precision']))

        macrof.append(scoredict['F_Score'])
        macror.append(scoredict['Recall'])
        macrop.append(scoredict['Precision'])

        # print(label, ': Scores:', scoredict, '\n', 'Confusions:', confdict, '\n')


        ClassInfo.append(scoredict)

        # print(label, '& ',scoredict['Precision'], '& ', scoredict['Recall'], '& ',scoredict['F_Score'] )

    Avg_F_Scores = {'Micro_F_Score': np.sum(fcontribution), 'Macro_F_Score' : np.mean(macrof)}
    Avg_R_Scores = {'Micro_R_Score': np.sum(rcontribution), 'Macro_R_Score': np.mean(macror)}
    Avg_P_Scores = {'Micro_P_Score': np.sum(preccontribution), 'Macro_P_Score': np.mean(macrop)}

    print(Avg_F_Scores)
    print(Avg_R_Scores)
    print(Avg_P_Scores)
    # print(macrof)
    #
    # print(Avg_F_Scores['Micro_F_Score'])
    # print(ClassInfo)

    return confdict, Avg_F_Scores

def EvaluatePredictions(Predictions, Gold):

    Total = len(Gold)
    labcount = Counter(Gold)
    fcontribution = []
    macrof = []

    ClassInfo = []
    errors = 0
    for label in ['B-NEG', 'O']:
        contribution = labcount[label]/Total

        confdict = {'TP': 0.01, 'TN': 0.01, 'FP': 0.01, 'FN': 0.01}
        scoredict = {'Precision': 0, 'F_Score': 0, 'Recall': 0}

        for i in range(Total):
            if Predictions[i] == Gold[i]:
                if Predictions[i] == label:
                    confdict['TP'] += 1
                else:
                    confdict['TN'] += 1

            else:
                if Predictions[i] == label:
                    confdict['FP'] += 1
                else:
                    confdict['FN'] += 1
                errors += 1

        scoredict['Precision'] = round(confdict['TP'] / (confdict['TP'] + confdict['FP']), 4)
        scoredict['Recall'] = round(confdict['TP'] / (confdict['TP'] + confdict['FN']),4)
        scoredict['F_Score'] = round((2 * (scoredict['Precision'] * scoredict['Recall'])) / (scoredict['Precision'] + scoredict['Recall']), 4)
        fcontribution.append((contribution * scoredict['F_Score']))
        macrof.append(scoredict['F_Score'])

        # print(label, ': Scores:', scoredict, '\n', 'Confusions:', confdict, '\n')

        ClassInfo.append((scoredict, confdict))

    Avg_F_Scores = {'Micro_F_Score': np.sum(fcontribution), 'Macro_F_Score' : np.mean(macrof), 'B-NEG/O F-Scores': macrof}

    # print(Avg_F_Scores)
    print(Avg_F_Scores['Micro_F_Score'])
    print(errors)

    return confdict, Avg_F_Scores

def evalpredfile(predfile):
    preds_df = pd.read_csv(predfile, encoding="utf-8")
    Predictions = preds_df['Pred']
    Gold = preds_df['Gold']
    EvaluatePredictionsI_NEG(Predictions, Gold)

def main(Predfile):
    preds_df = pd.read_csv(Predfile, encoding="utf-8")
    Predictions = preds_df['Prediction']
    Gold = preds_df['Negation_cue']
    EvaluatePredictions(Predictions, Gold)


testev = "/Users/joshuawork/Desktop/Assignment4/Data/Method2/Predictions/Development/SVM-HECA-Token_vector.csv"
evalpredfile(testev)

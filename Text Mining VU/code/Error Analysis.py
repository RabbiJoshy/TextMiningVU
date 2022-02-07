import pandas as pd
from Evaluation import EvaluatePredictions
from Evaluation import evalpredfile
from pathlib import Path

#Concat

# circ = "/Users/joshuawork/Desktop/Assignment4/Data/Method2/circle.Preprocessed.pickle"
# card = "/Users/joshuawork/Desktop/Assignment4/Data/Method2/cardboard.Preprocessed.pickle"
# def concattests(testfile1, testfile2):
#     test1 = pd.read_pickle(testfile1)
#     test2 = pd.read_pickle(testfile2)
#     test_features_df= pd.concat([test1, test2], ignore_index=True, sort=False)
#
#     return test_features_df
#
# concatdf = concattests(circ, card)

def fixsnakes(file):
    df = pd.read_csv(file)
    ErrorTypes = []
    for index, row in df.iterrows():
        if row['pred_CRF'] == row['Negation_cue']:
            ErrorTypes.append('True')
        elif row['Negation_cue'] == 'I-NEG':
            ErrorTypes.append('I-NEG')
        elif row['Negation_cue'] == 'B-NEG':
            ErrorTypes.append('FN')
        else:
            ErrorTypes.append('FP')

    df['Error Type'] = ErrorTypes

    # circ = df.iloc[:10184]
    # card = df.iloc[10184:]
    #
    # df = pd.concat([card,circ])

    df.to_csv(file)
# fixsnakes("/Users/joshuawork/Desktop/Assignment4/Data/Method2/Predictions/Snakes/CRF_dev_result.csv")

def comparepredictions(predictions1, predictions2):
    pred1 = pd.read_csv(predictions1)
    pred2 = pd.read_csv(predictions2)

    ErrorComparisonDF = pred1.loc[pred1['Error Type'] != pred2['Error Type']]
    ErrorComparisonDF['Pred2'] = pred2.loc[pred1['Error Type'] != pred2['Error Type'],'Pred']

    print(ErrorComparisonDF)
    print(ErrorComparisonDF.to_latex(index=False))

# comparepredictions(
#     "/Users/joshuawork/Desktop/Assignment4/Data/Method2/Predictions/Development/SVM-HECA-Token_vector.csv",
#     "/Users/joshuawork/Desktop/Assignment4/Data/Method2/Predictions/Development/SVM-HECA-HECT-trigrams_unlisted.csv"
# )

def comparecrf(predictions1, predictionssnake):
    pred1 = pd.read_csv(predictions1)
    pred2 = pd.read_csv(predictionssnake)

    ErrorComparisonDF = pred1.loc[pred1['Pred'] != pred2['pred_CRF']]
    ErrorComparisonDF['PredCRF'] = pred2.loc[pred1['Pred'] != pred2['Error Type'],'pred_CRF']
    ErrorComparisonDF.drop(columns=ErrorComparisonDF.columns[0], inplace=True)

    print(ErrorComparisonDF)
    print(ErrorComparisonDF.to_latex(index=False))

# comparecrf(
#     "/Users/joshuawork/Desktop/Assignment4/Data/Method2/Predictions/Development/SVM-HECA-Token_vector.csv",
#     "/Users/joshuawork/Desktop/Assignment4/Data/Method2/Predictions/Snakes/CRF_dev_result.csv"
# )
def comparemlp(predictions1, predictionssnake):
    pred1 = pd.read_csv(predictions1)
    pred2 = pd.read_csv(predictionssnake)

    ErrorComparisonDF = pred1.loc[pred1['Pred'] != pred2['pred_MLP']]
    ErrorComparisonDF['Pred2'] = pred2.loc[pred1['Pred'] != pred2['Error Type'],'pred_MLP']

    print(ErrorComparisonDF.reset_index())


# comparemlp(
#     "/Users/joshuawork/Desktop/Assignment4/Data/Method2/Predictions/Development/SVM-HECA-Token_vector.csv",
#     "/Users/joshuawork/Desktop/Assignment4/Data/Method2/Predictions/Snakes/MLP_test_result.csv"
# )

def comparefscores(directory_in_str):
    # import os
    # directory = os.fsencode(directory_in_str)
    # for file in os.listdir(directory):
    #     filename = os.fsdecode(file)
    #     print(filename)
    pathlist = Path(directory_in_str).glob('*')
    for path in pathlist:
        print(path)
        evalpredfile(path)

comparefscores("/Users/joshuawork/Desktop/Assignment4/Data/Method2/Predictions/Development")

def finderrors(predfile):
    df = pd.read_csv(predfile)
    errorsdf = df[df.Pred != df.Gold].reset_index().sort_values(by=['Error Type'])
    # print(errorsdf.columns)
    errorsdf.drop(columns = errorsdf.columns[0:2], inplace = True)

    print(errorsdf)
    print(errorsdf.to_latex(index=False))

# finderrors("/Users/joshuawork/Desktop/Assignment4/Data/Method2/Predictions/Development/SVM-HECA-HECT-Token_vector.csv")
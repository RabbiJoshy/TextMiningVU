import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from Evaluation import EvaluatePredictions
from Evaluation import EvaluatePredictionsI_NEG

enc = OneHotEncoder(sparse=False, handle_unknown="ignore")

def concattests(testfile1, testfile2):
    test1 = pd.read_pickle(testfile1)
    test2 = pd.read_pickle(testfile2)
    test_features_df= pd.concat([test1, test2], ignore_index=True, sort=False)

    return test_features_df


def extract_features_and_labels(
    trainingfile, testdf, featureselection=["HECT", "Highly Expected Cue"]
):

    print("Extracting")

    features_df = pd.read_pickle(trainingfile)
    test_features_df = testdf

    test_features = []
    for i in range(len(test_features_df)):
        test_features.append([test_features_df[fea][i] for fea in featureselection])
    enc.fit(test_features)
    test_features = enc.transform(test_features)

    training_features = []
    for i in range(len(features_df)):
        training_features.append([features_df[fea][i] for fea in featureselection])
    training_features = enc.transform(training_features)

    targets = list(features_df["Negation_cue"])
    test_targets = list(test_features_df["Negation_cue"])

    print("Done")

    return training_features, test_features, targets, test_targets, test_features_df


def create_classifier(classifier):
    """
    :param classifier:
    :return:
    """

    if classifier == "SVM":
        clf = svm.SVC(max_iter=10000, C = 100000)
    if classifier == "LogReg":
        clf = LogisticRegression(random_state=0)
    if classifier[0] == 'KNN':
        Neighbours = classifier[1]
        clf = KNeighborsClassifier(n_neighbors= Neighbours)

    return clf


def classifydata(trainingfile, testfile, feasel, model):
    """
    :param trainingfile:
    :param testfile:
    :param feasel:
    :param classifier:
    :return:
    """
    classifier = create_classifier(model)

    numfeas = len(feasel)
    print(numfeas)

    X, X_test, y, gold, Features_df = extract_features_and_labels(
        trainingfile, testfile, feasel
    )
    # if model == 'CRF':
    # print(type(X[0]))
    # print(y[0])
    print("fitting")
    classifier.fit(X, y)
    print("Done")

    predictions = classifier.predict(X_test)
    pred = list(predictions)


    # Create File with predictions

    Pred_df = pd.DataFrame()
    Pred_df['Token'] = Features_df['Token']
    Pred_df['Pred'] = pred
    Pred_df['Gold'] = gold
    Errtyp = []
    for i in range(len(Pred_df['Pred'])):
        if (Pred_df['Pred'][i] == 'O') and (Pred_df['Gold'][i] == 'B-NEG'):
            Errtyp.append('FN')
        elif (Pred_df['Pred'][i] == 'B-NEG') and (Pred_df['Gold'][i] == 'O'):
            Errtyp.append('FP')
        elif Pred_df['Gold'][i] == 'I-NEG':
            Errtyp.append('I-NEG')
        else:
            Errtyp.append('True')
    Pred_df['Error Type'] = Errtyp
    Pred_df['trigram'] = Features_df['trigram']

    outfile = "/Users/joshuawork/Desktop/Assignment4/Data/Method2/Predictions/" + model[0] + str(model[1]) + "_" + "-".join(feasel) + ".csv"


    print("creating file")
    Pred_df.to_csv(outfile)
    print("Predictions File Created")

    return pred, gold


def main(trainingfile, testfile1, testfile2, FeaSel, models):
    for model in models:
        testsconcat = concattests(testfile1, testfile2)
        pred, gold = classifydata(trainingfile, testsconcat, FeaSel, model)
        print("Done")
        print("Evaluating")
        EvaluatePredictionsI_NEG(pred, gold)
        print("Done")


main(
    "/Users/joshuawork/Desktop/Assignment4/Data/Method2/train.Preprocessed.pickle",
    "/Users/joshuawork/Desktop/Assignment4/Data/SEM-2012-SharedTask-CD-SCO-test-circle.Preprocessed.pickle",
    "/Users/joshuawork/Desktop/Assignment4/Data/SEM-2012-SharedTask-CD-SCO-test-cardboard.Preprocessed.pickle",
    ["HECT", "HECA"],
    ["SVM"]
)

# import pandas as pd
# test1 = pd.read_pickle("/Users/joshuawork/Desktop/Assignment4/Data/SEM-2012-SharedTask-CD-SCO-test-circle.Preprocessed.pickle")
# test2 = pd.read_pickle("/Users/joshuawork/Desktop/Assignment4/Data/SEM-2012-SharedTask-CD-SCO-test-cardboard.Preprocessed.pickle")
# test_features_df = pd.concat([test1, test2], ignore_index=True, sort=False)
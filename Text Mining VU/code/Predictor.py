import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from Evaluation import EvaluatePredictions

enc = OneHotEncoder(sparse=False, handle_unknown="ignore")


def extract_features_and_labels(
    trainingfile, testfile, featureselection=["HECT", "Highly Expected Cue"]
):

    print("Extracting")

    features_df = pd.read_pickle(trainingfile)
    test_features_df = pd.read_pickle(testfile)

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
        clf = svm.SVC(max_iter=10000)
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

    outfile = testfile.replace(
        os.path.basename(testfile),
        "Predictions/" + model[0] + str(model[1]) + "_" + "-".join(feasel) + ".csv",
    )

    print("creating file")
    Pred_df.to_csv(outfile)
    print("Predictions File Created")

    return pred, gold


def main(trainingfile, testfile, FeaSel, models):
    for model in models:
        pred, gold = classifydata(trainingfile, testfile, FeaSel, model)
        print("Done")
        print("Evaluating")
        EvaluatePredictions(pred, gold)
        print("Done")


main(
    "/Users/joshuawork/Desktop/Assignment4/Data/Method2/train.Preprocessed.pickle",
    "/Users/joshuawork/Desktop/Assignment4/Data/Method2/test.Preprocessed.pickle",
    ["HECT", "HECA", "POS_TAG", "POS"],
    [["KNN",5]]
)

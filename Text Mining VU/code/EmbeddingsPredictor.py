import os
import pandas as pd
# from copy import deepcopy
from sklearn import svm
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
# ['Token_ID', 'Token', 'Lemma', 'POS', 'POS_TAG', 'Dependency_Head',
#        'Dependency_Label', 'idx_sent', 'Negation_cue', 'Token_vector',
#        'next_token', 'next_token_vector', 'prev_token', 'prev_token_vector',
#        'trigram', 'trigram_vectors', 'prev_bigram_list_vectors',
#        'next_bigram_list_vectors', 'HECT', 'HECA', 'HECTNEXT']
argv = [
    '/Users/joshuawork/Desktop/Assignment4/Data/Method2/train.Preprocessed.pickle',
    '/Users/joshuawork/Desktop/Assignment4/Data/Method2/testsets.Preprocessed.pickle',
    ['Token_vector'],
    ['HECA'],
    ['SVM']
]

print(argv[1])
print('Importing Data')
traindf = pd.read_pickle(argv[0])
testdf = pd.read_pickle(argv[1])
print('Data imported')
EmbedFeatures = argv[2]
UnencodedFeatures = argv[3]
feasel = UnencodedFeatures + EmbedFeatures
model = argv[4]


print('Creating Feature Vector')
train_fea_vecs = []
test_fea_vecs = []

print('encoding')
if len(UnencodedFeatures) > 0:
    testunencoded = [[testdf[fea][i] for fea in UnencodedFeatures] for i in range(len(testdf))]
    trainunencoded = [[traindf[fea][i] for fea in UnencodedFeatures] for i in range(len(traindf))]
    enc.fit(testunencoded)
    testencoded = enc.transform(testunencoded)
    trainencoded = enc.transform(trainunencoded)
print('Done encoding')

for i in range(len(traindf['Token_vector'])):
    train_fea_vec = []

    for fea in EmbedFeatures:
        if fea in ['prev_bigram_list_vectors', 'next_bigram_list_vectors', 'HECTNEXT']:
            # print(list(traindf[fea][i][0]))
            train_fea_vec = train_fea_vec + list(traindf[fea][i][0])
        else:
            train_fea_vec = train_fea_vec + list(traindf[fea][i])

    if len(UnencodedFeatures) > 0:
        train_fea_vec = train_fea_vec + list(trainencoded[i])
    train_fea_vecs.append(train_fea_vec)

# print('trainfeavec', len(train_fea_vecs[0]))
# print('trainfeavecss0', train_fea_vecs[0])

for i in range(len(testdf['Token_vector'])):
    test_fea_vec = []

    for fea in EmbedFeatures:
        if fea in ['prev_bigram_list_vectors', 'next_bigram_list_vectors', 'HECTNEXT']:
            test_fea_vec = test_fea_vec + list(testdf[fea][i][0])
        else:
            test_fea_vec = test_fea_vec + list(testdf[fea][i])

    if len(UnencodedFeatures) > 0:
        test_fea_vec = test_fea_vec + list(testencoded[i])
    test_fea_vecs.append(test_fea_vec)

# print('testfealength', len(test_fea_vecs[0]))

targets = list(traindf['Negation_cue'][:])
training_features = train_fea_vecs[:]
test_targets = list(testdf['Negation_cue'])
test_features = test_fea_vecs

print('Done creating vectors')

# for i in range(len(test_features)):
#     if len(test_features[i]) != 300:
#         test_features[i] = [0]  * len(test_fea_vec)
# #
# for i in range(len(training_features)):
#     if len(training_features[i]) != 300:
#         training_features[i] = [0]  * len(test_fea_vec)


#Classifying

#classifier = KNeighborsClassifier(n_neighbors=3)
classifier = svm.SVC(max_iter=10000)
# classifier = LogisticRegression(random_state=0)
print('fitting')
classifier.fit(training_features, targets)
print('Done')
print('Predicting')
predictions = classifier.predict(test_features)
print('Done')
# training_features, test_features, targets, test_targets

# print(test_features[0][0])
# print(len(training_features[0]))
#Error Analysis

# from Evaluation import EvaluatePredictions
# EvaluatePredictions(predictions,test_targets)

Pred_df = pd.DataFrame()
Pred_df['Token']= testdf['Token']
Pred_df['Pred'] = predictions
Pred_df['Gold'] = test_targets
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
Pred_df['trigram']= testdf['trigram']

# outfile = DEV.replace(os.path.basename(DEV), "Predictions/" + model + "-" + "-".join(feasel) + '.pkl')
# Pred_df.to_pickle(outfile)
outfile = argv[1].replace(os.path.basename(argv[1]), "Predictions/Test/" + "".join(model) + "-" + "-".join(feasel) + '.csv')
Pred_df.to_csv(outfile)


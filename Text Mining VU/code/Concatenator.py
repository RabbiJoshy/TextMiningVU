import pandas as pd
import numpy as np

# dev = "/Users/joshuawork/Desktop/Assignment4/Data/Method2/Prep/dev.Preprocessed.pickle"
card = "/Users/joshuawork/Desktop/Assignment4/Data/Method2/cardboard.Preprocessed.pickle"
circ = "/Users/joshuawork/Desktop/Assignment4/Data/Method2/circle.Preprocessed.pickle"
# train = "/Users/joshuawork/Desktop/Assignment4/Data/Method2/Prep/train.Preprocessed.pickle"
def concattests(testfile1, testfile2):
    test1 = pd.read_pickle(testfile1)
    test2 = pd.read_pickle(testfile2)
    test_features_df= pd.concat([test1, test2], ignore_index=True, sort=False)

    return test_features_df

concatdf = concattests(circ, card)

outfile = "/Users/joshuawork/Desktop/Assignment4/Data/Method2/testsets.Preprocessed.pickle"
concatdf.to_pickle(outfile)

# unlistedtrigrams = []
# for i in concatdf['trigram_list_vectors']:
#     unlistedtrigram = [number for vector in i for number in vector]
#     unlistedtrigrams.append(unlistedtrigram)
#
# concatdf['trigrams_unlisted'] = unlistedtrigrams
# concatdf.at[19215,'prev_token_vector'] = concatdf['prev_token_vector'][10]
# concatdf.at[0,'next_token_vector'] = concatdf['next_token_vector'][10]
# concatdf = concatdf.drop(columns = ['trigram_list_vectors'])
#

#
#
# traindf = pd.read_pickle(dev)
#
# unlistedtrigrams = []
# for i in traindf['trigram_list_vectors']:
#     unlistedtrigram = [number for vector in i for number in vector]
#     unlistedtrigrams.append(unlistedtrigram)
#
# traindf['trigrams_unlisted'] = unlistedtrigrams
# traindf.at[13566,'prev_token_vector'] = traindf['prev_token_vector'][10]
# traindf.at[0,'next_token_vector'] = traindf['next_token_vector'][10]
# traindf = traindf.drop(columns = ['trigram_list_vectors'])
#
# outtrain = "/Users/joshuawork/Desktop/Assignment4/Data/Method2/final.train.pickle"
# traindf.to_pickle(outtrain)
#
#
# import pandas as pd
import numpy as np
# # finalpre = pd.read_pickle("/Users/joshuawork/Desktop/Assignment4/Data/Method2/final.testsets.pickle")
# finalpretrain = pd.read_pickle("/Users/joshuawork/Desktop/Assignment4/Data/Method2/train.Preprocessed.pickle")
# # finalpretrain = finalpretrain.sort_values(by=['HECA'])
# HECAAAA = finalpretrain[finalpretrain['Negation_cue'] == 'I-NEG']

# for i in range(len(finalpretrain)):
#     for j in finalpretrain['HECTNEXT'].iloc[i][0]:
#
#         print(type(j))
#
#         if type(j) != type(finalpretrain['next_bigram_list_vectors'].iloc[i][0][0]):
#             print('dad')
#
#     print(len(finalpretrain['next_bigram_list_vectors'].iloc[0][1]))
# len(finalpretrain['next_bigram_list_vectors'].iloc[-1])
#
# finalpretrain.columns
#
# x = len(finalpretrain)-1
# finalpretrain.iat[x, finalpretrain.columns.get_loc('next_token_vector')] = np.random.rand(96)
# print(finalpretrain['next_token_vector'][-1])
#
# finalpretrain.iat[0, finalpretrain.columns.get_loc('prev_token_vector')] = np.random.rand(96)
# print(finalpretrain['prev_token_vector'][0])
#
# print(finalpretrain['next_bigram_list_vectors'][0])

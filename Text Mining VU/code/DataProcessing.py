import spacy
import pandas as pd
import numpy as np
import re
from spacy.tokenizer import Tokenizer
import os


nlp = spacy.load("en_core_web_sm")
print("imports complete")


def LoadAnnotatedData(inputfile):
    """"""

    df_unprocessed = pd.read_csv(
        inputfile,
        header=None,
        encoding="utf-8",
        sep="\t",
        index_col=False,
    )

    # Somehow this fixes the `` issues
    nlp.tokenizer = Tokenizer(nlp.vocab)
    df_unprocessed.columns = [
        "Chapter",
        "Sentence_ID",
        "Token_ID",
        "Token",
        "Negation_cue",
    ]
    df_unprocessed["Sentence_ID_unique"] = df_unprocessed.groupby(
        ["Chapter", "Sentence_ID"]
    ).ngroup()
    regex = r"(Chapter \d+)(\.)"
    df_unprocessed["Token"] = df_unprocessed["Token"].replace("``", '"')
    df_unprocessed["Token"] = df_unprocessed["Token"].replace("`", '"')
    df_unprocessed["Token"] = df_unprocessed["Token"].replace("'", '"')
    df_unprocessed["Token"] = df_unprocessed["Token"].replace("''", '"')

    token_sent = []
    for i in range(len(set(df_unprocessed["Sentence_ID_unique"]))):
        if i % 500 == 0:
            print("loop", i, "/3700?")
        sent = " ".join(df_unprocessed[df_unprocessed.Sentence_ID_unique == i]["Token"])
        cue = list(
            df_unprocessed[df_unprocessed.Sentence_ID_unique == i]["Negation_cue"]
        )
        fixed_sent = re.sub(regex, r"\1", sent)
        doc = nlp(fixed_sent)

        token_sent.append((doc, cue))

    return token_sent


def create_parsed_df(token_sentences):
    """"""

    listOfDicts = []
    sent_idx = 0
    for (
        doc,
        cues,
    ) in token_sentences:  # for each tokenized/processed sentence in the list
        for (
            sent
        ) in (
            doc.sents
        ):  # take each sentence, since we only have 1 sentence it loops 1 time
            for i, word in enumerate(sent):  # go trough each token/word

                if word.head == word:  # reset counter
                    head_idx = 0
                else:  # otherwise calculate idx
                    head_idx = word.head.i - sent[0].i + 1
                dict_parser_output = (
                    {}
                )  # make dictionary and fill it values, as showed in the report appendix II
                dict_parser_output["idx_sent"] = sent_idx
                dict_parser_output["Token_ID"] = i
                dict_parser_output["Token"] = word.text
                dict_parser_output["Lemma"] = word.lemma_
                dict_parser_output["POS"] = word.pos_
                dict_parser_output["POS_TAG"] = word.tag_
                dict_parser_output["Dependency_Head"] = head_idx
                dict_parser_output["Dependency_Label"] = word.dep_
                dict_parser_output["Negation_cue"] = cues[i]
                dict_parser_output["Token_vector"] = word.vector

                listOfDicts.append(dict_parser_output)  # append to list
        sent_idx += 1

    columns_ = [
        "Token_ID",
        "Token",
        "Lemma",
        "POS",
        "POS_TAG",
        "Dependency_Head",
        "Dependency_Label",
        "idx_sent",
        "Negation_cue",
        "Token_vector",
    ]

    # filler = np.random.rand(96)

    df_output_parser = pd.DataFrame(listOfDicts, columns=columns_)
    df_output_parser["next_token"] = df_output_parser.Token.shift(fill_value="Boundary")
    df_output_parser["next_token_vector"] = df_output_parser.Token_vector.shift(
        -1, fill_value= None
    )
    df_output_parser["prev_token"] = df_output_parser.Token.shift(-1, fill_value="Boundary")
    df_output_parser["prev_token_vector"] = df_output_parser.Token_vector.shift(
        fill_value= None
    )

    x = len(df_output_parser)-1

    df_output_parser.iat[x, df_output_parser.columns.get_loc('next_token_vector')] = np.random.rand(96)
    df_output_parser.iat[x, df_output_parser.columns.get_loc('prev_token_vector')] = np.random.rand(96)
    df_output_parser.iat[0, df_output_parser.columns.get_loc('prev_token_vector')] = np.random.rand(96)
    df_output_parser.iat[0, df_output_parser.columns.get_loc('next_token_vector')] = np.random.rand(96)

    df_output_parser["trigram"] = (
        df_output_parser.Token.shift()
        + " "
        + df_output_parser.Token
        + " "
        + df_output_parser.Token.shift(-1)
    )
    df_output_parser.loc[0, "trigram"] = df_output_parser.trigram[1]
    df_output_parser.loc[
        len(df_output_parser) - 1, "trigram"
    ] = df_output_parser.trigram[len(df_output_parser) - 2]

    df_output_parser["trigram_list_tokens"] = df_output_parser.apply(
        lambda x: [x.next_token, x.Token, x.prev_token], axis=1
    )

    df_output_parser["trigram_list_tokens"].iloc[0] = df_output_parser[
        "trigram_list_tokens"
    ].iloc[1]
    df_output_parser["trigram_list_tokens"].iloc[-1] = df_output_parser[
        "trigram_list_tokens"
    ].iloc[-2]

    df_output_parser["trigram_list_vectors"] = df_output_parser.apply(
        lambda x: [x.next_token_vector, x.Token_vector, x.prev_token_vector], axis=1
    )

    df_output_parser["trigram_list_vectors"].iloc[0] = df_output_parser[
        "trigram_list_vectors"
    ].iloc[1]
    df_output_parser["trigram_list_vectors"].iloc[-1] = df_output_parser[
        "trigram_list_vectors"
    ].iloc[-2]

    unlistedtrigrams = []
    for i in df_output_parser['trigram_list_vectors']:
        unlistedtrigram = [number for vector in i for number in vector]
        unlistedtrigrams.append(unlistedtrigram)
    df_output_parser['trigram_vectors'] = unlistedtrigrams
    df_output_parser = df_output_parser.drop(columns=['trigram_list_vectors'])

    df_output_parser["prev_bigram"] = (
        df_output_parser.Token.shift() + " " + df_output_parser.Token
    )
    df_output_parser.loc[0, "prev_bigram"] = df_output_parser.prev_bigram[1]

    df_output_parser["prev_bigram_list_tokens"] = df_output_parser.apply(
        lambda x: [x.next_token, x.Token], axis=1
    )
    df_output_parser["prev_bigram_list_tokens"].iloc[0] = df_output_parser[
        "prev_bigram_list_tokens"
    ].iloc[1]

    # Create list of vectors of the corresponding tokens bigram based on current token and previous token
    df_output_parser["prev_bigram_list_vectors"] = df_output_parser.apply(
        lambda x: [x.prev_token_vector + x.Token_vector], axis=1
    )
    df_output_parser["prev_bigram_list_vectors"].iloc[0] = df_output_parser[
        "prev_bigram_list_vectors"
    ].iloc[1]

    # unlistedprevbigrams = []
    # for i in df_output_parser['prev_bigram_list_vectors']:
    #     unlistedprevbigram = [number for vector in i for number in vector]
    #     unlistedprevbigrams.append(unlistedprevbigram)
    # df_output_parser['prev_bi_vector'] = unlistedprevbigrams

    # df_output_parser = df_output_parser.drop(columns=['prev_bigram_list_vectors'])


    df_output_parser = df_output_parser.drop(columns=["prev_bigram_list_tokens"])
    df_output_parser = df_output_parser.drop(columns=["prev_bigram"])



    # Create string based bigram based on current token and next token
    df_output_parser["next_bigram"] = (
        df_output_parser.Token + " " + df_output_parser.Token.shift(-1)
    )
    df_output_parser.loc[
        len(df_output_parser) - 1, "next_bigram"
    ] = df_output_parser.next_bigram[len(df_output_parser) - 2]

    # Create list of tokens bigram based on current token and next token
    df_output_parser["next_bigram_list_tokens"] = df_output_parser.apply(
        lambda x: [x.Token, x.next_token], axis=1
    )
    df_output_parser["next_bigram_list_tokens"].iloc[-1] = df_output_parser[
        "next_bigram_list_tokens"
    ].iloc[-2]

    # Create list of vectors of the corresponding tokens bigram based on current token and next token
    df_output_parser["next_bigram_list_vectors"] = df_output_parser.apply(
        lambda x: [x.Token_vector + x.next_token_vector], axis=1
    )
    df_output_parser["next_bigram_list_vectors"].iloc[-1] = df_output_parser[
        "next_bigram_list_vectors"
    ].iloc[-2]

    # unlistednextbigrams = []
    # for i in df_output_parser['next_bigram_list_vectors']:
    #     unlistednextbigram = [number for vector in i for number in vector]
    #     unlistednextbigrams.append(unlistednextbigram)
    # df_output_parser['next_bi_vector'] = unlistednextbigrams
    #
    # df_output_parser = df_output_parser.drop(columns=['next_bigram_list_vectors'])


    df_output_parser = df_output_parser.drop(columns=["next_bigram_list_tokens"])
    df_output_parser = df_output_parser.drop(columns=["next_bigram"])
    df_output_parser = df_output_parser.drop(columns=["trigram_list_tokens"])

    return df_output_parser

def CreatePosVocab(dataframe, wordcolumn, POScolumn):
    PosVocab = set()
    for i, word in enumerate(dataframe[wordcolumn]):
        if dataframe[POScolumn][i] in ["NOUN", "ADV", "VERB", "ADJ", "VBN"]:
            PosVocab.add(word)

    return PosVocab

def CreateHighExpBools(dataframe, PosVocab, FrequentSet=("n't", "never", "no", "none", "nor", "not", "nothing", "without"), wordcolumn = 'Token'):

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    HighExpCueBool = []
    HighExpAffixBool = []

    for word in dataframe[wordcolumn]:
        word = word.lower()
        # root2 = word[2:]
        # lemma = " ".join([token.lemma_ for token in nlp(root2)])
        # print(lemma)
        if word[:2] in ["im", "ir", "no", "un", "in"]:
            # print(word)
            if word[2:] in PosVocab:
                # print(word[2:])
                 HighExpAffixBool.append(1)
            # elif lemma in PosVocab:
            #      # print(lemma)
            #      HighExpAffixBool.append(1)
            else:
                HighExpAffixBool.append(0)
        elif word[:3] in ["dis"] and word[3:] in PosVocab:
            HighExpAffixBool.append(1)
        elif word.endswith(("less", "lessly", "lessness")) and word not in [
            "unless",
            "bless",
        ]:  # and word[-4:] in Vocab:
            HighExpAffixBool.append(1)
        else:
            HighExpAffixBool.append(0)

        if word in FrequentSet:
            HighExpCueBool.append(1)
        else:
            HighExpCueBool.append(0)

    print(sum(HighExpAffixBool))

    # HECT: Highly expected Cue Token
    dataframe["HECT"] = HighExpCueBool
    # HECA: Higly expected Cue Affix
    dataframe["HECA"] = HighExpAffixBool

    return dataframe


def parse(unprocessed_file):

    print("Loading File into SpaCy")
    loadedsents = LoadAnnotatedData(unprocessed_file)
    print("Building Features DataFrame")
    parsed_dataframe = create_parsed_df(loadedsents)

    return parsed_dataframe


def main(trainfile, testfile):

    print("Parsing Training Data")
    parsedtraindf = parse(trainfile)
    trainpv = CreatePosVocab(parsedtraindf, wordcolumn = "Token", POScolumn = "POS")
    print("Parsing Test Data")
    parsedtestdf = parse(testfile)
    testpv =  CreatePosVocab(parsedtestdf, wordcolumn = "Token", POScolumn = "POS")
    posvocab = trainpv.union(testpv)

    print("Adding Expectancy Bool")
    train_features_df = CreateHighExpBools(parsedtraindf, posvocab)
    test_features_df = CreateHighExpBools(parsedtestdf, posvocab)

    print("Adding Composite Expectancy Bools")

    train_features_df["HECTNEXT"] = train_features_df.apply(
        lambda x: [x.HECT * 100 * x.next_token_vector], axis=1
    )
    train_features_df["HECTNEXT"].iloc[-1] = train_features_df[
        "HECTNEXT"
    ].iloc[-2]
    test_features_df["HECTNEXT"] = test_features_df.apply(
        lambda x: [x.HECT * 100 * x.next_token_vector], axis=1
    )
    test_features_df["HECTNEXT"].iloc[-1] = test_features_df[
        "HECTNEXT"
    ].iloc[-2]

    trainoutfile = trainfile.replace(os.path.basename(trainfile), "Method2/circle.Preprocessed.pickle")
    testoutfile = testfile.replace(os.path.basename(testfile), "Method2/cardboard.Preprocessed.pickle")
    print("PreProcessed Train File Creating")
    train_features_df.to_pickle(trainoutfile)
    print("PreProcessed Train File Created")
    test_features_df.to_pickle(testoutfile)
    print("PreProcessed Test File Created")

    return train_features_df, test_features_df


main("../Data/SEM-2012-SharedTask-CD-SCO-test-circle.txt", "../Data/SEM-2012-SharedTask-CD-SCO-test-cardboard.txt")

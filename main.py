import json
import numpy as np
import pandas as pd
import os
import sumy
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from rouge import Rouge
import rouge


def main():
    # YOUR CODE HERE
    df = pd.read_json(
        'data/essay-prompt-corpus.json')
    dk = pd.read_csv(
        r"data\train-test-split.csv", sep=';')
    dk['ID'] = dk['ID'].str.replace('essay', '')
    tr = dk[dk['SET'] == "TRAIN"]
    te = dk[dk['SET'] == "TEST"]
    tes = te['ID'].tolist()
    tes = list(map(int, tes))
    tra = tr['ID'].tolist()
    tra = list(map(int, tra))
    test = df[df['id'].isin(tes)]
    train = df[df['id'].isin(tra)]

    def summarizator(text):
        SENTENCES_COUNT = 2
        language = 'english'

        sumario_summy = []
        parser = PlaintextParser.from_string(text, Tokenizer(language))
        stemmer = Stemmer(language)
        summarizer = LexRankSummarizer(stemmer)
        summarizer.stop_words = get_stop_words(language)
        summary = []

        for sentence in summarizer(parser.document, SENTENCES_COUNT):
            summary.append(str(sentence))

        summary = ' '.join(summary)
        sumario_summy.append(summary)

        return ' '.join(sumario_summy)

    test["Sumy"] = test["text"].apply(summarizator)
    p = test['prompt']
    s = test['Sumy']
    evaluator = rouge.Rouge(metrics=['rouge-l'])
    scores = evaluator.get_scores(s, p, avg=True)
    print(scores)
    w = pd.DataFrame(columns=['id', 'prompt'])
    w[['id', 'prompt']] = test[['id', 'Sumy']]
    w.to_json("sample-prediction.json", orient='records')
    print("it works!")
    pass


if __name__ == '__main__':
    main()

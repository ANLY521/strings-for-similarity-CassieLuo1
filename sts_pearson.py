import scipy.stats
from scipy.stats import pearsonr
import argparse
from nltk import word_tokenize
from util import parse_sts
from sts_nist import symmetrical_nist
import warnings
warnings.filterwarnings("ignore")
from difflib import SequenceMatcher
from nltk.translate.nist_score import sentence_nist
from nltk.translate.bleu_score import SmoothingFunction,sentence_bleu
from nltk.metrics.distance import edit_distance
import numpy as np
def main(sts_data):
    """Calculate pearson correlation between semantic similarity scores and string similarity metrics.
    Data is formatted as in the STS benchmark"""

    # TODO 1: read the dataset; implement in util.py
    texts, labels = parse_sts(sts_data)
    print(f"Found {len(texts)} STS pairs")

    # TODO 2: Calculate each of the the metrics here for each text pair in the dataset
    # HINT: Longest common substring can be complicated. Investigate difflib.SequenceMatcher for a good option.
    score_types = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Edit Distance"]

# NIST
    scores_nist = []
    data_s = zip(labels, texts)

    for label,text_pair in data_s:
        score_nist= 0
        i=len(text_pair[0].split())
        score_nist = symmetrical_nist(text_pair)

        scores_nist.append(score_nist)

    NISTscore = pearsonr(labels, scores_nist)

# BLEU
    BLEU_scores=[]
    for text in texts:
        t1, t2 = text
        t1_toks = word_tokenize(t1.lower())
        t2_toks = word_tokenize(t2.lower())

        try:
            bleu1 = sentence_bleu([t2_toks, ], t1_toks, smoothing_function=SmoothingFunction().method0)
        except:
            bleu1 = 0

        try:
            bleu2 = sentence_bleu([t1_toks, ], t2_toks, smoothing_function=SmoothingFunction().method0)
        except:
            bleu2 = 0

        bleu = bleu1 + bleu2

        BLEU_scores.append(bleu)

    BLEUscore = pearsonr(labels,BLEU_scores)
                                     #score_bleu = sentence_bleu([text_pair[0].lower().strip().split()], text_pair[1].lower().strip().split())
#Word Error Rate
    WER_scores = []
    def wer(r, h):
        """
        Calculation of WER with edit distance.

        Parameters
        ----------
        r : list
        h : list
        """
        # initialisation
        d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8)
        d = d.reshape((len(r) + 1, len(h) + 1))
        for i in range(len(r) + 1):
            for j in range(len(h) + 1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i

        # computation
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        return(d[len(r)][len(h)] / len(r) + d[len(r)][len(h)] / len(h)) / 2

    for text in texts:
        t1, t2 = text
        t1_toks = word_tokenize(t1.lower())
        t2_toks = word_tokenize(t2.lower())
        w = wer(t1_toks, t2_toks)
        WER_scores.append(w)

    WERscore = pearsonr(WER_scores, labels)

# Longest common string
    LCS_scores=[]
    for text in texts:
        t1, t2 = text
        #t1_toks= t1.lower().strip().split() #0.449
        #t2_toks = t2.lower().strip().split()
        t1_toks = word_tokenize(t1.lower())
        t2_toks = word_tokenize(t2.lower())
        s = SequenceMatcher(None, t1_toks, t2_toks)
        Match1=s.find_longest_match(0, len(t1_toks), 0, len(t2_toks))
        LCS_scores.append(Match1.size)


    LCSscore = pearsonr(LCS_scores, labels)


#Edit Distance

    ED_scores = []

    for text in texts:
        t1, t2 = text
        d = edit_distance(t1, t2)
        ED_scores.append(d)

    EDscore = pearsonr(ED_scores, labels)


    #TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README
    print(f"Semantic textual similarity for {sts_data}\n")
    scores = [NISTscore[0], BLEUscore[0], WERscore[0], LCSscore[0], EDscore[0]]
    score_types = zip(score_types, scores)
    for metric_name,score in score_types:
        print(f"{metric_name} correlation: {score:.03f}")

    # TODO 4: Complete writeup as specified by TODOs in README (describe metrics; show usage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)


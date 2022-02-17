Semantic textual similarity using string similarity
---------------------------------------------------

This project examines string similarity metrics for semantic textual similarity.
Though semantics go beyond the surface representations seen in strings, some of these
metrics constitute a good benchmark system for detecting STS.

Data is from the [STS benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).

**TODO:**
Describe each metric in ~ 1 sentence



NIST: When a correct n-gram is found, the rarer that n-gram is, the more weight it will be given.

BLEU: Compare a candidate translation against multiple reference translations and calculate n-gram precisions.

Word Error Rate: Add up the substitutions, insertions, and deletions that occur to transform candidate words into 
reference words, and divide that number by the total number of candidate words.

Longest Common Substring: Find the longest string that is a substring of both strings.

Edit Distance: Count the minimum number of operations required to transform one string into the other.



**TODO:** Fill in the correlations. Expected output for DEV is provided; it is ok if your actual result
varies slightly due to preprocessing/system difference, but the difference should be quite small.

**Correlations:**

Metric | Train | Dev | Test 
------ | ----- | --- | ----
NIST | 0.496 | 0.593 | 0.475
BLEU | 0.371 | 0.433 | 0.353
WER | -0.353 | -0.452| -0.358
LCS | 0.363 | 0.468| 0.330
Edit Dist | 0.033 | -0.175| -0.039

**TODO:**
Show usage of the homework script with command line flags (see example under lab, week 1).
 
python sts_pearson.py --sts_data stsbenchmark/sts-dev.csv



python sts_pearson.py --sts_data stsbenchmark/sts-train.csv


python sts_pearson.py --sts_data stsbenchmark/sts-test.csv
## lab, week 1: sts_nist.py

Calculates NIST machine translation metric for sentence pairs in an STS dataset.

Example usage:

`python sts_nist.py --sts_data stsbenchmark/sts-dev.csv`

## lab, week 2: sts_tfidf.py

Calculate pearson's correlation of semantic similarity with TFIDF vectors for text.

## homework, week 1: sts_pearson.py

Calculate pearson's correlation of semantic similarity with the metrics specified in the starter code.
Calculate the metrics between lowercased inputs and ensure that the metric is the same for either order of the 
sentences (i.e. sim(A,B) == sim(B,A)). If not, use the strategy from the lab.
Use SmoothingFunction method0 for BLEU, as described in the nltk documentation.

Run this code on the three partitions of STSBenchmark to fill in the correlations table above.
Use the --sts_data flag and edit PyCharm run configurations to run against different inputs,
 instead of altering your code for each file.
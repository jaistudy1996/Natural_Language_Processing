import nltk, csv
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia

def main():
    #Open file
    with open('train.csv', newline='') as csvfile:
        test = csv.reader(csvfile)
        for row in test:
            comment = row[2]
            print(row[2])
            ss = sia.polarity_scores(comment)
            for k in sorted(ss):
                print('{0}: {1}, '.format(k, ss[k]), end='')
            print()
            

main()

import nltk, csv
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

def main():
    #Open file
    with open('train.csv', newline='') as csvfile:
        test = csv.reader(csvfile)
        sid = SIA()
        
        for row in test:
            comment = row[2]
            ss = sid.polarity_scores(comment)
            #print("Comment:", row[2])
            #print("Value:", row[0])
            #Get pos/neg score for comment
            #for k in sorted(ss):
            #    print('{0}: {1}, '.format(k, ss[k]), end='')
            #print("\n")

            #Create Dictionary of negitive comments
            negitive = {}
            if ss['compound'] < 0:
                negitive[row[2]] = ss['compound']
            
            for key,value in negitive.items():
                print(key, value, )

main()


import nltk, csv, prettytable, math
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

def main():
    #Open file
    table = prettytable.PrettyTable(field_names=["FLAG_Original", "Rounded Compund", "Compound", "Negative_value", "Sentence"])
    with open('train.csv', newline='') as csvfile:
        test = csv.reader(csvfile)
        sid = SIA()
        counter = 0
        for row in test:
            if(counter>100):
                break
            comment = row[2]
            #print(row[2])
            ss = sid.polarity_scores(comment)
            #for k in sorted(ss):
                # print('{0}: {1}, '.format(k, ss[k]), end='')
                # print(k, ss[k])
            table.add_row([row[0], math.floor(ss["compound"]), ss["compound"], ss["neg"], row[2][:100]])
            #print("\n")
            counter += 1      
    print(table)

main()

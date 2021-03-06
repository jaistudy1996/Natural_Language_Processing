import nltk, csv, prettytable, math
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

def main():
    #Open file
    table = prettytable.PrettyTable(field_names=["Match", "FLAG_Original", "Rounded Compund", "Compound", "Negative_value", "Sentence"])
    with open('train.csv', newline='') as csvfile:
        test = csv.reader(csvfile)
        sid = SIA()
        counter = 0
        trueCount = 0
        falseCount = 0
        for row in test:
            if(counter>100):
                break
            comment = row[2]
            #print(row[2])
            ss = sid.polarity_scores(comment)
            #for k in sorted(ss):
                # print('{0}: {1}, '.format(k, ss[k]), end='')
                # print(k, ss[k])
            value = 0
            match = ""
            if(ss["compound"] < 0):
                if(ss["compound"] < -0.50):
                    value = 1   
            if(ss["compound"] > 0):
                value = 0
            table.add_row([int(row[0]) == value, int(row[0]), value, ss["compound"], ss["neg"], row[2][:100]])
            if(int(row[0]) == value):
                trueCount += 1
            else:
                falseCount += 1
            #print("\n")
            counter += 1      
    table.add_row([trueCount, falseCount, "=====", "=====", "=====", "====="])
    table.add_row(["Total Accuracy: ", (trueCount/(trueCount+falseCount))*100, "=====", "=====", "=====", "====="])
    print(table)

main()

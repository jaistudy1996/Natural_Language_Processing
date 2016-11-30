import nltk, csv, prettytable, math
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

from nltk.chunk import *
from nltk.chunk.util import *
from nltk.chunk.regexp import *
from nltk import Tree

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

            ss = sid.polarity_scores(comment)
            #for k in sorted(ss):
                # print('{0}: {1}, '.format(k, ss[k]), end='')
                # print(k, ss[k])
            value = 0
            match = ""
            if(ss["compound"] < 0):
                if(ss["compound"] < -0.50):
                    #Start noun-chunking (check for false positives)
                    
                    #first tokenize comment and add POS tags
                    tokenizedComment = word_tokenize(comment)
                    negComment = nltk.pos_tag(tokenizedComment)

                    #set pattern to check for pronouns
                    pattern = "NP:{<PRP>}"
                    NPChunker = nltk.RegexpParser(pattern)

                    result = NPChunker.parse(negComment)
                    #print (result)
                    for n in result:
                        if isinstance(n, nltk.tree.Tree):
                            if n.label() == 'PRN':
                                value = 1
                            else:
                                value = 0
                    
            if(ss["compound"] > 0):
                value = 0
                
                #Start noun-chunking (check false negitives)
                    
                #first tokenize comment and add POS tags
                #tokenizedComment = word_tokenize(comment)
                #negComment = nltk.pos_tag(tokenizedComment)

                #set pattern to check for pronouns
                #pattern = "NP:{<PRP>}"
                #NPChunker = nltk.RegexpParser(pattern)

                #result = NPChunker.parse(negComment)
                #print (result)
                
            table.add_row([int(row[0]) == value, int(row[0]), value, ss["compound"], ss["neg"], row[2][:100]])
            if(int(row[0]) == value):
                trueCount += 1
            else:
                falseCount += 1

            counter += 1
    table.add_row([trueCount, falseCount, "=====", "=====", "=====", "====="])
    table.add_row(["Total Accuracy: ", (trueCount/(trueCount+falseCount))*100, "=====", "=====", "=====", "====="])
    print(table)

            
   

main()

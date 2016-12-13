# Authors: Jayant Arora, Marie Hilpl, Robert Elliot
# main file for the project
import nltk, csv, prettytable, math, re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

from nltk.chunk import *
from nltk.chunk.util import *
from nltk.chunk.regexp import *
from nltk import Tree

#leaves() get_terms() and normalize() have code from this site  http://alexbowe.com/au-naturale/
def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
        yield subtree.leaves()
def get_terms(tree):
    for leaf in leaves(tree):
        term = [ normalise(word) for word, tag in leaf]
        yield term
def normalise(word):
    """Normalises words to lowercase """
    word = word.lower()
    return word

def main():
    #Open file
    table = prettytable.PrettyTable(field_names=["Match", "FLAG_Original", "FLAG_Predicted", "Compound", "Negative_value", "Sentence"])
    with open('train.csv', newline='') as csvfile:
        test = csv.reader(csvfile)
        sid = SIA()
        counter = 0
        trueCount = 0
        falseCount = 0
        for row in test:
            #if(counter>100):
            #    break
            comment = row[2]

            #Make comment all lower case
            if (comment == comment.upper()):
                comment = comment.lower()

            ss = sid.polarity_scores(comment)
            #for k in sorted(ss):
                # print('{0}: {1}, '.format(k, ss[k]), end='')
                # print(k, ss[k])
            value = 0
            match = ""
            if(ss["compound"] < -0.5):
                #check for username
                if( re.match('@.*', comment)):
                    value = 1
                    continue
                        
                #Start noun-chunking (check for false positives)
                    
                #first tokenize comment and add POS tags
                tokenizedComment = word_tokenize(comment)
                negComment = nltk.pos_tag(tokenizedComment)

                for n in comment.split():
                    if(n.lower() == "you" or n.lower() == "your" or n.lower() == "yourself" or n.lower() == "you're"):
                        value = 1
                        #print("After: " + n)

                #set pattern to check for pronouns
                #pattern = "NP:{<PRP>|<PRP$>}"
                #NPChunker = nltk.RegexpParser(pattern)

                #result = NPChunker.parse(negComment)
                #print (result)
                #for n in result:
                 #   if (isinstance(n, nltk.tree.Tree)):
                        #get list of pronouns (including tag)
                    #      noun_phrase_words = get_terms(result)
                        #flag comment if pronoun matches specific pronouns
                  #     for term in noun_phrase_words:
                    #        for word in term:
                        #           print("Word Before: " + word)
                        #          if(word == 'you' or word == 'your' or word == 'you\'re' or word == 'yourself'):
                        #             value = 1
                        #            print("Catch: " + word)
                            #       else:
                            #          value = 0
                        
                    
            if(ss["compound"] > -0.5):
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

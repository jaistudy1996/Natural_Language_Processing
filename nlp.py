# Authors: Jayant Arora, Marie Hilpl
# This file will have most of the nltk code

import nltk, csv
from nltk.tokenize import word_tokenize

def main():
	print("NLTK imported successfully.")
	with open('test.csv', newline='') as csvfile:
		test = csv.reader(csvfile)
		for row in test:
			#print(row[2])
			#Tokenize each comment
			tokens = nltk.word_tokenize(row[2])
			#Assign POS tag to each word in the comment
			pos = nltk.pos_tag(tokens)

main()	

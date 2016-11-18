# Authors: Jayant Arora, Marie Hilpl, Robert Elliot
# This file will have most of the nltk code

import nltk, csv
from nltk.tokenize import word_tokenize

def main():
	print("NLTK imported successfully.")
	with open('train.csv', newline='') as csvfile:
		test = csv.reader(csvfile)
		counter = 0
		for row in test:
			if(counter>1):
				break
			#print(row[2])
			#Tokenize each comment
			tokens = nltk.word_tokenize(row[2])
			#Assign POS tag to each word in the comment
			pos = nltk.pos_tag(tokens)
			counter += 1
			print(tokens)
			print(pos)
			print(row[0])

main()	

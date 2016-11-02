# Authors: Jayant Arora, Marie Hilpl
# This file will have most of the nltk code

import nltk, csv


def main():
	print("NLTK imported successfully.")
	with open('test.csv', newline='') as csvfile:
	test = csv.reader(csvfile)
	for row in test:
		print(row[2])


main()	

	

import random
random.seed(123) #Step 1.0
import codecs
import re
from nltk.stem.porter import PorterStemmer
import gensim

f = codecs.open("pg3300.txt", "r", "utf-8") #Step 1.1

#!READ! I have documented the program, my problem-solving and my interpretations in the comments next to each line. Also, I have organized print-statements that represent all the useful data for the assignments, so running the file will neatly organize it.

paragraph = ""  #Step 1.2 and 1.3, line 11-21
docPartition = [] #Remark 1 says to keep the original paragraphs
docPartitionWords = [] #So therefore we have this line for separated terms

for line in f:  #Cycle through every line in the document
    if line.isspace(): #If we reach a newline(where we separate by paragraphs)
        if paragraph != "" and 'gutenberg' not in paragraph.lower(): #This if-statement is to make sure that we don't add empty lines as stand-alone paragraphs, as well as filtering out all instances of 'Gutenberg'
            docPartition.append(paragraph)
        paragraph = "" #Reset the constructed paragraph as it is time to construct a new one
    else:
        paragraph += line #Adding a line onto the given paragraph, constructing it until we reach an empty line to meet the if-condition above.

stemmer = PorterStemmer()
for p in range(0, len(docPartition)):
    docPartitionWords.append(re.findall(r"[\w']+", docPartition[p].lower())) # Step 1.4 and 1.5, splits the paragraphs into a list of words in lower-case, without punctuation, using python regular expressions

for q in range(0, len(docPartition)):
    for r in range(0, len(docPartitionWords[q])):
        docPartitionWords[q][r] = stemmer.stem(docPartitionWords[q][r]) # Step 1.6, using the nltk library to stem each word in the file, saving some space in the corpus



f.close() #I'm not sure if this is necessary, but the program still ran, and I wanted to be on the safe side

c = codecs.open("common-english-words.txt", "r", "utf-8")
dictionary = gensim.corpora.Dictionary(docPartitionWords) #Step 2.1(listed twice in assignment), line 36-46
stop_ids = []
stopwords = []

for line in c:
    stopwords = line.split(",") #Separating each stopword

stop_ids = [dictionary.token2id[stopword] for stopword in stopwords if stopword in dictionary.token2id] #Building a list of stopword-ids relating to the dictionary

dictionary.filter_tokens(stop_ids) #Filtering out stopwords through our list of stopword-ids
c.close()

bagsOfWords = [dictionary.doc2bow(word) for word in docPartitionWords] #Step 2.2, building a bag of words using gensim


tfidf_model = gensim.models.TfidfModel(bagsOfWords) #Step 3.1, using our previously saved list of paragraphs
tfidf_corpus = tfidf_model[bagsOfWords] #Step 3.2, applying a transformation to the vector
matrixSimilarity = gensim.similarities.MatrixSimilarity(bagsOfWords) #Step 3.3, constructing a MatrixSimiliarity object

lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100) #Step 3.4, constructing a simliarity matrix for an LSI-model
lsi_corpus = lsi_model[tfidf_corpus]
lsiMatrixSimilarity = gensim.similarities.MatrixSimilarity(lsi_corpus)

print("Lsi-model topics for the corpus:")
print(lsi_model.show_topics()) #Step 3.5, topic interpretations to be typed as a program comment in the next line
#The first three topics seems to be about about the finances behind employing manual workers, likely construction workers or the like that work on land. It also seems to refer to historical customs regarding employment and payment.

queryString = "What is the function of money?" #Step 4.1, line 62-66. Repeating the procedure from earlier to preprocess the query and convert it into a BOW expression
query = re.findall(r"[\w']+", queryString.lower())
for j in range(0, len(query)):
    query[j] = stemmer.stem(query[j])
query = dictionary.doc2bow(query)

queryTfIdf = tfidf_model[query] #Step 4.2. From the print-statements below, I deduced that the TF-IDF weights are (money: 0.31, function: 0.95)
print("\nTF-IDF weights:", queryTfIdf)
print("Found TF-IDF terms:")
print(dictionary[806])
print(dictionary[1130])

doc2similarity = enumerate(matrixSimilarity[queryTfIdf]) #Step 4.3, line 73-84. Applying the query to the matrix similarity calculation. My code, using the query 'How taxes influence Economics?' ommited paragraph 379, instead giving paragraph 1882, but it still kept paragraph 2003 and 2013
print("\nDocuments similarity with TF-IDF-model:")
print(sorted(doc2similarity, key=lambda kv: -kv[1])[:3]) #Sorting documents by similarity values
relevantParagraphs = [49, 676, 987] # The found paragraphs as indices

print("\nTF-IDF-model top 3 relevant paragraphs:") #The following lines is just truncating the paragraphs to 5 lines
for i in relevantParagraphs:
    lines = docPartition[i].split("\n")
    counter = 0
    print("Paragraph:", i + 1)
    for r in range(0, len(lines)):
        if counter < 5:
            print(lines[r])
            counter += 1
    print("\n")

queryLsi = lsi_model[queryTfIdf] #Step 4.4, through the rest of this file. In the previous step, testing the query 'How taxes influence Economics?' gave one wrong document representation, but the topics representation was correct here, being topics 3, 5 and 9
print("\nLsi-model topics:")
print(sorted(queryLsi, key=lambda kv: -abs(kv[1]))[:3]) #Sorting weights by absolute values"
topics = [4, 16, 61] # The found topics in indices
print("\nLsi-model top 3 relevant topics with terms:")
for topic in topics:
    print("Topic:", topic)
    print(lsi_model.show_topic(topic))

doc2similarity = enumerate(lsiMatrixSimilarity[queryLsi])
print("\nDocument similarity with Lsi-model:")
print(sorted(doc2similarity, key=lambda kv: -kv[1])[:3]) #Sorting documents by similarity values
relevantParagraphs = [987, 1002, 1003] # The found paragraphs as indices. We can see that paragraph index 987 is found again. This was also the case when trying the query 'How taxes influence Economics?', but sharing the "replaced" paragraph which I described earlier in the document

print("\nLsi-model top 3 relevant paragraphs:") #Parsing like with the TF-IDF-model
for k in relevantParagraphs:
    lines = docPartition[k].split("\n")
    counter = 0
    print("Paragraph:", k + 1)
    for s in range(0, len(lines)):
        if counter < 5:
            print(lines[s])
            counter += 1
    print("\n")

print("End of program")


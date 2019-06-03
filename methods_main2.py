import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import codecs
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import warnings
from nlp_methods import nlpMethods
from  collections import Counter
import string
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
warnings.filterwarnings('ignore')
stop = set(stopwords.words('english'))
stop.add("gt")
stop.add("et")
stop.add("al")

class mainMethods:

    def __init__(self, path):
        # tell the class where the directory is
        self.path = path
        # dictionary to hold all of our accroynms
        self.accronymList = {}
        # dataFrame that holds the information of the class
        self.df = pd.DataFrame()
        # dictionary to hold the phrases present
        self.phraseDict = {}
        # keep a record of the frequency of occurence of terms
        self.termsDict = {}
        # keep a records of the tfidf vectorizer
        self.tfidf_vectoriser = TfidfVectorizer()
        # # keep a records of the tfidf_matrix
        self.tfidf_matrix = pd.DataFrame()
        # store a count of all terms in the corpus
        self.termCorpusCount = {}
        # a list of all of the recorded dfs
        self.allTermIDFList = pd.DataFrame()
        # store a reference to words and dict locations
        self.wordStore = {}
        # pattern to change text area targeted
        self.pattern = "REFERENCES"

#####################################################################
# keyPhraseExtraction
    def computePageRank(self, graph):
        # loop over the values and create the k value for the nodes
        inoutDict = {}
        for a in graph.nodes():
            #print(a)
            #print(graph[a])
            for k , v in graph[a].items():
                if k in inoutDict:
                    inoutDict[k] += v['cousin']
                else:
                    inoutDict[k] = v['cousin']
        # create a matrix of output

        # inialise score to 1
        #score = np.array([1] * len(inoutDict))

        vocab = graph.nodes()
        score = np.ones(len(vocab), dtype = np.float32)

        MAX_ITERATIONS = 50
        d = 0.85
        threshold = 0.0001 # convergence threshold
        count = 0
        for iter in range(MAX_ITERATIONS):
            prev_score = np.copy(score)

            for i in range(len(vocab)):
                summation = 0
                for j in range(len(vocab)):
                    if i != j:
                        if graph.has_edge(vocab[i], vocab[j]):
                            #print(graph[vocab[i]][vocab[j]]['cousin'])
                            #print(vocab[i], vocab[j])
                            summation += (graph[vocab[i]][vocab[j]]['cousin']/inoutDict[vocab[j]])*score[j]
                #print(" {} : {} ".format(str(vocab[i]), summation))
                score[i] = (1 - d) + d*(summation)

            if np.sum(np.fabs(prev_score - score)) <= threshold:
                # convergence baby
                print("convergence at {}".format(iter))
                break


        #print(score)

        textRankDict = dict(zip(vocab, score))

        d = dict(sorted(textRankDict.items(), key=lambda x: x[1], reverse = True))

        return d


    def extractFles(self):

        dataset = pd.DataFrame()
        # initialise methods class
        path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
        methods = mainMethods(path)
        # extracts the files handles
        methods.extractFileNames(path)
        # load up full filepath (xml)
        methods.df['fileNames'] = methods.df.handle.apply(methods.extractXMLFiles)
        # extract text content
        methods.df['files'] = methods.df.fileNames.apply(methods.extractContent)
        ### assigning sectionDictionary to dataset
        ### breaks the xml into sections allowing us to compartmentalise analysis
        dataset['sectionsDict'] = methods.df.files.apply(methods.extractSections)
        # remove the punctuation
        dataset['punctuationClean'] = methods.punctuationClean(dataset)
        # iterate over sections and create one doc
        # alternative was to parse the xml and remove the sections header ....
        dataset['wholeSections'] = dataset.punctuationClean.apply(methods.concatDocSections)
        return dataset
#####################################################################

# main 9

        # plotting vectors to the array
    def plotArray(self, array, depth, g):
        #print(array)
        #print(len(array))
        if len(array) < depth + 1:
            depth = len(array)
        targetTerm = array[0]
        #print(targetTerm)
        for i in range(1, depth):
            #if i == 1:
            g = self.AddGraphConnectionCousin( g, targetTerm, array[i], i)
            #g = self.AddGraphConnectionDistantCousin(g, targetTerm, array[i])

        return g , array[1:]

    def AddGraphConnectionCousin(self, g, a, b, i):
        g.add_edge(a, b)

        try:
            g[a][b]['cousin'] += 1/i
        except:
            g.add_edge(a, b, cousin = 1)
        return g

    def AddGraphConnectionDistantCousin(self, g, a, b):
        try:
            g[a][b]['distantCousin'] +=1
        except:
            g.add_edge(a, b, distantCousin= 1)
        return g

    def plotDiGraph(self, corpus):
        g = nx.Graph()
        # methods takes in an array (corpus) of arrays (docs) each containg tokenised sentences (section)
        for doc in corpus:
            for section in doc:
                #print(section)
                #print(10*"-")
                depth = 4
                while(len(section) > 1):
                    g, section = self.plotArray( section, depth, g)
                    #print(10*"....")
        #print(g.nodes(data = True))
        #print(g.edges(data = True))
        return g

    def expandNGram(self, brokenCorpus):
        all_corpus_rejoined = []
        for doc in brokenCorpus:
            docArray = []
            #print(doc)
            for line in doc:
                lineArray = []
                while len(line) > 0:
                    line, stringReturn = self.returnNGramConstruct(line)
                    #print(tester)
                    lineArray.extend(stringReturn)
                docArray.append(lineArray)
            all_corpus_rejoined.append(docArray)
        return all_corpus_rejoined

    def stemTheCorpus(self, corpus):
        corpusArray = []
        for doc in corpus:
            docArray = []
            for sect in doc:
                sectArray = []
                for line in sect:
                    stemLine = nlpMethods.stem_corpus(self, line)
                    sectArray.append(stemLine)
                docArray.append(sectArray)
            corpusArray.append(docArray)

        return corpusArray

    def stemTargetTerms(self, allTerms):
        allStemTerms = []
        for terms in allTerms:
            stemTerms = []
            for term in terms:
                stemTerm = nlpMethods.stem_corpus(self, term)
                stemTerms.append(stemTerm)
            allStemTerms.append(stemTerms)
        return allStemTerms

    def lemmatiseTargetTerms(self, allTerms):
        allLemmaTerms = []
        for terms in allTerms:
            lemmaTerms = []
            for term in terms:
                lemmaTerm = nlpMethods.lemmatise_corpus(self, term)
                lemmaTerms.append(lemmaTerm)
            allLemmaTerms.append(lemmaTerms)
        return allLemmaTerms

    def lemmatiseTheCorpus(self, corpus):
        corpusArray = []
        # first array holding the arrays of documents
        for doc in corpus:
            docArray = []
            for sect in doc:
                sectArray = []
                for line in sect:
                    #print(line)
                    #print(10*"-")
                    lemmaLine = nlpMethods.lemmatise_corpus(self, line)
                    sectArray.append(lemmaLine)
                docArray.append(sectArray)
            corpusArray.append(docArray)

        return corpusArray


    def expandAcronymsInText(self, dataset, accronymDict):
    # switching dataset.stopWordRemoved for brokenCorpus
        brokenCorpus_augment_anagram = []
        #for text in brokenCorpus:
        for text in list(dataset['stopWordRemoved']):
            # outer array
            aug_doc_array = []
            for t in text:
                # inner array
                #sectionDict = []
                sectionDoc = []
                for line in t:
                    sectionDoc.append(line)
                    for term in line.split():
                        if term.isupper():
                            if len(term) > 1:
                                if term in accronymDict:
                                    #print(term , accronymDict[term])
                                    acc = " ".join(accronymDict[term])
                                    sectionDoc.append(acc)
                aug_doc_array.append(sectionDoc)
            brokenCorpus_augment_anagram.append(aug_doc_array)
        return brokenCorpus_augment_anagram

    def extractAccronymnsCorpus(self, brokenCorpus):
        liste = []
        accronymDict = {}
        for text in brokenCorpus:
            # outer array - list[sections]
            for t in text:
                # inner array - sections[lines]
                for line in t:
                    # indicated potential accroynm definition
                    if "(" in line:
                        # pass to method that checks previous words to see if accroynm
                        tester = self.extractCandidateSubstring(line)
                        # if not blank add to dict
                        if tester[0] != '':
                            accronymDict[tester[0]] = tester[1]
                            liste.append(tester)
        return accronymDict


    def stopwordRemoval(self, text):
        # method for iterating over items in the dataframe
        # each item is a list of list of strings  (corpus/section/lines)
        # stopwords are removed and the the stringArrays are rejoined and returned for tokenizer down the lines
        # array to hold the doc
        docArray = []
        for sect in text:
            # array to hold each section
            sectArray = []
            # break each string in the list and rremove stopwords
            for line in sect:
                line = line.split()
                lineArray = [x for x in line if x not in stop]
                # rejoin the stingArrays
                sectArray.append(" ".join(lineArray))
            # append the sections to the docArray
            docArray.append(sectArray)
        # return the entire document as list of list of strings
        return docArray


    def extractTargetTerms(self, dataset):
        self.df['keywords'] = self.df.handle.apply(self.extractKeyWordFiles)
        self.df['competition_terms'] = self.df.handle.apply(self.extractKeyWordFilesTerms)
        return  self.extractKeyPhrases()


    def divideByIndicator(self, array, indicator):
        chunkArray = []

        for sent in array:
            sent1 = sent.split(indicator)
            chunkArray.extend(sent1)
        #print('\n')
        #print(chunkArray)
        #print(10*"==")

        #    for chunk in value.split(indicator):
        #        chunkArray.extend(chunk)

        return chunkArray


    def tokeniseCorpus(self, dataset):
        corpusArray = []
        # loop over corpus ( list of dictionaries)
        for i in range(len(dataset.sectionsDict)):
            # store each line in one document array
            sectArray = []
            # loop through each value in the dictionaries set
            for value in list(dataset.sectionsDict[i].values()):
                # remove "\n" as they can interfer with results
                value = " ".join(value.split("\n"))
                # split into sentences
                value  = value.split(". ")
                #print(value)
                # further split into commas
                commaArray = self.divideByIndicator(value, ",")
                collonArray = self.divideByIndicator(commaArray, ":")
                semiCollonArray = self.divideByIndicator(collonArray, ";")
                #    commaArray = self.divideByIndicator(sent, ",")
                #    commaArray.extend(v)
                #    print(v)
                    #print("----->>>----")
                #print(2*"\n")
                sectArray.append(semiCollonArray)
            corpusArray.append(sectArray)

        return corpusArray

    def cleanString(self, corpusArray):
        allDocs = []
        for i in range(len(corpusArray)):
            sectArray = corpusArray[i]
            # each section is addressed on a sentence level
            sentArray = []
            for sect in sectArray:
                for sent in sect:
                    sent = self.cleanSent(sent)
                    sentArray.append(sent)

            # cleaning by sentence
            cleanerSent = []
            for sent in sentArray:
                sent = self.cleaningStep(sent)
                cleanerSent.append(sent)

            docArray = []
            for i in range(len(cleanerSent)):
                cleanerCleanerSent = []
                for word in cleanerSent[i].split():
                    word = word.strip().lower()
                    cleanerCleanerSent.append(word)
                docArray.append(cleanerCleanerSent)
            allDocs.append(docArray)

        #print(len(corpusArray))
        return allDocs

    def extractIndexLocationForAllTargetTerms(self, df, dataset, title = "indexListDf.pkl", failSafe = False):
        ## method loops over the dataset of terms and ranks them according to tfidf
        ## method extracts the location for each target term and returns term, index location
        indexList = []
        exceptionCount = 0
        if (failSafe):
            print('# extract the doc target terms ')

            # list to store dictionary result of term positions
            indexList = []
            # loop over all documents
            #for i in range(dataset.shape[0]):
            for i in range(211):
                # dictinoary to store index result location per document
                indexLocation = {}
                # extract doc target terms
                doc_target_terms = dataset.targetTerms[i]
                # extract the doc terms
                doc_corpusTerms = df[df.doc_id_list  == i]
                # order your terms
                doc_corpusTerms.sort_values(by=['term_idf_list'], inplace=True, ascending=False)
                # reset the index so as to know where your terms fall on the scale of things
                doc_corpusTerms.reset_index(inplace = True, drop=True)
                # loop over terms list and extract index location
                for term in doc_target_terms:
                    try:
                        print(term)
                        indexLocation[term] = doc_corpusTerms[doc_corpusTerms['term_list']==term].index.item()
                    except:
                        #print(i)
                        #print(term)
                        indexLocation[term] = -1
                        exceptionCount = exceptionCount + 1
                # add dict to list
                indexList.append(indexLocation)
                # rinse and repeat

                with open(title, 'wb') as f:
                    pickle.dump(indexList, f, pickle.HIGHEST_PROTOCOL)

        else:
            with open(title, 'rb') as f:
                indexList =  pickle.load(f)

        print(len(indexList))
        print("the exception count for this run is {}".format(exceptionCount))
        return indexList

    def extractNgramsFromSentArrays(self, datasetSection):
        #print(len(datasetSection))
        docArray = []
        for doc in datasetSection:
            sentArray = []
            for word in doc:
                #word = 'a b c d e f g'
                wordArray = word.split()
                #print(10*"=+=")
                stringArray = []
                while len(wordArray) > 1:
                    wordArray, stringReturn = self.returnNGramConstruct(wordArray)
                    stringArray.extend(stringReturn)
                #print(len(stringArray))
                #print(stringArray)
                sentArray.extend(stringArray)
            docArray.append(sentArray)
        #print(docArray)
        return docArray

    def returnNGramConstruct2(self, wordArray):
        print(wordArray)

    def returnNGramConstruct(self, wordArray):
        string = ""
        nGram = 4
        sentLen = len(wordArray)
        if sentLen < nGram:
            nGram = sentLen + 1
        processSection = wordArray[:4]
        stringArray = []
        for i in range(nGram - 1):
            string = string.strip() + " " +  processSection[i].strip() + " "
            stringArray.append(string.strip())
        #print(stringArray)

        return wordArray[1:] , stringArray


    def breakDocIntoChunks(self, textSections):
        brokenSentArray = []
        text = " ".join(textSections.split("\n"))
        sentArray = text.split(". ")
        for sent in sentArray:
            sent = sent.split(", ")
            brokenSentArray.extend(sent)
        return brokenSentArray


    def CutToChunks(self, dataset):
        # loop over docs and reduce to sent chunks
        docChunkArrayList = []
        for docSections in  list(dataset.sectionsDict):
            #print(type(docSections))
            docChunkArray = []
            for section in docSections.values():
                chunkDoc = self.breakDocIntoChunks(section)
                docChunkArray.extend(chunkDoc)
            docChunkArrayList.append(docChunkArray)
        return docChunkArrayList

    def punctuationCleanSentenceLevel(self, datasetSection):
        # takes in a series of array of arrays of doc sentences
        corpusList = []
        for doc in datasetSection:
            docString = []
            for sent in doc:
                # TODO
                docString.append(self.cleanSent(sent))
            corpusList.append(docString)
        return corpusList

    def cleanSent(self, sent):
        remove = string.punctuation
        remove = remove.replace("-", "")
        pattern = r"[{}]".format(remove)
        sent = re.sub(pattern, " ", sent.strip().lower())
        return sent

#####################################################################
    def plotIndexResults(self, list1):
        objects = ("<15", "<100", "<500", "<1000", ">1000" ,  "absent")
        y_pos = np.arange(len(objects))
        print(list1)
        plt.bar(y_pos , list1)
        plt.xticks(y_pos, objects)
        plt.ylabel("Occurences")

        plt.title("Index Location of Target Term")

        plt.show()

    def rankLocationIndex(self, indexList):
        cups = [15, 100, 500, 1000, 0]
        notPresent = 0
        zero = 0
        one = 0
        two = 0
        three = 0
        four = 0
        for docIndex in indexList:
            for index in docIndex:
                if index == -1:
                    notPresent = notPresent + 1
                elif index < cups[0]:
                    zero = zero + 1
                elif index < cups[1]:
                    one = one + 1
                elif index < cups[2]:
                    two = two + 1
                elif index < cups[3]:
                    three = three + 1
                else:
                    four = four + 1

        return [zero, one, two, three, four, notPresent]

    def extractKeyPhrases(self):
        keyWords = list(self.df.keywords)
        competitionWords = list(self.df.competition_terms)

        allWordsList = []
        for i in range(211):
            wordList = []
            if type(competitionWords[i]) is list:
                wordList.extend(competitionWords[i])
            if type(keyWords[i]) is str:
                # an unwanted character simlar to \n
                KeyWords_ = re.sub("\r", "", keyWords[i].lower())
                keywords = KeyWords_.split("\n")
                if keywords[-1] == "":
                    keywords = keywords[:-1]
                    wordList.extend(keywords)
            allWordsList.append(wordList)
        return allWordsList

    def concatDocSections(self, text):
        ''' takes in sections as dict type and concatenates them '''
        stringConCat = ""
        for item in text.values():
            stringConCat = stringConCat + " " + item

        return stringConCat

    def punctuationClean(self, dataset):
        punctuationCleanList = []
        remove = string.punctuation
        remove = remove.replace("-", "")
        pattern = r"[{}]".format(remove)

        for i in range(dataset.shape[0]):
            punctuationClean = {}
            for k, v in dataset.sectionsDict[i].items():
                k = re.sub(pattern, "", k)
                k = " ".join(k.split("\n"))
                v = re.sub(pattern, "", v)
                v = " ".join(v.split("\n"))

                punctuationClean[k] = v.lower()
            punctuationCleanList.append(punctuationClean)
        #text = dataset.sectionsDict[0]['"ABSTRACT"']
        #text = re.sub(pattern, "", text)
        #print(text)
        return punctuationCleanList

    def minimalCleaning(self, dataset):
        minimalCLeanlist = []
        for i in range(dataset.shape[0]):
            minimalCleanText = {}
            for k, v in dataset.sectionsDict[i].items():
                v = self.cleaningStep( v)
                k = self.cleaningStep( k)

                minimalCleanText[k] = v

            minimalCLeanlist.append(minimalCleanText)
        return minimalCLeanlist


    def cleaningStep(self, v):
        #newString = "".join(v.split(" "))
        newString = "".join(v.split(","))
        newString = "".join(newString.split("?"))
        newString = "".join(newString.split("'"))
        newString = "".join(newString.split("\""))
        newString = "".join(newString.split(":"))
        newString = "".join(newString.split(";"))
        newString = " ".join(newString.split("\n"))
        newString = "".join(newString.split(")"))
        newString = "".join(newString.split("("))
        return newString.lower()


    def extractSections(self, text):
        ## method to extract sections
        sectionsArray = []
        # identifies sections
        for result in re.findall('<SECTION(.*?)</SECTION>', text, re.S):
            sectionsArray.append(result)

        # further extract headers (will be used as keys)
        sectionHeaders= []
        for section in sectionsArray:
            for result in re.findall('header=(.*?)>', section):
                sectionHeaders.append(result)

        # append key headers to text
        sectionDict = dict(zip(sectionHeaders, sectionsArray))
        return sectionDict


    def combineListofLists(self,  lista, listb):
        termsArray = []
        for terms in lista:
            if not isinstance(terms, int):
                termsArray.extend(terms)
        for terms in listb:
            if not isinstance(terms, int):
                termsArray.extend(terms)
        termsDict = dict(Counter(termsArray))
        return termsDict


    def returntermDictKeys(self, text):
        keys = list(text.keys())
        keysArray = []
        for key in keys:
            text = nlpMethods.lemmatise_corpus(self, re.sub('\"', "" , key.lower()))
            keysArray.append(text)

        return keysArray

    def returntermDictKeysNOT_lemmatised(self, text):
        keys = list(text.keys())
        keysArray = []
        for key in keys:
            text = re.sub('\"', "" , key.lower())
            keysArray.append(text)

        return keysArray

    def extractSectionContent(self, dataset, keyTerm):
        abstract = 0
        counter = 1
        contentList = []
        keyList = []
        for data in dataset.punctuationClean:
            boolean = False
            for key, value in list(data.items()):
                prospect = key.lower().split()
                if len(prospect) == 1:
                    if keyTerm in prospect[0]:
                        abstract = abstract + 1
                        contentList.append(value)
                        keyList.append(key)
                        counter = counter + 1
                        boolean = True
            if not boolean:
                counter = counter + 1
                contentList.append("dud")
                keyList.append("dud")

        return keyList , contentList

    def extractRefernces(self, text):
        #pattern = 'REFERENCES'
        if text is not None:
            try:
                refs_loc = text.index(self.pattern)
                text1 = text[refs_loc:]
                return text1
            except:
                i = 1
            try:
                pattern1 = self.pattern.lower()
                pattern1 = pattern1[0].upper() + pattern1[1:]
                refs_loc = text.index(pattern1)
                text1 = text[refs_loc:]
                percent_loc  = float(refs_loc)/len(text)
                if percent_loc > .80:
                    return text1
            except:
                i = 1



    def lemmatiseCompTerms(self, text):
        sentArray = []
        # make sure that terms are present
        try:
            # split input text into array
            terms = text.split("\n")
            # loop over terms
            for term in terms:
                # phrases to be reduced to terms for lemmatising
                term = term.split()
                # ensure no blank strings
                if len(term) > 0:
                    # singletons
                    if len(term) == 1:
                        term = nlpMethods.lemmatise_corpus(self, term[0].lower())
                        sentArray.append(term)
                    # phrases
                    else:
                        termArray = []
                        # lemmatise each word in phrase
                        for t in term:
                            termArray.append(nlpMethods.lemmatise_corpus(self, t.lower()))
                        #remake phrase
                        term = " ".join(termArray)
                        sentArray.append(term)
        except:
            print("detected problem with below")
            print(text)
        return sentArray

    def extractIdfTermsDoc(self, text):
        #print(self.df.handle.index.values[text - 1])
        df1 = self.allTermIDFList[self.allTermIDFList.doc_id_list == (text- 1)]
        df1.sort_values(by=['term_idf_list'], inplace=True, ascending=False)
        tempDict = dict(zip(df1.term_list, df1.term_idf_list))

        return tempDict

    def amalgamateAllDocTermDictionaries(self, dict):
        self.termCorpusCount.update(dict)


    def extractAlternativeCorpus(self, path):
        #path = "Krapivin2009/all_docs_abstacts_refined/"
        _ , docs = self.extractFileNames(path)
        #print(docs)
        docLocations = []
        for item in docs:
            if ".txt" in item:
                path1 = path + item
                docLocations.append(path1)

        return docLocations

    def termCountDictionaryMethod(self, text):
        text = dict(Counter(text.split()))
        return text

    def countTotalTerms(self, text):
        # split the terms into dictionary counts
        text = text.split()
        return text

    def extractAccronymns(self):
        accroDict = {}
        for file in self.df.files:
            for f in file:
                acc , substring = self.extractCandidateSubstring(f)
                # make sure there are no empty strings or single letter matches
                if len(acc) > 1:
                    # store all accronyms
                    acc = acc.lower()
                    accroDict[acc] = substring

        # populate the class dictionary with accronyms from the corpus
        self.accronymList = accroDict

    def extractFileNames(self, path):
        listee = []
        dirs =  os.listdir(path)
        for d in dirs:
            path1 = path + d
            if(os.path.isdir(path1)):
                listee.append(int(d))
        listee.sort()
        # assigns the value to the class def
        self.df = pd.DataFrame({"handle" : listee})
        return listee , dirs

    def cleanData(self, data):

        cleaned = []
        for line in data:

            line = self.expandAcronyms(line)
            #remove urls
            line = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', line, flags=re.MULTILINE)
            # keep only text tokens
            line = re.sub('[^a-zA-Z]', ' ' , line)
            # break strings into an array
            line = line.split()
            # apply lower casing
            line = [x.lower() for x in line if x not in stop]
            # remove all single letters resulting from processing
            line = [x for x in line if len(x) > 1]
            # append to a list
            cleaned.extend(line)
        # convert list to string for processor friendly format
        cleaned = " ".join(cleaned)

        return cleaned

    # method for cleaning an array of string Terms
    def cleanTermsArray(self, array):
        clean_array = []
        # loop over array

        for a in array.split("\n"):
            # remove all non characters
            a = re.sub('[^a-zA-Z]', ' ' , a)
            clean_array.append(a.strip().lower())

        return clean_array

    def expandAcronyms(self, line):
        # identifier pattern for acronyms
        pattern = '\((.*?)\)'

        # cast the string to an array
        line = line.split(" ")
        # loop over the string looking for acronyms
        for i in range(0, len(line)):
            # extract terms inside brackets
            cand = re.search(pattern, line[i])
            # if there is a pattern match
            if cand:
                 # extract candidate
                candidate = cand.group(1)
                # check if candidate exists in established accronyms
                if candidate.lower() in self.accronymList:
                    # if yes update string with expanded accronym
                    accronym = " ".join(self.accronymList[candidate.lower()])
                    line.append(accronym)
        # return line as string
        line = " ".join(line)
        return line



    def extractCandidateSubstring(self, match):
        pattern = '\((.*?)\)'
        candidate = ""
        substring = ""
        #match = match.strip("\n")

        match = match.split(" ")
        for i in range(0, len(match)):
            cand = re.search(pattern, match[i])
            if cand:
                print(match[i])
                candidate = cand.group(1)
                # check that it is longer than 1
                if len(candidate) > 1:
                    # check and remove for non capital mix
                    if(self.lookingAtAcroynms(candidate)):
                        candidate = self.removeNonCapitals(candidate)
                    j = len(candidate)
                    substring = match[i-j:i]
                    # check if accronym is present
                    wordsAccro = self.returnPotentAccro(substring)
                    if candidate.lower() == wordsAccro.lower():
                        # return the correct accro and definition
                        return (candidate.lower(), substring)

        # no accronym found return blank , will be filtered out
        return("", "")

    def extractFiles(self, text):
        #path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
        return self.path + str(text) + "/" + str(text) + ".txt"

    def extractXMLFiles(self, text):
        #path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
        return self.path + str(text) + "/" + str(text) + ".xml"

    def extractKeyWordFiles(self, text):

        #path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
        path = self.path + str(text) + "/" + str(text) + ".kwd"
        try:
            text = self.extractContent(path)
            #print(text)
        except:
            #print("keyword absent: " + str(text))
            x = 1
        return text

    def extractKeyWordFilesTerms(self, text):

        #path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
        path = self.path + str(text) + "/" + str(text) + ".term"
        try:
            text = self.extractContent(path)
            text = self.cleanTermsArray(text)
            text = nlpMethods.lemmatise_corpus(self, " ".join(text))
            text = text.split()
        except:
            #print("keyword absent: " + str(text))
            x = 1
        return text

    def extractContent(self, text):
        #with open(text, "rb")   as file:
        with codecs.open(text, 'r', encoding='utf8', errors="ignore") as file:
            lines = file.read()
        return lines


    def applyTFidfToCorpus(self, dfList, title = "tfidf_store.pkl", failSafe = False):
        # create tf-idf matrix for the corpus
        #tfidf_matrix = None
        try:
            if (failSafe):
                ''' purposely crash try/except to force vectoriser rebuild '''
                x = 1/0

            print("-- Retrieving stored tfidf_matrix --")

            tfidf_matrix = pickle.load(open("matrix_" + title, "rb" ) )
            tfidf_vectoriser = pickle.load(open("vectorisor_" + title, "rb" ) )


        except:

            print("failed to load -- building tokeniser --")
            # initialise vectoriser and pass cleaned data
            tfidf_vectoriser = TfidfVectorizer(max_df = 0.9, min_df = 0.1, ngram_range = (1,4), stop_words ='english', tokenizer = self.tokenize_only)
            tfidf_matrix = tfidf_vectoriser.fit_transform(list(dfList))

            #df= pd.DataFrame({"tfidf_matrix" : tfidf_matrix}, index=[0])
            #save_tfidf.to_pickle("tfidf_min_04.pkl")
            #df.to_pickle("tfidf_matrix.pkl")

            # pickle tfidf matrix for faster future load
            with open("matrix_" + title, 'wb') as handle:
                        pickle.dump(tfidf_matrix, handle)

            # pickle tfidf vectoriser for faster future load
            with open("vectorisor_" + title, 'wb') as handle:
                        pickle.dump(tfidf_vectoriser, handle)

        return tfidf_matrix , tfidf_vectoriser


    def ExtractSalientTerms(self, tfidf_vectoriser, tfidf_matrix, title = "tfidf_.pkl",  failSafe = True):
        print('salient terms')
        df = pd.DataFrame()
        try:
            if (failSafe):
                ''' purposely crash try/except to force vectoriser rebuild '''
                x = 1/0

            print("loading presaved processed corpus --")
            df = pd.read_pickle(title)
            # lists for storing data

        except:
            print(" failed to load terms -- rebuilding -- ")
            doc_id_list = []
            term_list = []
            term_idf_list = []

            # extract terms from vectoriser
            terms = tfidf_vectoriser.vocabulary_


            keys = terms.keys()
            values = terms.values()

            # invert the dict so the keys are the values and values the keys
            dict1 = dict(zip(values, keys))

            # shortcut for saving and loading dictionary
            self.wordStore = dict1
            with open('dict' + title + '.pkl', 'wb') as f:
                pickle.dump(dict1, f, pickle.HIGHEST_PROTOCOL)


            # iterate through matrix
            for i in range(0, (tfidf_matrix.shape[0])):
                for j in range(0, len(tfidf_matrix[i].indices)):
                    # append the appropriate list with the appropriate value
                    doc_id_list.append(i)
                    term_list.append(dict1[tfidf_matrix[i].indices[j]])
                    term_idf_list.append(tfidf_matrix[i].data[j])

            # cast to dataframe
            df = pd.DataFrame({"doc_id_list": doc_id_list, "term_list" : term_list, "term_idf_list": term_idf_list})
            # pickle process for future fast retrieval
            df.to_pickle(title)

        print('loading dictionary')
        with open('dict' + title + '.pkl', 'rb') as f:
            self.wordStore =  pickle.load(f)

        #print(list(self.wordStore.items())[:5])
        return df

    def extractTopNTerms(self, df ,  N = 10, title = "alt_termsList.pkl" , failSafe = False):
        # extract the terms specific to that document
        # list for storing document top terms
        try:
            if(failSafe):
                x = 1/0
            print("-- loading saved terms --")
            self.df['method_termDict'] = pd.read_pickle(title)

        except:
            print("failSafe -- building term list")
            termList = []
            for i in self.df.index:
                df1 = df[df.doc_id_list == i]
                # order the terms from highest to lowest
                df1.sort_values(by=['term_idf_list'], inplace=True, ascending=False)
                # extract the top N values
                values = list(df1.term_idf_list)[:N]
                # extract the top N terms
                terms =list(df1.term_list)[:N]
                # cast to dictionary
                termDict = dict(zip(terms, values))
                termList.append(termDict)

            # update the class df so that each doc has a corresponding termDict
            self.df['method_termDict'] = termList
            self.df['method_termDict'].to_pickle(title)

        return self.df['method_termDict']


    def tokenize_only(self, text):
        # data has been processed and requires only splitting into tokens
        tokens = [word for word in text.split(" ")]
        return tokens

    ################################################################################
    # code for looking at accroynms

    def extractCandidateSubstring(self, match):
        pattern = '\((.*?)\)'
        candidate = ""
        substring = ""
        #match = match.strip("\n")

        match = match.split(" ")
        for i in range(0, len(match)):
            cand = re.search(pattern, match[i])
            if cand:
                candidate = cand.group(1)
                # check that it is longer than 1
                if len(candidate) > 1:
                    # check and remove for non capital mix
                    if(self.lookingAtAcroynms(candidate)):
                        candidate = self.removeNonCapitals(candidate)
                    j = len(candidate)
                    substring = match[i-j:i]
                    # check if accronym is present
                    wordsAccro = self.returnPotentAccro(substring)
                    if candidate.lower() == wordsAccro.lower():
                        # return the correct accro and definition
                        return (candidate, substring)
        # no accronym found return blank , will be filtered out
        return("", "")

    # check of the main lettes match
    def returnPotentAccro(self, substring):
        firsts = ""
        for s in substring:
            if(len(s) > 0):
                firsts = firsts + s[0]
        return firsts


    def lookingAtAcroynms(self, accro):
        # case one check if accroynm has an append s
        bool = False
        for s in accro[:1]:
            if s.isupper:
                bool = True
        return bool

    def removeNonCapitals(self, accro):
        string = ""
        for s in accro:
            if s.isupper():
                string = string + s
        return string

    def extractCorpusPhraseArray(self, corpus, failSafe = False):
        try:
            if(failSafe):
                ''' purposely crash try/except to force phrase rebuild '''
                x = 1/0
            print("-- Retrieving stored phrase df --")
            df = pd.read_pickle("phraseDF.pkl")

        except:
            print("building -- extracting phrases from text -- ")
            # array to store each document dictionary
            dictionaryList = []
            for text in corpus:
                vector = self.createTextVectors(text)
                dict = self.extractDocPhraseArray(vector)
                dictionaryList.append(dict)

            df = pd.DataFrame({"phraseLists": dictionaryList})
            df.to_pickle("phraseDF.pkl")

        return df



    def extractDocPhraseArray(self, vector):
        doc_dict = {}
        index = 0
        sliding_window = 0
        term = ""
        #loop over the text vector
        while index < (len(vector)):
            if(index == sliding_window):
                term = vector[index]
            # if not in dict, add it
            if term not in doc_dict:
                # assign a value for the instance
                doc_dict[term] = 1
                # move the index forward
                index  = index + 1
                # reset the sliding_window
                sliding_window = index
            else:
                doc_dict[term] = doc_dict[term] + 1
                # increment the sliding window
                sliding_window = sliding_window + 1
                # create phrase term
                term = term + "_" +  vector[sliding_window]
            if sliding_window == len(vector) - 1:
                # take the sliding window back one to prevent outofbounderror
                sliding_window  = sliding_window - 1
                # increment index to move the term capture
                index = index + 1

        return doc_dict

    def createTextVectors(self, text):
        text_array = []
        # loop over each array in the text vector
        for t in text:
            # remove non characters
            t = re.sub('[^a-zA-Z]', ' ' , t)
            # split the strings into word vectors
            for t in t.split():
                # ignore any words less than 1
                if len(t) > 1:
                    # append the result to overall vector
                    text_array.append(t.lower())

        return text_array

    def removeIndex(df):
        for lines in df.files[0]:
            print(lines)
            print(10*"*")

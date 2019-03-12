import xml.etree.ElementTree as ET
import json
import pandas as pd
from pprint import pprint

file_dir = "./data/AviationData.xml"
narr_dir = "./data/NarrativeData_"

tree = ET.parse(file_dir)
root = tree.getroot()

### count for each types of nodes in XML
tag_count_dict = dict()
for node in tree.iter():
    if node.tag not in tag_count_dict:
        tag_count_dict[node.tag] = 0
    tag_count_dict[node.tag] += 1
all_count = tag_count_dict["{http://www.ntsb.gov}ROW"]
pprint(tag_count_dict)

### count for event in JSON
narr_name = ["000"]
narr_name.extend(range(499, 71000, 500))
narr_name.append(999999)
narr_name = list(map(lambda x: str(x)+".json", narr_name))

report_count = 0
narr_count = 0
cause_count = 0
missing_count = 0

for name in narr_name:
    with open(narr_dir + name) as f:
        narr_data = json.load(f)["data"]   # a list of events
        for event in narr_data:
            report_count += int(event["EventId"] != "")
            narr_count += int(event["narrative"] != "")
            cause_count += int(event["probable_cause"] != "")
            missing_count += int(event["narrative"] == "" and event["probable_cause"] == "")

# short summary for JSON files
print("Count for -\nEvent:\t%d\nNarr:\t%d\nCause:\t%d\nMiss:\t%d" %
      (report_count, narr_count, cause_count, missing_count))

# add items_d into dictionary with key eventId
def add_dict(d, eventId, items_d):
    ''' d   - the dictionary of all events
    eventId - the id of a event to be added
    items_d - items (dict) to be added to eventId'''
    
    assert type(d) == dict and type(items_d) == dict and type(eventId) == str, "input type error"
    
    if eventId not in d:
        d[eventId] = dict()
    d[eventId]
    for key, value in items_d.items():
        if key in d[eventId] and d[eventId][key] != value:
            pass
            print("weird thing happens to %s" % eventId)
        else:
            d[eventId][key] = value

### iterate through XML to add information
accident_dict = dict()  # AccidentNumber as unique key
eid_anum_dict = dict()  # use EventId to get AccidentNumber
for node in tree.iter(tag="{http://www.ntsb.gov}ROW"):
    anum = node.attrib["AccidentNumber"]
    eid = node.attrib["EventId"]
    accident_dict[anum] = node.attrib
    if eid not in eid_anum_dict:
        eid_anum_dict[eid] = list()
    eid_anum_dict[eid].append(anum)

#df_accident = pd.DataFrame.from_dict(accident_dict, orient='index')
#df_accident.loc[["WPR15LA253A","WPR15LA253B","WPR15FA243A","WPR15FA243B"],:] # EventId is not unique

### iterate through JSON to add information
narr_dict = dict()
for name in narr_name:
    with open(narr_dir + name) as f:
        narr_data = json.load(f)["data"]   # a list of events
        for event in narr_data:
            eid = event["EventId"]
            ''' # check for duplicate - NO duplicate
            if eid in narr_dict:
                print("weird:\t", eid)
            '''
            narr_dict[eid] = event

import os
import tempfile
TEMP_FOLDER = tempfile.gettempdir()
print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import string
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()
stoplist = set(stopwords.words('english'))
frequency = Counter()

# remove stop words
# remove punctuations
# lower words
# stemming
documents = list()
eventID = list()
for key, value in narr_dict.items():
    sent = (value["narrative"]+value["probable_cause"]).translate(str.maketrans('', '', string.punctuation)).lower().split()
    text = [ps.stem(word) for word in sent if word not in stoplist]
    documents.append(text)
    eventID.append(key)
    frequency.update(text)

# remove rare words
documents = [[word for word in document if frequency[word] > 1] for document in documents]

# create dictionary between tokens and counts
from gensim import corpora
dictionary = corpora.Dictionary(documents)
dictionary.save(os.path.join(TEMP_FOLDER, 'aviation.dict'))  # store the dictionary, for future reference
print(dictionary)

corpus = [dictionary.doc2bow(document) for document in documents]
corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'aviation.mm'), corpus)  # store to disk, for later use

from gensim import models

tfidf = models.TfidfModel(corpus)

vec = corpus[0]


from gensim import similarities
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=5)

sims = index[tfidf[vec]]

print(list(enumerate(sims))[0:10])




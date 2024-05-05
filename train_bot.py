#Text Data Preprocessing Lib
import nltk #natural language toolkit
from nltk.stem import PorterStemmer
import json 
import pickle
word_tags_list=[]
stemmer=PorterStemmer()
ignore_words=['?', '!',',','.', "'s", "'m"]
words=[]
classes=[]
intents=json.loads('intents.json')
# function for appending stem words
def get_stem_words(words,ignore_words):
    stem_words=[]
    for word in words:
        if word not in ignore_words:
            w=stemmer.stem(word.lower())
            stem_words.append(w)
    return stem_words

for intent  in intents['intents']:
    for pattern in intent['patterns']:
        pattern_word= nltk.word_tokenize(pattern)  
        words.extend(pattern_word)
        word_tags_list.append(pattern_word,intent['tag'])

#Create word corpus for chatbot
def create_bot_corpus(stem_words,classes):
    stem_words=sorted(list(set(stem_words)))
    classes=sorted(list(set(classes)))
    pickle.dump(stem_words,open('words.pkl','wb'))
    pickle.dump(classes,open('classes.pkl','wb'))
    return stem_words,classes
stem_words,classes=create_bot_corpus(stem_words,classes)    
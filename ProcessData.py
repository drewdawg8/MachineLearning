import os
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import hashing_trick
from nltk.tokenize import RegexpTokenizer
def textfile_to_array(path):
    ''' Takes a path argument and retrieves all the text lines from
        within a folder'''
    name_list = []
    texts = []
    for file_ in os.listdir(path):
        name_list.append(file_)
    for i in name_list:
        file_ = open(path+i)
        texts+=file_.read().splitlines()

    return texts

def process_for_model(textArray):
    '''
     Given a 2D array of the form:
     [[fileLines1],[fileLines2]...[fileLinesN]]
     converts the text into integers
    '''
    result = []
    for file_ in textArray:
        inner = []
        for line in file_:
            length = len(set(text_to_word_sequence(line)))
            inner.append(hashing_trick(line,round(length*1.3),hash_function='md5'))
        result.append(inner)

    return result

def convert_text(text,vocab_text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    text_list = tokenizer.tokenize(text)
    text_list = [x for x in text_list if len(x)>1]
    result = [vocab_text[t]+1 for t in text_list]
    return result

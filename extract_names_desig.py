# -*- coding: utf-8 -*-
"""
Created on Sun May 28 19:16:43 2017
 
   Name and designation extraction from text
 
@author: RAHUL
"""

#Import parent libraries
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk.regexp import ChunkRule, ExpandLeftRule, ExpandRightRule, UnChunkRule
from nltk.chunk import RegexpChunkParser
 
#Fetch sample text
sample = open('./sample.txt', 'r').read()

#Clean Junk text using regex 
def clean_junk(sent_frag):
    print('.....')
    clean_sent = []
    regex = re.compile(r"^[^\t\s]+.{1,100}")
    for frag in sent_frag:
        for req_frag in re.findall(regex, frag):
            if req_frag != '':
                clean_sent.append(req_frag)
    return clean_sent

#Process and split sentences using regex 
def preprocess(document):
    print('Preprocessing text file')
    sentences = sent_tokenize(document)
    cleaned_sentences = []
    print('Cleaning unrequired junk')
    for sent in sentences:
        sent = re.split(r"\n", sent)
        sent = clean_junk(sent)
        cleaned_sentences.append(sent)
    print('Preprocessing complete\n')
    return cleaned_sentences

#Tokenize, Pos tag and chunk to extract nouns 
def extract_nouns(cleaned_sentences):
    print('Extracting nouns')
    chunk_set = []
    for sent in cleaned_sentences:
        for sub_sent in sent:
            words = word_tokenize(sub_sent)
            word_tagged = nltk.pos_tag(words)
            pn = ChunkRule('<NNP><NNP>', 'proper noun, singular')
            el = ExpandLeftRule('<.*>', '<NNP>', 'get left proper noun')
            er = ExpandRightRule('<NNP>', '<.*>', 'get right hyphen')
            un = UnChunkRule('<CC|CD|DT|EX|FW|IN|JJ.*|LS|MD|NN|NNS|NNPS|PDT|POS|PRP.*|RB.*|RP|SYM|TO|UH|VB.*|WDT|WP.*>+<NNP>+<CC|CD|DT|EX|FW|IN|JJ.*|LS|MD|NN|NNS|NNPS|PDT|POS|PRP.*|RB.*|RP|SYM|TO|UH|VB.*|WDT|WP.*>+', 'unchunk unrequired')
            chunker = RegexpChunkParser([pn, el, er, un], chunk_label = 'Chunk')
            chunked = chunker.parse(word_tagged)
            for subtree in chunked.subtrees(filter = lambda t: t.label()=='Chunk'):
                ck = nltk.tag.untag(subtree.leaves())
                if '-' == ck[0]:
                    del ck[0]
                chunk_set.append(ck)
    print('Extraction commplete\n')
    return chunk_set

#re-chunk to extract names 
def extract_name_desig(ck_set):
    print('Extracting names and designations')
    final_set = []
    chunkGram = r"""Chunk:
                     {^<NN.?><NNP><NNP>?}
                     <NN.?>{}<NN.?>
                 """
    chunkParser = nltk.RegexpParser(chunkGram)
    for ck in ck_set:
        ckd = chunkParser.parse(nltk.pos_tag(ck))
        for subtree in ckd.subtrees(filter = lambda t: t.label()=='Chunk'):
                final_set.append(nltk.tag.untag(subtree.leaves()))
    print('Extraction complete\n')
    return final_set

#Clean up and print final result
def display_stuff(name_des):
    print('Generating Final Results file')
    #Final result file
    result = open('./names_designations.txt', 'w')
    result.write("NAME"+"\t\t"+"DESIGNATION") 
    
    k =0
    buffer, j = '', ''
    
    for t in name_des:
        i = str(' '.join(c for c in t))
        if k == 0:
            j = i
            buffer = i
            k+=1
        else:
            if i == j:
                result.write('\n'+buffer)
                k = 0
                continue
            elif i in ('Farmer Award', 'Land Development Board') or (i != j and k==3):
                k = 0
                result.write('\n'+buffer)
                continue
            else:
                if i == str(' '.join(c for c in name_des[len(name_des)-1])):
                    result.write('\n'+buffer)
                else:
                    buffer = buffer+'\t'+i
                    k+=1
    result.close()
    print('Names and Designations successfully generated and saved in the file \'names_designations.txt\'')
    
if __name__ == '__main__':
    noun_set = extract_nouns(preprocess(sample))
    name_des = extract_name_desig(noun_set)
    display_stuff(name_des) 
import json
import nltk
from pprint import pprint

"""with open('en_de_corpus.json') as data_file:    
    data = json.load(data_file)

en = open('en.txt', 'w')
de = open('de.txt', 'w')

if len(data['en']) == len(data['de']):
    print 'Total Sentences: {}'.format(len(data['en']))

for i in range(0,len(data['en'])) :
    tok_en = nltk.word_tokenize(data['en'][i])
    tok_de = nltk.word_tokenize(data['de'][i])
    if len(tok_en)<6 and len(tok_de)<6 :
        for item in tok_en:
            en.write(item.encode("UTF-8"))
            en.write(" ")
        en.write("\n")
        for item in tok_de:
            de.write(item.encode("UTF-8"))
            de.write(" ")
        de.write("\n")

en.close()
de.close()
"""

en = open('en.txt')
en_content = en.read()
en_word_list = en_content.split()
en_word_list.append('SEN-STRT')
en_word_list.append('SEN-END')
en_unique_word = set(en_word_list)

print "en_unique_word {}".format(len(en_unique_word))

en_hot_vec = open('en_hot_vec.txt','w')
en_dict = {}
index = 0
list_zero = [0]*len(en_unique_word)
for item in en_unique_word:
    en_dict[item] = index
    list_zero[index] = 1
    en_hot_vec.write(item)
    en_hot_vec.write(" ")
    en_hot_vec.write(" ".join(str(x) for x in list_zero))
    en_hot_vec.write('\n')
    list_zero[index] = 0
    index = index+1
en.close()
en = open('en.txt')
en_lines = en.readlines()
print "en_lines {}".format(len(en_lines))
en_labeled = open('en_label.txt','w')
for line in en_lines:
    items = line.split()
    en_labeled.write(str(en_dict['SEN-STRT']))
    en_labeled.write(" ")
    for item in items:
        en_labeled.write(str(en_dict[item]))
        en_labeled.write(" ")
    en_labeled.write(str(en_dict['SEN-END']))
    en_labeled.write("\n")

en.close()
en_hot_vec.close()
en_labeled.close()


de = open('de.txt')
de_content = de.read()
de_word_list = de_content.split()
de_word_list.append('SEN-STRT')
de_word_list.append('SEN-END')
de_unique_word = set(de_word_list)

print "de_unique_word {}".format(len(de_unique_word))

de_hot_vec = open('de_hot_vec.txt','w')
de_dict = {}
index = 0
list_zero = [0]*len(de_unique_word)
for item in de_unique_word:
    de_dict[item] = index
    list_zero[index] = 1
    de_hot_vec.write(item)
    de_hot_vec.write(" ")
    de_hot_vec.write(" ".join(str(x) for x in list_zero))
    de_hot_vec.write('\n')
    list_zero[index] = 0
    index = index+1
de.close()
de = open('de.txt')
de_lines = de.readlines()
print "de_lines {}".format(len(de_lines))
de_labeled = open('de_label.txt','w')
for line in de_lines:
    items = line.split()
    de_labeled.write(str(de_dict['SEN-STRT']))
    de_labeled.write(" ")
    for item in items:
        de_labeled.write(str(de_dict[item]))
        de_labeled.write(" ")
    de_labeled.write(str(de_dict['SEN-END']))
    de_labeled.write("\n")

de.close()
de_hot_vec.close()
de_labeled.close()




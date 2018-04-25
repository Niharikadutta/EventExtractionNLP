import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.wsd import lesk

# sent = "In 1952, he moved with his family to Chapel Hill, North Carolina where his father, Isaac M. Taylor, was Dean of the Medical School at UNC."
word = "moved"
sent = "In 1952, he moved with his family to Chapel Hill, North Carolina."
list = lesk(sent, word)
print(list.lexname())
# print(wordnet.synset('visit.v.01').lemma_names())
# print("word:", wordnet.synset('visit.v.01').lexname)
# print("synsets: visit 3 ", wordnet.synset('visit.v.03').lemma_names())
# print("synsets: visit 1 ", wordnet.synset('visit.v.01').lemma_names())

# for synset in wordnet.synsets('visit'):
#     print(synset.lexname)

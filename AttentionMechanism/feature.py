import random
import re
from nltk.wsd import lesk

PUNCTUATION = ["-LRB-", "-RRB-", "$", "#", "``", "''", ",", ".", ":", "SYM"]

quant_adj = ["whole", "enough", "little", "all", "hundred", "whole",
                 "no", "some", "sufficient", "any", "few", "most", "heavily",
                 "empty", "great", "couple", "half", "many", "much", "less",
                 "insufficient", "double", "hundreds", "thousands", "substantial",
                 "single", "each", "one", "two", "first", "second", "last", "several", "every"]


# Yahya///0-5///PERSON///NNP///1 -> word///index-index///NER///POS///arg
PATTERN = re.compile(r"(.+?)///(\d+-\d+)///(.+?)///(.+?)///(.+?)")

def extract_verb_type(word, sent):
    # print("word: ", word)
    # print("sentence: ", sent)
    sense = lesk(sent, word, 'v')
    if not sense:
        return "verb.misc"
    return sense.lexname()

class Feature:
    def __init__(self, word_dictionary, entity_type_dictionary, window_size=10):
        self.word_dictionary = word_dictionary
        self.unknow_index = 0
        self.zero_index = 0
        self.entity_type_dictionary = entity_type_dictionary
        self.window_size = window_size

    def word_2_index(self, word):
        return self.word_dictionary.get(word.lower(), self.unknow_index)

    def words_2_index(self, words):
        indexs = [
            self.word_dictionary.get(word.lower(), self.unknow_index) 
                if word is not None 
                else self.zero_index
            for word in words
        ]
        return indexs

    def entities_2_index(self, entities):
        indexs = [
            self.entity_type_dictionary.get(entity, 0)
            for entity in entities
        ]
        return indexs

    # change for quant entity
    def quantity_type(self, words):
        indices = [
            self.entity_type_dictionary.get("QUANTITY", 0)
            if word in quant_adj
            else self.zero_index
            for word in words
        ]
        return indices


    # change for verb entity
    def verb_type(self, verb_words, sentence):
        # verb_type = []
        indices = [0] * len(verb_words)
        for i in range(len(verb_words)):
            if verb_words[i] and "VB" in verb_words[i][1]:
                indices[i] = self.entity_type_dictionary["VERB"].get(extract_verb_type(verb_words[i][0], sentence), 0)

        # indices = [
        #     self.entity_type_dictionary["VERB"].get(extract_verb_type(token[0], sentence), 0)
        #     if token[0] in verb_words and "VB" in token[3]
        #     else self.zero_index
        #     for token in tokens
        # ]
        return indices



    def context_word_feature(self, sent_info):
        tokens = sent_info["tokens"]
        trigger_word = sent_info["trigger_word"]
        if trigger_word is not None:
            trigger_word_index = trigger_word[2]
            category = trigger_word[1][:-8]
        else:
            trigger_word_index = random.randint(0, len(tokens)-1)
            category = "none"
        context_word = [
            tokens[i].split('///')[0] if 0 <= i < len(tokens) else None
            for i in range(trigger_word_index-self.window_size, trigger_word_index+self.window_size+1)
                if i != trigger_word_index
        ]
        context_index = self.words_2_index(context_word)
        target_word_index = self.word_2_index(tokens[trigger_word_index].split('///')[0])

        return target_word_index, context_index, category

    def context_word_and_entity_feature(self, sent_info):
        tokens = sent_info["tokens"]
        trigger_word = sent_info["trigger_word"]
        if trigger_word is not None:
            trigger_word_index = trigger_word[2]
            category = trigger_word[1][:-8]
        else:
            trigger_word_index = random.randint(0, len(tokens)-1)
            category = "none"
        context_word = [
            tokens[i].split('///')[0] if 0 <= i < len(tokens) else None
            for i in range(trigger_word_index-self.window_size, trigger_word_index+self.window_size+1)
                if i != trigger_word_index
        ]
        context_index = self.words_2_index(context_word)
        target_word_index = self.word_2_index(tokens[trigger_word_index].split('///')[0])
        entities = [
            tokens[i].split('///')[2] if 0 <= i < len(tokens) else None
            for i in range(trigger_word_index-self.window_size, trigger_word_index+self.window_size+1)
                if i != trigger_word_index
        ]
        entity_index = self.entities_2_index(entities)

        return target_word_index, context_index, entity_index, category
    
    # remove puntuation?
    def all_context_word_and_entity_feature(self, sent_info, negative=True):
        try:
            tokens = [PATTERN.match(token).groups() for token in sent_info["tokens"]]
        except AttributeError as e:
            print(e)
            from pprint import pprint
            pprint(sent_info)
            quit()
        #pprint(sent_info) # added to see sent_info
        trigger_word = sent_info["trigger_word"]
        trigger_word_index = trigger_word[2] if trigger_word is not None else None
        split = sent_info["split"]
        if trigger_word is None:
            return None
        # if sent_info["sentence"] == "Kosikowski transferred in 1946 to the University of Notre Dame, which was regrouping under head coach Frank Leahy after losing many of its best players to the war effort.":
        #     pass
        result = []
        for index, token in enumerate(tokens):
            if token[3] in PUNCTUATION: 
                continue
            else:
                category = "none" if trigger_word_index is None or index != trigger_word_index else trigger_word[1]
                target_word_index = self.word_2_index(token[0])
                context_word = [
                    tokens[i][0] if 0 <= i < len(tokens) else None
                    for i in range(index-self.window_size, index+self.window_size+1)
                        if i != index
                ]
                context_index = self.words_2_index(context_word)
                # change for quant entity
                quantity_words = [
                    tokens[i][0] if 0 <= i < len(tokens) else None
                    for i in range(index - self.window_size, index + self.window_size + 1)
                    if i != index
                    ]
                quantity_index = self.quantity_type(quantity_words)
                verb_words = [
                    (tokens[i][0], tokens[i][3]) if 0 <= i < len(tokens) else None
                    for i in range(index - self.window_size, index + self.window_size + 1)
                    if i != index
                    ]
                verb_index = self.verb_type(verb_words, sent_info["sentence"])
                entities = [
                    tokens[i][2] if 0 <= i < len(tokens) else None
                    for i in range(index-self.window_size, index+self.window_size+1)
                        if i != index
                ]
                entity_index = self.entities_2_index(entities)
                arguments = [
                    int(tokens[i][4]) if 0 <= i < len(tokens) else 0
                    for i in range(index-self.window_size, index+self.window_size+1)
                        if i != index
                ]
                result.append((
                    target_word_index, 
                    context_index, 
                    entity_index,
                    category,
                    arguments,
                    split,
                    quantity_index,     # change for quant entity
                    verb_index          # change for verb entity
                ))
                # print(result)

        return result




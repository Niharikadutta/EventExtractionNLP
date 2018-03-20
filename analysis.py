import json
from pprint import pprint
import numpy as np
import csv
from sklearn.metrics import precision_recall_fscore_support

def data_statistic():
    file_name = '../data/event_detection.data.filter.json'
    with open(file_name, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    positive_sentences_count = sum([1 for d in data if d["trigger_word"] is not None])
    positive_words_count = sum([len(d['tokens']) for d in data if d['trigger_word'] is not None])

    # category
    category_count = {}
    for d in data:
        if d['trigger_word'] is None:
            continue
        category = d['trigger_word'][1]
        if category not in category_count:
            category_count[category] = 0
        category_count[category] += 1
    pprint(category_count)

def analyze_result():
    file_name = "../result/testing_upsampling_280.res"
    confusion_matrix = np.zeros((15, 15), dtype=int)
    true = []
    predict = []
    with open(file_name, 'r', encoding='utf-8') as infile:
        for line in infile:
            p, s = line.strip().split(', ')
            p, s = int(p), int(s)
            confusion_matrix[p, s] += 1
            true.append(s)
            predict.append(p)
    
    with open(file_name+'.csv', 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(confusion_matrix.tolist())

    result = precision_recall_fscore_support(true, predict, average="macro")
    print("macro", result)
    result = precision_recall_fscore_support(true, predict, average="micro")
    print("micro", result)



if __name__ == "__main__":
    #data_statistic()
    analyze_result()

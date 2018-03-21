import json

def parse_file():
    file_name = '/home/appleternity/corpus/event_detection/wiki_sentence_annotated.with_trigger.tsv'
    with open(file_name, 'r', encoding='utf-8') as infile:
        data = []
        for index, line in enumerate(infile):
            row = line.split("\t")
            if index % 1000 == 0:
                print(index)
            sent = row[3]
            argument_list = []
            trigger_word = None
            for entity in row[4:]:
                entity = entity.replace(", ", "++++++++++")
                split_entity = entity.split(",")
                split_entity[2].replace("++++++++++", ", ")
                # print(split_entity)
                info, _, start, end = split_entity
                start, end = int(start), int(end)
                word = sent[start:end]
                if info == "negative":
                    continue
                if info[-7:] == "trigger":
                    trigger_word = (word, info, start, end)
                else:
                    argument_list.append((word, info, start, end))
            data.append({
                "sentence":sent,
                "trigger_word":trigger_word,
                "argument_list":argument_list
            })

    with open("../data/event_detection.json", 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4)


def add_entity():
    event_data_path = "/home/appleternity/workspace/lab/event_detection/data/event_detection.new.parse.json.index"
    with open(event_data_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)    

    file_name = '/home/appleternity/corpus/event_detection/wiki_sentence_annotated.with_trigger.tsv'
    with open(file_name, 'r', encoding='utf-8') as infile:
        for index, line in enumerate(infile):
            row = line.split('\t')
            sent = row[3]
            for entity in row[4:]
                entity = entity.replace(", ", "++++++++++")
                split_entity = entity.split(",")
                split_entity[2].replace("++++++++++", ", ")
                info, _, start, end = split_entity
                start, end = int(start), int(end)
                word = sent[start:end]
                if info[-7:] == "trigger":
                    continue
                if info == "negative":
                    
    

def test_same_sent():
    file_name = '/home/appleternity/corpus/event_detection/wiki_sentence_annotated.with_trigger.tsv'
    with open(file_name, 'r', encoding='utf-8') as infile:
        sentences = [
            line.split("\t")[3]
            for line in infile
        ]
    dictionary = {}
    for sent in sentences:
        if sent not in dictionary:
            dictionary[sent] = 1
        else:
            dictionary[sent] += 1
    same_count = sum([1 for key, val in dictionary.items() if val != 1])
    print("same_count:", same_count)

def test_trigger_num():
    file_name = '/home/appleternity/corpus/event_detection/wiki_sentence_annotated.with_trigger.tsv'
    with open(file_name, 'r', encoding='utf-8') as infile:
        multi_trigger_count = 0
        for line in infile:
            row = line.split('\t')
            trigger_count = 0
            for entity in row[4:]:
                entity = entity.replace(", ", "++++++++++")
                split_entity = entity.split(",")
                split_entity[2].replace("++++++++++", ", ")
                info, _, start, end = split_entity
                if info[-7:] == "trigger":
                    trigger_count += 1
            if trigger_count > 1:
                multi_trigger_count += 1

    print("multi_trigger_count:", multi_trigger_count)

if __name__ == "__main__":
    #parse_file()
    #test_same_sent()
    test_trigger_num()

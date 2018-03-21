import requests
import json
import random
from pprint import pprint
import re

def parse_sentences(sent_list):
    res = requests.post(
        'http://localhost:9000/?properties={"annotators": "depparse,parse", "outputFormat": "json"}', 
        data="\n".join(sent_list).encode("utf-8")
    )
    data = json.loads(res.content.decode("utf-8"))
    if len(sent_list) != len(data["sentences"]):
        print("ERROR: Lengths don't match!")
        return None
    parse_result = []
    for sent, parse_info in zip(sent_list, data["sentences"]):
        parsing_tree = parse_info["parse"]
        tokens = parse_info["tokens"]
        dep = parse_info["enhancedPlusPlusDependencies"]
        parse_result.append({
            "sentence":sent,
            "tokens":tokens,
            "parsing_tree":parsing_tree,
            "dependency":dep
        })
    return parse_result

def parse_pos(sent_info):
    res = requests.post(
        'http://localhost:9000/?properties={"annotators": "pos", "outputFormat": "json"}', 
        data=sent_info["sentence"].encode("utf-8")
    )
    data = json.loads(res.content.decode('utf-8'))
    for index, (token, t) in enumerate(zip(sent_info["tokens"], data["sentences"][0]["tokens"])):
        sent_info['tokens'][index] = token + '///' + t["pos"]

def parse_ner(sent_info):
    res = requests.post(
        'http://localhost:9000/?properties={"annotators": "ner", "outputFormat": "json"}', 
        data=sent_info["sentence"].encode("utf-8")
    )
    data = json.loads(res.content.decode("utf-8"))
    if len(sent_info["tokens"]) != len(data["sentences"][0]["tokens"]):
        print("WTF")
    for index, (token, t) in enumerate(zip(sent_info["tokens"], data['sentences'][0]['tokens'])):
        sent_info["tokens"][index] = token+"///"+t["ner"]

def parse_sentence(sent):
    res = requests.post(
        'http://localhost:9000/?properties={"annotators": "depparse,parse", "outputFormat": "json"}', 
        data=sent.encode("utf-8")
    )
    data = json.loads(res.content.decode("utf-8"))
    data = data['sentences'][0]
    # print(data)
    tokens = [
        "{}///{}-{}".format(
            token["originalText"], 
            token["characterOffsetBegin"], 
            token["characterOffsetEnd"]
        ) 
        for token in data['tokens']
    ]

    dependency = [
        "{}///{}-{}".format(
            rule["dep"],
            rule["governor"],
            rule["dependent"]
        )
        for rule in data['enhancedPlusPlusDependencies']
    ]
    result = {
        "dep":" ".join(dependency),
        "tokens":tokens,
        "parsing":data['parse'],
        "sent":sent
    }
    return result

def parse_event_data():
    from datetime import datetime
    from pprint import pprint

    file_name = "../data/event_detection.new.parse.json"
    with open(file_name, 'r', encoding='utf-8') as infile:
        event_data = json.load(infile)

    total_length = len(event_data)
    error = 0
    need_parsing_count = sum([1 for sent in event_data if "tokens" in sent])
    print("need_parsing_count:", need_parsing_count)
    for i, info in enumerate(event_data, 1):
        print("\rProcessing {:>6} / {}".format(i, total_length), datetime.now(), end="")
        if "tokens" in info: continue
        try:
            result = parse_sentence(info["sentence"])
        except Exception as e:
            error += 1
            print()
            print(e)
            print(error)
            continue
        for key in ["tokens", "parsing", "dep"]:
            info[key] = result[key]
        
        if i % 100 == 0:
            with open(file_name+'.parse', "w", encoding='utf-8') as outfile:
                json.dump(event_data, outfile, indent=4)

    with open(file_name+".parse", "w", encoding='utf-8') as outfile:
        json.dump(event_data, outfile, indent=4)
        
def add_word_index():
    file_name = "../data/event_detection.data.json"
    with open(file_name, "r", encoding="utf-8") as infile:
        event_data = json.load(infile)

    error_file = open("../data/error.log", 'w', encoding='utf-8')
    error_count = 0
    total_count = len(event_data)
   
    filter_data = []
    for count, sent in enumerate(event_data):
        if count % 1000 == 0:
            print("\r{:>6} / {}".format(count, total_count), end='')
        if sent["trigger_word"] is not None and type(sent["trigger_word"][3]) != str:
            try:
                tokens = sent["tokens"]
                trigger_word = sent["trigger_word"]
                trigger_word_index = tokens.index("{}///{}-{}".format(
                    trigger_word[0],
                    str(trigger_word[2]),
                    str(trigger_word[3]),
                ))
                sent["trigger_word"] = [
                    trigger_word[0],
                    trigger_word[1],
                    trigger_word_index,
                    "{}-{}".format(trigger_word[2], trigger_word[3])
                ]
            except ValueError as e:
                error_count += 1
                error_file.write("{}\n{}\n{}\n\n".format(
                    sent["sentence"], "trigger_word", sent["trigger_word"][0]
                ))
                continue
        if sent["argument_list"] is not None:
            for i, arg in enumerate(sent['argument_list']):
                try:
                    tokens = sent['tokens']
                    arg_index = tokens.index("{}///{}-{}".format(
                        arg[0], str(arg[2]), str(arg[3])
                    ))
                    sent['argument_list'][i] = [
                        arg[0],
                        arg[1],
                        arg_index,
                        "{}-{}".format(arg[2], arg[3])
                    ]

                except ValueError as e:
                    #error_count += 1
                    #error_file.write("{}\n{}\n{}\n\n".format(
                    #    sent["sentence"], "argument", arg[0]
                    #))
                    pass
        filter_data.append(sent)

    print("\nerror_count:", error_count)
    with open(file_name+".filter", 'w', encoding='utf-8') as outfile:
        json.dump(filter_data, outfile, indent=4)

def add_ner_info():
    from pprint import pprint
    file_name = "../data/event_detection.new.parse.json.index"
    with open(file_name, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    total_count = len(data)
    for index, sent_info in enumerate(data):
        print("\r{:>6} {}".format(index, total_count), end='')
        parse_ner(sent_info)

    with open(file_name+'.ner', 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4)

def fix():
    reference_name = "../data/event_detection.json.parse"     
    with open(reference_name, 'r', encoding='utf-8') as infile:
        reference_data = json.load(infile)
        reference_data = {
            (
                sent_info["sentence"], 
                sent_info["trigger_word"][0] if sent_info["trigger_word"] is not None else None
            ) : sent_info
            for sent_info in reference_data
        }

    event_data_name = "../data/event_detection.json"
    with open(event_data_name, 'r', encoding='utf-8') as infile:
        event_data = json.load(infile)
    
    count = 0
    for index, sent in enumerate(event_data):
        if sent["trigger_word"] is not None:
            key = (sent["sentence"], sent["trigger_word"][0])
        else:
            key = (sent["sentence"], None)
        if key not in reference_data:
            count += 1
        else:
            event_data[index] = reference_data[key]

    print("Count:", count)
    
    output_file_name = "../data/event_detection.new.parse.json"
    with open(output_file_name, 'w', encoding='utf-8') as outfile:
        json.dump(event_data, outfile, indent=4)

def add_pos_info():
    file_name = '../data/event_detection.data.filter.json'
    with open(file_name, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    total_count = len(data)
    for index, sent_info in enumerate(data):
        print("\r{:>6} {}".format(index, total_count), end='')
        parse_pos(sent_info)

    with open(file_name+'.pos', 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4)

def add_train_eval_test_info():    
    file_name = '../data/event_detection.data.filter.json'
    with open(file_name, 'r', encoding='utf-8') as infile:
        data = json.load(infile) 
    total_count = len(data)
    index_list = [i for i in range(0, total_count)]
    random.shuffle(index_list)
    random.shuffle(index_list)
    random.shuffle(index_list)
    
    train_split = int(0.7*total_count)
    eval_split = int(0.8*total_count)
    
    train_index_list = {i for i in index_list[0:train_split]}
    eval_index_list = {i for i in index_list[train_split:eval_split]}
    test_index_list = {i for i in index_list[eval_split:]}

    for index, sent_info in enumerate(data):
        if index in train_index_list:
            sent_info["split"] = 0
        elif index in eval_index_list:
            sent_info["split"] = 1
        elif index in test_index_list:
            sent_info["split"] = 2
    
    with open(file_name+'.split', 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4)

def add_argument_info():
    file_name = "../data/event_detection.data.filter.json"
    with open(file_name, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    pattern = re.compile(r"(.+?)///(\d+)-(\d+)///(.+?)///(.+?)")

    error_count = 0
    error_list = []
    for sent_info in data:
        try:
            tokens = [
                [int(p) for i, p in enumerate(pattern.match(token).groups()) if i == 1 or i == 2]
                for token in sent_info["tokens"]
            ]
        except ValueError as e:
            print(e)
            pprint(sent_info)
            exit(-1)

        # fix argument_list
        for i, arg in enumerate(sent_info["argument_list"]):
            if type(arg[3]) == str:
                s = int(arg[3].split('-')[0])
                e = int(arg[3].split('-')[1])
                sent_info["argument_list"][i] = [arg[0], arg[1], s, e]

        arg_list = sent_info["argument_list"]
        arg_label = [0 for t in tokens]
        
        for arg in arg_list:
            start, end = arg[2], arg[3]
            
            # find start_i and end_i
            start_i = -1
            end_i = -1
            for i, (s, e) in enumerate(tokens):
                if s > start:
                    start_i = i-1
                    break
            for i, (s, e) in enumerate(tokens):
                if e >= end:
                    end_i = i
                    break
            if start_i == -1 or end_i == -1:
                error_count += 1
                error_list.append(sent_info)

                if end_i == -1:
                    end_i = len(tokens)-1
            
            for i in range(start_i, end_i+1):
                arg_label[i] = 1
        
        sent_info["tokens"] = [
            token + "///" + str(label)
            for token, label in zip(sent_info["tokens"], arg_label)
        ]
    print("Error Count:", error_count)

    with open(file_name+'.arg', 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4)

    with open(file_name+'.error', 'w', encoding='utf-8') as outfile:
        json.dump(error_list, outfile, indent=4)

def sample_data():
    file_name = "../data/event_detection.data.filter.json"
    output_file_name = '../data/event_detection.data.sample.json'
    with open(file_name, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    data = random.sample(data, 500)
    with open(output_file_name, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4)

def main():
    sent_list = [
        "This is a very good situation.", 
        "How are you?"
    ]
    parsing_result = parse_sentences(sent_list)
    results = []
    for sent in sent_list:
        result = parse_sentence(sent)
        results.append(result)
    with open("test2.json", 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4)

if __name__ == "__main__":
    # main()
    # parse_event_data()
    # add_word_index()
    # fix()
    # add_ner_info()
    # add_pos_info()
    # add_train_eval_test_info()
    # add_argument_info()
    sample_data()

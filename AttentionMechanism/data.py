import json
import util
import numpy as np

class Data:
    def __init__(self):
        pass

    def load_data(self, file_path, extractor):
        with open(file_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
            # data = [extractor(d) for d in data]
            # data = [d for d in data if d is not None]
            extracted_data = []
            length = len(data)
            for i, d in enumerate(data):
                print("\rLoading Data... {:>7}/{:>7} [{:>3.2f}%]".format(i, length, i/length*100), end='')
                res = extractor(d)
                if res is not None:
                    extracted_data.append(res)
            print("\nFinish Loading Data...")
            return extracted_data
   
    def to_category(self, samples):
        dictionary = {}
        results = []
        for sample in samples:
            if sample not in dictionary:
                dictionary[sample] = len(dictionary)
            results.append(dictionary[sample])

        result_array = np.zeros((len(results), len(dictionary)), dtype=np.int32)
        for index, result in enumerate(results):
            result_array[index, result] = 1
        
        return result_array, dictionary

    def to_dict_category(self, samples, dictionary):
        num_category = len({val for val in dictionary.values()})
        result_array = np.zeros((len(samples), num_category), dtype=np.int32)
        for index, sample in enumerate(samples):
            result_array[index, dictionary[sample]] = 1
        return result_array


if __name__ == "__main__":
    from pprint import pprint
    from feature import Feature

    glove_path = "/home/appleternity/corpus/glove/glove.6B.50d"
    event_data_path = "/home/appleternity/workspace/lab/event_detection/data/event_detection.new.parse.json.index"
    word_dictionary, matrix = util.load_embedding(glove_path)
    feature = Feature(word_dictionary)
    data_manager = Data()

    # context_word
    data = data_manager.load_data(event_data_path, feature.context_word_feature)
    x = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])
    print("x.shape = ", x.shape)
    print("y.shape = ", y.shape)
    print(y)
    y, category_dictionary = data_manager.to_category(y)
    print("y.shape = ", y.shape)
    pprint(category_dictionary)
    
    # context_word and entities type
    data = data_manager.load_data(event_data_path, feature.context_word_and_entity_feature)
    x_word   = np.array([d[0] for d in data])
    x_entity = np.array([d[1] for d in data])
    y        = np.array([d[2] for d in data])
    print("x_word.shape = ", x_word.shape)
    print("x_entity.shape = ", x_entity.shape)
    print("y.shape = ", y.shape)
    x_entity, x_entity_dictionary = data_manager.to_category(x_entity)
    y, y_dictionary = data_manager.to_category(y)
    print("x_entity.shape = ", x_entity.shape)
    print("y.shape = ", y.shape)



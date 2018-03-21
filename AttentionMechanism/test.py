import re
import json

# import GraphNode as g
import itertools as it
from itertools import chain
from nltk.corpus import wordnet
from duplicity.path import Path


class FeatureExtractor:

    def get_tokens(self, raw_tokens):
        PATTERN = re.compile(r"(.+?)///(\d+-\d+)///(.+?)///(.+?)///(.+?)")
        tokens = [PATTERN.match(token).groups() for token in raw_tokens]
        return tokens

    def get_similar_words(self, target_word, how_many=10):
        synonyms = wordnet.synsets(target_word)
        set_of_syns = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
        return it.islice(set_of_syns, how_many)

    def create_graph(self, graph_node, value, has_child_dict, total_nodes):
        graph_node.value = value
        total_nodes[value] = graph_node
        if value in has_child_dict:
            children = has_child_dict.get(value)
            child_nodes = []
            for child in children:
                if child in total_nodes:
                    child_node = total_nodes.get(child)
                else:
                    child_node = GraphNode()
                self.create_graph(child_node,child,has_child_dict,total_nodes)
                child_nodes.append(child_node)
            graph_node.children = child_nodes

    def find_from_below(self,child,final_scores_dict,running_scores_dict, level):
        running_scores_dict[child.value] = level

        children = child.children
        if len(children)>0:
            for ch in children:
                if ch.value not in final_scores_dict:
                    self.find_from_below(ch, final_scores_dict, running_scores_dict, level+1)
        else:
            for key in running_scores_dict:
                final_scores_dict[key] = running_scores_dict.get(key)
            running_scores_dict = {}

    def find_from_above(self, graph_node, final_scores_dict, running_scores_dict, par_scores_dict, target_word, words_dict):
        if words_dict[graph_node.value] == target_word:
            for key in running_scores_dict:
                final_scores_dict[key] = running_scores_dict[key]
            self.find_from_below(graph_node,final_scores_dict,{},0)
        else:
            if graph_node.value not in running_scores_dict:
                children = graph_node.children
                if len(children)>0:
                    running_scores_dict[graph_node.value] = 1
                    for s in par_scores_dict:
                        par_scores_dict[s] += 1
                    for s in par_scores_dict:
                        running_scores_dict[s] = par_scores_dict[s]
                    par_scores_dict[graph_node.value] = 1
                    for child in children:
                        self.find_from_above(child, final_scores_dict, running_scores_dict, par_scores_dict, target_word, words_dict)


    def get_all_paths(self, root_node, all_paths, path=[]):
        val = root_node.value
        path.append(val)
        children = root_node.children
        if len(children)>0:
            for child in children:
                path = self.get_all_paths(child,all_paths,path)
            path = path[:-1]
        else:
#             print path
            all_paths.append(path)
            path = path[:-1]
        return path
        
    def get_dependency_weight(self, tokens_list, dep_tree_info, target_word):

        token_words_list = [tok[0] for tok in tokens_list]

        i = 1
        words_dict = {}
        words_dict[str(0)] = "r00t"
        for word in token_words_list:
            words_dict[str(i)] = word
            i += 1

        pattern = re.compile(r"(.+?)///(\d+)-(\d+)")

        deps = re.findall(pattern,dep_tree_info)
        has_par = set()
        has_par_dict = {}
        has_child_dict = {}
        for d_rel, arg1_index, arg2_index in deps:
            if arg2_index not in has_par:
                has_par.add(arg2_index)

            if arg1_index in has_child_dict:
                tmp_list = has_child_dict.get(arg1_index)
            else:
                tmp_list = []
            tmp_list.append(arg2_index)
            has_child_dict[arg1_index] = tmp_list

            if arg2_index in has_par_dict:
                tmp1_list = has_par_dict.get(arg2_index)
            else:
                tmp1_list = []
            tmp1_list.append(arg1_index)
            has_par_dict[arg2_index] = tmp1_list

        roots = []
        for i in range(0,tokens_list.__len__()):
            if not has_par.__contains__(str(i)):
                roots.append(str(i))

        graphs_list = []
        total_nodes = {}
        for root in roots:
            graph_node = GraphNode()
            self.create_graph(graph_node, root, has_child_dict,total_nodes)
            graphs_list.append(graph_node)


# List all the paths in the graphs:
        for root_node in graphs_list:
            all_paths = []
            self.get_all_paths(root_node,all_paths,[])
            print all_paths
            
            final_scores_dict = {}
            for path in all_paths:
                indx = -1
#                 found = False
                for i in range(0,len(path)):
                    if words_dict[path[i]] == target_word:
                        indx = i
#                         found = True
                        break
                if indx!=-1:
                    tmp = indx
                    for j in range(0,indx):
                        if path[j] in final_scores_dict:
                            if final_scores_dict[path[j]] > tmp:
                                final_scores_dict[path[j]] = tmp
                        else:
                            final_scores_dict[path[j]] = tmp
                        
                        tmp = tmp-1
                    
                    counter = 1
                    for k in range(indx+1,len(path)):
                        if path[k] in final_scores_dict:
                            if final_scores_dict[path[k]] > counter:
                                final_scores_dict[path[k]] = counter
                        else:
                            final_scores_dict[path[k]] = counter
                        
                        counter = counter+1

#         final_scores_dict = {}
#         running_scores_dict = {}
#         par_scores_dict = {}
#         for root_node in graphs_list:
#             if words_dict[root_node.value] == target_word:
#                 children = root_node.children
#                 if len(children)>0:
#                     for child in children:
#                         self.find_from_below(child,final_scores_dict,{},1)
#                 print "hello"
#             else:
#                 running_scores_dict[root_node.value] = 1
#                 par_scores_dict[root_node.value] = 1
#                 children = root_node.children
#                 if len(children)>0:
#                     for child in children:
#                         self.find_from_above(child,final_scores_dict,running_scores_dict,par_scores_dict,target_word,words_dict)
#                 else:
#                     print "test"

        result = []
        for i in range(0,words_dict.__len__()):
            if str(i) in final_scores_dict:
                if final_scores_dict.get(str(i)) != 0:
                    result.append(1/float(final_scores_dict.get(str(i))))
                else:
                    result.append(0)
            else:
                result.append(0)

        # print final_scores_dict
        return result

    def __init__(self):
        print "Feature Extraction initiated"


class GraphNode:
    value = None
    children = []
    edge_labels = []

if __name__ == "__main__":
    fe = FeatureExtractor()

    s = fe.get_similar_words("appeared",2)
    print "Similar words are: " + [ss for ss in s]

    data = json.loads(open('test1.json').read().decode('utf-8-sig'))

    for d in data:
        tokens = fe.get_tokens(d["tokens"])
        dep_weights = fe.get_dependency_weight(tokens, d["dep"], "appeared")
        print dep_weights
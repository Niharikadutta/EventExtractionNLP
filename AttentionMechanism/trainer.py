from data import Data
from feature import Feature
import util
from model import AttentationContextModel
from config import Config
from config import entity_type_dictionary, trigger_type_dictionary, binary_trigger_type_dictionary
from config import remove_list_less_50, remove_list_less_100, reinforce_list
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import precision_recall_fscore_support
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Trainer():
    def __init__(self):
        pass

    # last elem of data have to be y
    def sample_negative(self, data, ratio=5):
        y = data[-1]
        negative_index = [i for i, _y in enumerate(y) if _y[0]==1]
        positive_index = [i for i, _y in enumerate(y) if _y[0]!=1]
        negative_num = int(len(positive_index)*ratio)
        negative_index_sample = np.random.choice(negative_index, negative_num, True)

        new_data = []
        for d in data:
            d_pos = d[positive_index] 
            d_neg = d[negative_index_sample]
            if(len(d.shape)==1):
                new_d = np.hstack((d_pos, d_neg))
            else:
                new_d = np.vstack((d_pos, d_neg))
            new_data.append(new_d)
        return tuple(new_data)

    def upsampling(self, data, y_label, number=5000):
        print("length of y_label", len(y_label))
        print("shape of data", data[0].shape)

        # count number of labels
        labels_num = {}
        for index, label in enumerate(y_label):
            if label not in labels_num:
                labels_num[label] = []
            labels_num[label].append(index)

        # upsampling classes
        new_index_list = []
        for label, index_list in labels_num.items():
            if len(index_list) < number:
                sample_index_list = np.random.choice(index_list, number - len(index_list))
                new_index_list.append(sample_index_list)

        new_index_list = np.hstack(new_index_list)
        new_data = []
        for d in data:
            if (len(d.shape) == 1):
                new_data.append(np.hstack((d, d[new_index_list])))
            else:
                new_data.append(np.vstack((d, d[new_index_list])))
        return tuple(new_data)

    def get_batch_data(self, data, batch_size, phase="train"):
        if phase == 'train':
            data_count = data[0].shape[0]
            index = [i%data_count for i in range(0, int(np.ceil(data_count/batch_size)*batch_size))]
            index = np.array(index)
            np.random.shuffle(index)
            index = index.reshape((-1, batch_size))
        else:
            data_count = data[0].shape[0]
            index = [
                [i+j for j in range(0, batch_size) if i+j < data_count]
                for i in range(0, data_count, batch_size)
            ]
        for batch_index in index:
            res = [d[batch_index] for d in data]
            yield res

    def get_inverse_weight(self, y, dictionary):
        freq_dict = {}
        for key, val in dictionary.items():
            count = len(y[y==val])+1
            #print("{:>2} {:>50} {}".format(val, key, count))
            freq_dict[key] = count

        # inverse
        weight = {key:10.0/val for key, val in freq_dict.items()}

        return np.array([weight[_y] for _y in y])

    def get_weight(self, y, reinforce_list, weight):
        weight = [weight if _y in reinforce_list else 1 for _y in y]
        return np.array(weight)

    def gaussian(self, x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def alpha_attention(self, alpha, mode="normalize", window_size=None):
        if mode == "window":
            assert window_size is not None
            gaussian_value = self.gaussian(
                np.array([i for i in range(-window_size, window_size+1)]), 
                0.0, 
                1.0
            )
            new_alpha = np.zeros(alpha.shape, dtype=np.float32)
            size = alpha.shape[1]
            for i, _a in enumerate(alpha):
                for j, _aa in enumerate(_a):
                    if _aa == 1.0:
                        start = 0 if j-window_size < 0 else j-window_size
                        end = size if j+window_size+1 > size else j+window_size+1
                        g_start = window_size-j if window_size-j > 0 else 0
                        g_end = j+window_size-size+1 if j+window_size-size+1 > 0 else 0
                        if g_end == 0:
                            new_alpha[i, start:end] += gaussian_value[g_start:]
                        else:
                            new_alpha[i, start:end] += gaussian_value[g_start:-g_end]
            alpha = new_alpha

        norm_alpha = np.sum(alpha, axis=1).reshape(-1, 1)+1e-20
        return alpha/norm_alpha

    def pack_data(self, data_list, index_list):
        return tuple([d[index_list] for d in data_list])

    def data_filter(self, data_list, y_label, remove_list):
        index_list = np.array([y not in remove_list for y in y_label])
        new_y_label = [y for i, y in zip(index_list, y_label) if i]
        return tuple(d[index_list] for d in data_list), new_y_label

    # train the model without alpha_learning
    def train_normal(self, config):
        # load data
        word_dictionary, matrix = util.load_embedding(config.glove_path)
        feature = Feature(
            word_dictionary, 
            entity_type_dictionary,
            window_size=config.context_window_size
        )
        data_manager = Data()
        data = data_manager.load_data(
            config.event_data_path, 
            feature.all_context_word_and_entity_feature
        )
        x_target = np.array([dd[0] for d in data for dd in d])
        x_word   = np.array([dd[1] for d in data for dd in d])
        x_entity = np.array([dd[2] for d in data for dd in d])
        x_arg    = np.array([dd[4] for d in data for dd in d])
        split    = np.array([dd[5] for d in data for dd in d])
        x_quant = np.array([dd[6] for d in data for dd in d])  # change for quant
        y_label  = [dd[3] for d in data for dd in d]
        (x_target, x_word, x_entity, x_arg, split, x_quant), y_label = self.data_filter(
            (x_target, x_word, x_entity, x_arg, split, x_quant),
            y_label,
            config.remove_list
        )
        y        = data_manager.to_dict_category(y_label, config.trigger_dictionary)
        #y        = data_manager.to_dict_category(y, binary_trigger_type_dictionary)
        y_weight = self.get_weight(y_label, config.reinforce_list, 1)
        x_arg    = self.alpha_attention(x_arg, mode="window", window_size=3)

        print("x_target.shape = ", x_target.shape)
        print("x_word.shape =   ", x_word.shape)
        print("x_entity.shape = ", x_entity.shape)
        print("y.shape =        ", y.shape)
        print("y_label.length = ", len(y_label))

        self.training_data = self.pack_data((x_target, x_word, x_entity, x_arg, x_quant, y_weight, y), index_list=(split==0))
        self.eval_data     = self.pack_data((x_target, x_word, x_entity, x_quant, y), index_list=(split==1))
        self.testing_data  = self.pack_data((x_target, x_word, x_entity, x_quant, y), index_list=(split==2))
        
        # sample
        self.training_data = self.upsampling(
            self.training_data,
            [y_label[i] for i, s in enumerate((split==0)) if s],
            10000
        )
        self.training_data = self.sample_negative(
            self.training_data, 
            ratio=config.negative_ratio,
        )

        print("training_data:   ", self.training_data[0].shape[0])
        print("eval_data:       ", self.eval_data[0].shape[0])
        print("testing_data:    ", self.testing_data[0].shape[0])

        # update config
        config.output_size = y.shape[1]
        config.embedding_size = matrix.shape[0]
        config.embedding_dim = matrix.shape[1]
        config.data_count = int(np.ceil(y.shape[0]/config.batch_size)*config.batch_size)
        config.train_size = int(np.ceil(self.training_data[-1].shape[0]/config.batch_size)*config.batch_size)

        # build model
        model_manager = AttentationContextModel()
        #sess = tf.InteractiveSession()
        sess = tf.Session()
        with tf.device(config.machine):
            input_context_words = tf.placeholder(tf.int32, [None, config.context_window_size*2], name="input_context_words")
            input_target_word = tf.placeholder(tf.int32, [None], name="input_target_word")
            input_entities = tf.placeholder(tf.int32, [None, config.context_window_size*2], name="input_entities")
            output = tf.placeholder(tf.int32, [None, config.output_size], name="output")
            target_alpha = tf.placeholder(tf.float32, [None, config.context_window_size*2], name="target_alpha")
            output_weight = tf.placeholder(tf.float32, [None], name='output_weight')
            input_quant = tf.placeholder(tf.int32, [None, config.context_window_size*2], name="input_quantities")

            alpha, predict, drop_rate, regularizer = model_manager.build_model(
                input_target_word,
                input_context_words,
                input_entities,
                matrix,
                config,
                verbose=False
            )
        with tf.device("/cpu:0"):
            # define loss, optimizer, initialization
            loss_softmax = tf.losses.softmax_cross_entropy(output, predict)
            loss_alpha = tf.losses.mean_squared_error(alpha, target_alpha)
            loss = loss_softmax + config.lamb*loss_alpha
            
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            print(reg_variables)
            reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            loss += reg_term

            #optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)
            #optimizer = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(loss)
            #optimizer_alpha = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss_alpha)
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss)

            output_ans = tf.argmax(output, axis=1)
            predict_ans = tf.argmax(predict, axis=1)
            acc = tf.reduce_mean(tf.cast(tf.equal(output_ans, predict_ans), tf.float32))
            pre = tf.metrics.precision(output_ans, predict_ans)
            rec = tf.metrics.recall(output_ans, predict_ans)

        # start training
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables())
        #saver.restore(sess, "../model/attention.model.weighted-3000")
        for e in range(1, config.epochs+1):
            loss_list = []
            acc_list = []
            pre_list = []
            rec_list = []
            p_ans_list = []
            o_ans_list = []
            # training
            for i, (x_target, x_word, x_entity, x_alpha, x_quant, y_weight, y) in enumerate( # change for quant
                self.get_batch_data(self.training_data, config.batch_size)
            ):
                l, _, a, o_ans, p_ans = sess.run(
                    [loss, optimizer, acc, output_ans, predict_ans],
                    feed_dict = {
                        input_target_word:x_target,
                        input_context_words:x_word,
                        input_entities:x_entity,
                        target_alpha:x_alpha,
                        input_quant:x_quant,  # change for quant
                        output_weight:y_weight,
                        output:y,
                        drop_rate:config.drop_rate,
                    }
                )
                loss_list.append(l)
                acc_list.append(a)
                #pre_list.append(_pre)
                #rec_list.append(_rec)
                p_ans_list.extend(p_ans)
                o_ans_list.extend(o_ans)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = precision_recall_fscore_support(o_ans_list, p_ans_list, average="macro")

                print("\rEpoch: {:>4}, {:>7} / {:>7} [{:>2.2f}%], Loss:{:>2.3f}, Acc:{:>2.2f}, Prec:{:>2.2f}, Rec:{:>2.2f}, F1:{:>2.2f}".format(
                    e,
                    i*config.batch_size,
                    config.train_size,
                    i*config.batch_size/config.train_size*100.0,
                    sum(loss_list)/len(loss_list),
                    sum(acc_list)/len(acc_list)*100,
                    #sum(pre_list)/len(pre_list)*100,
                    #sum(rec_list)/len(rec_list)*100
                    result[0]*100,
                    result[1]*100,
                    result[2]*100
                ), end='')
            print()
            
            # evaluation
            eval_acc_list = []
            eval_pre_list = []
            eval_rec_list = []
            p_ans_list = []
            o_ans_list = []
            for i, (x_target, x_word, x_entity, x_quant, y) in enumerate(
                self.get_batch_data(self.eval_data, config.batch_size, phase="eval")
            ):
                a, p_ans, o_ans = sess.run(
                    [acc, predict_ans, output_ans],
                    feed_dict = {
                        input_target_word:x_target,
                        input_context_words:x_word,
                        input_entities:x_entity,
                        input_quant:x_quant,
                        output:y
                    }
                )
                eval_acc_list.append(a)
                #eval_pre_list.append(_pre)
                #eval_rec_list.append(_rec)
                p_ans_list.extend(p_ans)
                o_ans_list.extend(o_ans)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")    
                result = precision_recall_fscore_support(o_ans_list, p_ans_list, average="macro")

            print("evaluation acc = {:>2.3f}, prec = {:>2.3f}, rec = {:>2.3f}, f1 = {:>2.3f}\n".format(
                sum(eval_acc_list) / len(eval_acc_list) * 100.0,
                #sum(eval_pre_list) / len(eval_pre_list) * 100.0,
                #sum(eval_rec_list) / len(eval_rec_list) * 100.0
                result[0]*100, result[1]*100, result[2]*100
            ))
            
            if e%10 == 0:
                if e%100 == 0:
                    saver.save(sess, 'D:\Academic\ASU\Sem 4\\NLP\project\dataset\\attention.model.upsampling', global_step=e)
        
                # testing
                #saver.save(sess, '../model/attention.model')
                test_acc_list = []
                test_pre_list = []
                test_rec_list = []
                test_p_ans_list = []
                test_o_ans_list = []
                for i, (x_target, x_word, x_entity, x_quant, y) in enumerate(
                    self.get_batch_data(self.testing_data, config.batch_size, phase="test")
                ):
                    a, p, ans, (_pre, _), (_rec, _) = sess.run(
                        [acc, predict_ans, output_ans, pre, rec],
                        feed_dict = {
                            input_target_word:x_target,
                            input_context_words:x_word,
                            input_entities:x_entity,
                            input_quant:x_quant,
                            output:y
                        }
                    )
                    test_acc_list.append(a)
                    test_pre_list.append(_pre)
                    test_rec_list.append(_rec)
                    test_p_ans_list.extend(p)
                    test_o_ans_list.extend(ans)

                with open("D:\Academic\ASU\Sem 4\\NLP\project\dataset\\testing_upsampling_{}.res".format(e), 'w', encoding='utf-8') as outfile:
                    for _p, _ans in zip(test_p_ans_list, test_o_ans_list):
                        outfile.write("{}, {}\n".format(_p, _ans))

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = precision_recall_fscore_support(test_o_ans_list, test_p_ans_list, average="macro")
        
                print("testing acc = {:>2.3f}, prec = {:>2.3f}, rec = {:>2.3f}, f1 = {:>2.3f}".format(
                    sum(test_acc_list) / len(test_acc_list) * 100,
                    #sum(test_pre_list) / len(test_pre_list) * 100,
                    #sum(test_rec_list) / len(test_rec_list) * 100
                    result[0]*100, result[1]*100, result[2]*100
                ))


def train_model():

    # Update the glove pretrained model path and the data path in the path variables below
    glove_path = 'D:\Academic\ASU\Sem 4\\NLP\project\dataset\glove.6B.300d'
    data_path = "D:\Academic\ASU\Sem 4\\NLP\project\dataset\event_detection.data.sample.json"

    trigger_type = sorted(trigger_type_dictionary.items(), key=lambda x: x[1])
    trigger_type = [t for t, i in trigger_type if t not in remove_list_less_100]
    trigger_dictionary = {t:i for i, t in enumerate(trigger_type)}

    config = Config(
        # glove_path=os.path.join(home_dir, "corpus/glove/glove.6B.300d"),
        # event_data_path=os.path.join(home_dir, "workspace/lab/event_detection/data/event_detection.data.filter.json"),
        glove_path=glove_path,
        event_data_path=data_path,
        entity_dim=50,
        entity_size=len(entity_type_dictionary),
        context_window_size=5,
        batch_size=128,
        epochs=50,
        machine="/cpu:0",
        negative_ratio=0.1,
        lamb=5,
        drop_rate=0.6,
        remove_list=remove_list_less_100,
        trigger_dictionary=trigger_dictionary,
        reinforce_list=reinforce_list
    )

    trainer = Trainer()
    trainer.train_normal(config)

if __name__ == "__main__":
    train_model()    

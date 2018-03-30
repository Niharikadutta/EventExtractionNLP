import tensorflow as tf
import prettytensor as pt
from tensorflow import layers

class AttentationContextModel:
    
    def __init__(self):
        pass

    def build_model(self, 
                    input_target_word,
                    input_context_words, 
                    input_entities, 
                    embedding_matrix,
                    config,
                    verbose=False):
        with tf.device(config.machine):
            # setting
            embedding_dim = embedding_matrix.shape[1]
            entity_size = config.entity_size
            entity_dim = config.entity_dim
            output_size = config.output_size
            context_size = config.context_window_size*2
            regularizer = tf.contrib.layers.l2_regularizer(scale=1e-6)
            dropout_prob = tf.placeholder_with_default(1.0, shape=())

            # embedding layers
            word_embedding = tf.get_variable(
                name="word_embedding",
                initializer=embedding_matrix,
                trainable=False
            )
            entity_embedding = tf.get_variable(
                name="entities_embedding",
                shape=[entity_size, entity_dim],
                # initializer=tf.initializers.truncated_normal(),
                initializer=None,
                trainable=True,
                dtype=tf.float32
            )

            # context_word_embeddings = (?, word_num, embedding_dim)
            # target_word_embedding = (?, embedding_dim)
            # entities_embeddings = (?, )
            context_word_embeddings = tf.nn.embedding_lookup(
                word_embedding, input_context_words
            )
            target_word_embedding = tf.nn.embedding_lookup(
                word_embedding, input_target_word
            )
            target_word_embedding = tf.reshape(
                target_word_embedding, shape=(-1, embedding_dim)
            )
            entities_embeddings = tf.nn.embedding_lookup(
                entity_embedding, input_entities
            )
           
            if verbose:
                print("context_word_embeddings", context_word_embeddings.get_shape())
                print("targe_word_embedding", target_word_embedding.get_shape())
                print("entities_embeddings", entities_embeddings.get_shape(), entities_embeddings)
        
            # alpha_w
            w_contexts = tf.nn.relu(
                layers.dense(
                    inputs=context_word_embeddings,
                    units=embedding_dim,
                    name="W_w",
                    reuse=False,
                    kernel_regularizer=regularizer,
                )
            )
            w_contexts = tf.nn.dropout(w_contexts, dropout_prob)
            w_target = tf.nn.relu(
                layers.dense(
                    inputs=target_word_embedding,
                    units=embedding_dim,
                    name="W_w",
                    reuse=True,
                    kernel_regularizer=regularizer,
                )
            )
            w_target = tf.nn.dropout(w_target, dropout_prob)
            if verbose:
                print("w_contexts", w_contexts.get_shape())
                print("w_target", w_target.get_shape())
            w_target = tf.reshape(w_target, shape=(-1, 1, embedding_dim))
            weight_contexts_ori = tf.matmul(w_target, w_contexts, transpose_b=True)
            weight_contexts = tf.reshape(weight_contexts_ori, (-1, context_size))
            alpha_w = tf.nn.softmax(weight_contexts)
            
            if verbose:
                print("weight_contexts", weight_contexts.get_shape())
                print("alpha_w", alpha_w.get_shape())

            # alpha_e
            w_e = tf.nn.relu(
                layers.dense(
                    inputs=target_word_embedding,
                    units=entity_dim,
                    name="W_e",
                    reuse=False,
                    kernel_regularizer=regularizer,
                )
            )
            w_e = tf.nn.dropout(w_e, dropout_prob)
            w_e = tf.reshape(w_e, shape=(-1, 1, entity_dim))
            weight_entities = tf.matmul(w_e, entities_embeddings, transpose_b=True)
            weight_entities = tf.reshape(weight_entities, (-1, context_size))
            alpha_e = tf.nn.softmax(weight_entities)
            if verbose:
                print("weight_entities", weight_entities.get_shape())
                print("alpha_e", alpha_e.get_shape())

            # alpha
            alpha = alpha_w + alpha_e
            alpha = tf.reshape(alpha, shape=(-1, 1, context_size))

            # representation
            c_w = tf.reduce_sum(
                tf.matmul(alpha, context_word_embeddings),
                axis=1
            )
            c_e = tf.reduce_sum(
                tf.matmul(alpha, entities_embeddings),
                axis=1
            )
            representation = tf.concat([target_word_embedding, c_w, c_e], axis=1)
            representation = tf.nn.dropout(representation, dropout_prob)
            alpha = tf.reshape(alpha, shape=(-1, context_size))
            if verbose:
                print("representation", representation.get_shape())


            # prediction
            hidden = tf.nn.relu(
                layers.dense(
                    inputs=representation,
                    units=embedding_dim,
                    name="w_h_1",
                    kernel_regularizer=regularizer,
                )
            )
            hidden = tf.nn.dropout(hidden, dropout_prob)
            if verbose:
                print("hidden", hidden.get_shape())
            hidden = tf.nn.relu(
                layers.dense(
                    inputs=hidden,
                    units=50,
                    name="w_h_2",
                    kernel_regularizer=regularizer,
                )
            )
            hidden = tf.nn.dropout(hidden, dropout_prob)
            if verbose:
                print("hidden", hidden.get_shape())
            output = tf.nn.softmax(
                layers.dense(
                    inputs=hidden,
                    units=output_size,
                    name="w_h_3",
                    kernel_regularizer=regularizer
                )
            )
            if verbose:
                print("output", output.get_shape())

            return alpha, output, dropout_prob, regularizer

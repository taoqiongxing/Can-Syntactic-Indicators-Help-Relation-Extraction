import tensorflow as tf

from utils import initializer

def attention1(inputs, e1, e2, p1, p2, attention_size):

    def extract_entity(x, e):
        e_idx = tf.concat([tf.expand_dims(tf.range(tf.shape(e)[0]), axis=-1), tf.expand_dims(e, axis=-1)], axis=-1)
        return tf.gather_nd(x, e_idx)  # (batch, hidden)
    hidden_size = inputs.shape[2].value  
    e1_h = extract_entity(inputs, e1)  # (batch, hidden)
    e2_h = extract_entity(inputs, e2)  # (batch, hidden)

    u1 = tf.get_variable("u1", [hidden_size], initializer=initializer())
    u2 = tf.get_variable("u2", [hidden_size], initializer=initializer())

    trans =  tf.tanh(e1_h*u1 - e2_h*u2)
    v = tf.layers.dense(tf.concat([inputs, p1, p2], axis=-1), attention_size, use_bias=False, kernel_initializer=initializer())
    v = tf.nn.tanh(v)
    u_omega = tf.get_variable("u_omega", [attention_size], initializer=initializer())
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (batch, seq_len)
    alphas = tf.nn.softmax(vu, name='alphas')  # (batch, seq_len)
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)  # (batch, hidden)
    output = output*trans

    return output, alphas, trans

def relation_words_attention(r0, r1, r2, r3, latent_relation_num, latent_relation_size):
    # Latent Entity Type Vectors
    latent_relation_type = tf.get_variable("latent_relation_type", shape=[latent_relation_num, latent_relation_size], initializer=initializer())

    r0_sim = tf.matmul(r0, tf.transpose(latent_relation_type))  # (batch, num_type)
    r0_alphas = tf.nn.softmax(r0_sim, name='r0_alphas')  # (batch, num_type)
    r0_type = tf.matmul(r0_alphas, latent_relation_type, name='r0_type')  # (batch, hidden)

    r1_sim = tf.matmul(r1, tf.transpose(latent_relation_type))  # (batch, num_type)
    r1_alphas = tf.nn.softmax(r1_sim, name='r1_alphas')  # (batch, num_type)
    r1_type = tf.matmul(r1_alphas, latent_relation_type, name='r1_type')  # (batch, hidden)

    r2_sim = tf.matmul(r2, tf.transpose(latent_relation_type))  # (batch, num_type)
    r2_alphas = tf.nn.softmax(r2_sim, name='r2_alphas')  # (batch, num_type)
    r2_type = tf.matmul(r2_alphas, latent_relation_type, name='r2_type')  # (batch, hidden)

    r3_sim = tf.matmul(r3, tf.transpose(latent_relation_type))  # (batch, num_type)
    r3_alphas = tf.nn.softmax(r3_sim, name='r3_alphas')  # (batch, num_type)
    r3_type = tf.matmul(r3_alphas, latent_relation_type, name='r3_type')  # (batch, hidden)

    return r0_type, r1_type, r2_type, r3_type

def multihead_attention(queries, keys, num_units, num_heads,
                        dropout_rate=0, scope="multihead_attention", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Linear projections
        Q = tf.layers.dense(queries, num_units, kernel_initializer=initializer())  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, kernel_initializer=initializer())  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, kernel_initializer=initializer())  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs /= K_.get_shape().as_list()[-1] ** 0.5

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        alphas = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        alphas *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        alphas = tf.layers.dropout(alphas, rate=dropout_rate, training=tf.convert_to_tensor(True))

        # Weighted sum
        outputs = tf.matmul(alphas, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Linear
        outputs = tf.layers.dense(outputs, num_units, activation=tf.nn.relu, kernel_initializer=initializer())

        # Residual connection
        outputs += queries

        # Normalize
        outputs = layer_norm(outputs)  # (N, T_q, C)

    return outputs, alphas

def multihead_attention2(queries, keys, num_units, num_heads,
                        dropout_rate=0.5, scope="multihead_attention2", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Linear projections
        Q = tf.layers.dense(queries, num_units, kernel_initializer=initializer())  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, kernel_initializer=initializer())  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, kernel_initializer=initializer())  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs /= K_.get_shape().as_list()[-1] ** 0.5

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        alphas = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        alphas *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        alphas = tf.layers.dropout(alphas, rate=dropout_rate, training=tf.convert_to_tensor(True))

        # Weighted sum
        outputs = tf.matmul(alphas, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Linear
        outputs = tf.layers.dense(outputs, num_units, activation=tf.nn.relu, kernel_initializer=initializer())

        # Residual connection
        outputs += queries

        # Normalize
        outputs = layer_norm(outputs)  # (N, T_q, C)

    return outputs, alphas


def layer_norm(inputs, epsilon=1e-8, scope="layer_norm", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

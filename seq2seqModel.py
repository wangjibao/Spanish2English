# _*_ coding: utf-8 _*_
"""
@author: Jibao Wang
@time: 2019/12/3 17:36
"""
import getConfig
import tensorflow as tf

config = getConfig.get_config("config.ini")

class Encoder(tf.keras.Model):  # 继承了 tf.keras.Model 类
    def __init__(self, enc_vocab_size, embedding_dim, enc_hidden_size, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_hidden_size = enc_hidden_size
        self.embedding = tf.keras.layers.Embedding(enc_vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_hidden_size, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    # 定义 RNN 结构
    def call(self, x, initial_state):
        # 输入 x 的维度 [batch_size, max_input_length]
        x = self.embedding(x)  # 此时 x 的维度：[batch_size, max_input_length, embedding_dim]
        # gru是一个GRU的实例化对象，如果该类实现了call函数，可以直接gru()这么调用相当与调用了 call 函数
        output, state = self.gru(x, initial_state=initial_state)
        return output, state # output:[batch_size, max_input_length, enc_hidden_size], state:[batch_size, enc_hidden_size]

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    def call(self, query, values):
        # shape: [batch_size, 1, enc_hidden_size]
        hidden_with_time_axis = tf.expand_dims(query, axis=1)
        # shape: [batch_size, input_max_length, 1]
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        # shape: [batch_size, input_max_length, 1]
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)  # shape: [batch_size, enc_hidden_size]
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, dec_vocab_size, embedding_dim, dec_hidden_size, batch_size):
        super(Decoder, self).__init__()
        self.dec_hidden_size = dec_hidden_size
        self.batch_size = batch_size
        self.embedding = tf.keras.layers.Embedding(dec_vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_hidden_size, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        # 定义全连接输出层
        self.fc = tf.keras.layers.Dense(dec_vocab_size)
        # 使用 Attention 机制
        self.attention = BahdanauAttention(self.dec_hidden_size)
    def call(self, x, hidden_state, enc_output):
        context_vector, attention_weights = self.attention(hidden_state, enc_output)
        x = self.embedding(x)  # shape: [batch_size, 1, embedding_dim]
        x = tf.concat([tf.expand_dims(context_vector, axis=1), x], axis=-1) # [batch_size, 1, embedding_dim + enc_units]
        output, state = self.gru(x)  # output/state-->[batch_size, 1, dec_hidden_size]
        output = tf.reshape(output, (-1, output.shape[2])) # output-->[batch_size, dec_hidden_size]
        outputs = self.fc(output) # [batch_size, dec_vocab_size]
        # [batch_size, dec_vocab_size]  [batch_size, dec_vocab_size]  [batch_size, input_max_length, 1]
        return outputs, state, attention_weights

# 实例化 Encoder 模型，Decoder 模型
encoder = Encoder(config["encode_vocab_size"], config["embedding_dim"], config["enc_hidden_size"], config["batch_size"])
decoder = Decoder(config["decode_vocab_size"], config["embedding_dim"], config["dec_hidden_size"], config["batch_size"])
# 定义优化器、损失目标函数
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)

def loss_function(real, pred):
    # real--> [batch_size],  pred-->[batch_size, dec_vocab_size]
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # 有单词的位置为 1， 无单词的位置为 0
    loss_ = loss_object(real, pred)  # 一个实数
    mask = tf.cast(mask, dtype=loss_.dtype)  # shape: [batch_size]
    loss_ *= mask  # loss_的shape-->[batch_size], 预测出单词的位置才计算入 loss_
    return tf.reduce_mean(loss_)

def train_step(input, target, target_language, enc_hidden):
    # input: [batch_size, input_max_length], target: [batch_size, output_max_length]
    loss = 0
    with tf.GradientTape() as tape:
        # enc_output: [batch_size, max_length, enc_hidden_size], enc_hidden: [batch_size, enc_hidden_size]
        enc_output, enc_hidden = encoder(input, enc_hidden)
        dec_hidden = enc_hidden
        # dec_input: [batch_size, 1]
        dec_input = tf.expand_dims([target_language.word_index['start']]*config["batch_size"], 1)
        for t in range(1, target.shape[1]):
            # 使用解码器上一步的输出：dec_hidden 和 enc_output 计算出 context_vector,
            # 然后 context_vector 和 dec_input 作为 decoder 的输入，计算输出 predictions 和 dec_hidden
            # predictions: [batch_size, dec_vocab_size]  dec_hidden: [batch_size, dec_hidden_size]
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(target[:, t], predictions) # target[:,t] --> [batch_size],即[index1, index2, ... ]
            dec_input = tf.expand_dims(target[:,t], 1)  # [batch_size, 1]
        # 计算批处理平均损失值
        batch_loss = (loss / int(target.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

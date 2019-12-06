# _*_ coding: utf-8 _*_
"""
@author: Jibao Wang
@time: 2019/12/5 15:02
"""

import getConfig
import time
import tensorflow as tf
import seq2seqModel
import data_util

config = getConfig.get_config('config.ini')
resource_data_file = config['resource_data_file']


def train():
    # 创建训练、验证 批数据
    sp_input_tensor, en_target_tensor, sp_input_language_tokenizer, en_target_language_tokenizer = data_util.load_dataset(resource_data_file)
    sp_input_tensor_train, sp_input_tensor_val, en_target_tensor_train, en_target_tensor_val = data_util.train_test_split(en_input_tensor, sp_target_tensor, test_size=0.2)
    dataset = tf.data.Dataset.from_tensor_slices((sp_input_tensor_train, en_target_tensor_train)).shuffle(len(sp_input_tensor_train))
    dataset = dataset.batch(config['batch_size'], drop_remainder=True)
    for epoch in range(config['epochs']):
        start = time.time()
        enc_hidden = tf.zeros((config["batch_size"], config["enc_hidden_size"]))
        total_loss = 0
        for (batch, (input_lang, target_lang)) in enumerate(dataset):
            # input_lang--> [batch_size, max_input_length], target_lang-->[batch_size, max_target_length]
            batch_loss = seq2seqModel.train_step(input_lang, target_lang, en_target_language_tokenizer, enc_hidden)
            total_loss += batch_loss
            if batch % 100 == 0:
                print('Epoch %d, Batch %d, Loss %.4f' % (epoch + 1, batch, batch_loss.numpy()))
        # 2 个 epoch， 保存一个 checkpoint
        if (epoch + 1) % 2 == 0:
            seq2seqModel.checkpoint.save(file_prefix=config['model_dir'])
        print("Epoch %d, Loss %.4f" % (epoch + 1, total_loss / (len(sp_input_tensor_train) // config['batch_size'])))
        print("Time taken for one epoch %f sec" % (time.time() - start))


def reload_model():
    # 实例化 Encoder 模型，Decoder 模型
    encoder = seq2seqModel.Encoder(config["encode_vocab_size"], config["embedding_dim"], config["enc_hidden_size"], config["batch_size"])
    decoder = seq2seqModel.Decoder(config["decode_vocab_size"], config["embedding_dim"], config["dec_hidden_size"], config["batch_size"])
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(config['model_dir']))
    return encoder, decoder

# 翻译
def translate(sentence):
    sp_input_tensor, en_target_tensor, sp_input_language_tokenizer, en_target_language_tokenizer = data_util.load_dataset(resource_data_file)
    encoder, decoder = reload_model()

    sentence = data_util.preprocess_sentence(sentence)
    inputs = [sp_input_language_tokenizer.word_index[i] for i in sentence.split(' ')]
    # inputs--> [1, input_max_length]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=sp_input_tensor.shape[1], padding='post')
    inputs = tf.convert_to_tensor(inputs)

    enc_hidden = tf.zeros((1, config['enc_hidden_size']))
    enc_output, enc_state = encoder(inputs, enc_hidden)
    dec_hidden = enc_state
    # dec_input-->[1, 1]
    dec_input = tf.expand_dims([en_target_language_tokenizer.word_index['<start>']], 0)
    result = ""
    for t in range(en_target_tensor.shape[1]):
        # prediction/dec_hidden-->[1, dec_vocab_size]
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
        predictioned_id = tf.argmax(predictions[0]).numpy()
        result += en_target_language_tokenizer.index_word[predictioned_id] + " "
        if en_target_language_tokenizer.index_word[predictioned_id] == '<end>':
            return result, sentence
        # feedback 回去
        dec_input = tf.expand_dims([predictioned_id], 0)
    return result, sentence

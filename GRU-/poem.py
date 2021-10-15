# _*_ coding:utf-8 _*_
# @Time : 2021/10/14 21:59
# @Author : xupeng
# @File : poem.py
# @software : PyCharm
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import LSTM, GRU, Dropout,Dense, Input, Embedding,Bidirectional, Flatten
from tensorflow.keras.optimizers import Adam
import random
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint

# 第一，语料准备。一共四万多首古诗，每行一首诗，标题在预处理的时候已经去掉了。
#
# 第二，文件预处理。首先，机器并不懂每个中文汉字代表的是什么，所以要将文字转换为机器能理解的形式，这里我们采用 One-Hot 的形式，这样诗句中的每个字都能用向量来表示，下面定义函数 preprocess_file() 来处理。
# puncs = [']', '[', '（', '）', '{', '}', '：', '《', '》','，','。',':','?','!']
puncs = [']', '[', '（', '）', '{', '}', '：', '《', '》']

#传入诗词预料文本路径
#返回字转索引号的函数
#返回num2word的映射字典
#返回按照频率排序后的字列表
#返回所有唐诗字符串， 每首唐诗结尾是】
def preprocess_file(Config):

    # 语料文本内容
    files_content = ''
    with open(Config.poetry_file, 'r', encoding='utf-8') as f:
        for line in f:

            # 每行的末尾加上"]"符号代表一首诗结束
            for char in puncs:
                line = line.replace(char, "")

            files_content += line.strip() + "]"

    words = sorted(list(files_content))
    words.remove(']')
    counted_words = {}
    #统计词频
    for word in words:
        if word in counted_words:
            counted_words[word] += 1
        else:
            counted_words[word] = 1

    #除去低频字
    erase = []
    for key in counted_words:
        if counted_words[key] <= 2:
            erase.append(key)

    # print(erase)
    for key in erase:
        del counted_words[key]
    del counted_words["]"]
    #按照字出现的次数进行降序排列
    wordPairs = sorted(counted_words.items(), key=lambda x:-x[1])
    # print(wordPairs)
    words, _ = zip(*wordPairs)


    #从word到id的映射
    word2num = dict((c, i+1) for i, c in enumerate(words))
    # print(word2num)
    num2word = dict((i, c) for i, c in enumerate(words))
    # print(num2word)
    #传入一个字，返回编号，不存在就返回0
    word2numF = lambda x: word2num.get(x, 0)
    return word2numF, num2word,words, files_content


# 模型参数配置。预先定义模型参数和加载语料以及模型保存名称，通过类 Config 实现
class Config(object):
    poetry_file = 'new_poem.txt'
    weight_file = 'poetry_model.h5'
    # 根据前六个字预测第七个字
    max_len = 6
    batch_size = 512
    learning_rate = 0.001

# 构建模型，通过 PoetryModel 类实现，类的代码结构如下：

class PoetryModel(object):
    # 函数定义，通过加载Config配置信息，进行语料预处理和模型加载，
    # 如果模型文件存在则直接加载模型，否则开始训练。
    def __init__(self, config):
        self.model = None
        self.do_train = True
        self.load_model = False
        self.config = config

        #文件预处理
        self.word2numF, self.num2word,self.words, self.files_content = preprocess_file(self.config)

        if os.path.exists(self.config.weight_file):
            self.model = load_model(self.config.weight_file)
            self.model.summary()
        else:
            self.train()
        self.do_train = False
        self.load_model= True

    # 函数主要用Keras来构建网络模型，这里使用LSTM的 GRU来实现，当然直接使用   LSTM 也没问题。
    def build_model(self):
        input_tensor = Input(shape=(self.config.max_len,))
        embedd = Embedding(len(self.num2word) + 1, 300, input_length=self.config.max_len)(input_tensor)
        lstm = Bidirectional(GRU(128,return_sequences=True))(embedd)
        dropout = Dropout(0.6)(lstm)
        flatten = Flatten()(dropout)
        dense = Dense(len(self.words), activation='softmax')(flatten)
        self.model = Model(inputs=input_tensor,outputs=dense)
        optimizer = Adam(lr=self.config.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # sample函数，在训练过程的每个epoch迭代中采样。
    def sample(self, preds, tempeature=1.0):
        '''
         当temperature=1.0时，模型输出正常
         当temperature=0.5时，模型输出比较open
         当temperature=1.5时，模型输出比较保守
         在训练的过程中可以看到temperature不同，结果也不同
         '''
        preds = np.array(preds).astype('float64')
        preds = np.log(preds) / tempeature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    # 训练过程中，每个epoch打印出当前的学习情况。
    def generate_sample_result(self, epoch, logs):
        print("\n==================Epoch {}=====================".format(epoch))
        for diversity in [0.5, 1.0, 1.5] :
            print("------------Diversity {}--------------".format(diversity))
            start_index = random.randint(0, len(self.files_content) - self.config.max_len - 1)
            generated = ''
            sentence = self.files_content[start_index:start_index+self.config.max_len]
            generated += sentence
            for i in range(20):
                x_pred = np.zeros((1, self.config.max_len))
                for t, char in enumerate(sentence[-6:]):
                    x_pred[0,t] = self.word2numF(char)
                preds= self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_char = self.num2word[next_index]
                generated += next_char
                sentence = sentence + next_char
            print(sentence)


    # 函数，用于根据给定的提示，来进行预测。根据给出的文字，生成诗句，如果给的text不到四个字，则随机补全。
    def predict(self, text):
        print("start predict")
        if not self.load_model:
            return
        with open(self.config.poetry_file, 'r', encoding='utf-8') as f:
            file_list = f.readlines()
        random_line = random.choice(file_list)
        # print("random_line",random_line)
        # 如果给的text不到四个字，则随机补全
        if not text or len(text) != 4:
            for _ in range(4 - len(text)):
                random_str_index = random.randrange(0, len(self.words))
                text += self.num2word.get(random_str_index) if self.num2word.get(random_str_index)\
                    not in [',', '。','，'] else self.num2word.get(random_str_index + 1)
        # print("text:",text)
        seed = random_line[-(self.config.max_len):-1]
        # print("seed:",seed)
        res = ''
        seed = 'c' + seed
        # print("seed:", seed)
        for c in text:
            seed = seed[1:] + c
            # print("seed:", seed)
            for j in range(5):
                x_pred = np.zeros((1, self.config.max_len))
                for t, char in enumerate(seed):
                    x_pred[0,t] = self.word2numF(char)
                preds = self.model.predict(x_pred,verbose=0)[0]
                # print("preds",preds)
                next_index = self.sample(preds, 1.0)
                # print("next_index",next_index)
                next_char = self.num2word[next_index]
                # print("next_char",next_char)
                seed = seed[1:] + next_char
            res += seed
            # print("res",res)
        return  res

    # 函数，用于生成数据，提供给模型训练时使用。
    def data_generator(self):
        i = 0
        while 1:
            x = self.files_content[i:i+self.config.max_len]
            y = self.files_content[i + self.config.max_len]
            # puncs = [']', '[', '（', '）', '{', '}', '：', '《', '》', '，', '。', ':', '?', '!']
            puncs = [']', '[', '（', '）', '{', '}', '：', '《', '》']
            if len([i for i in puncs if i in x]) != 0:
                i += 1
                continue
            if len([i for i in puncs if i in y]) != 0:
                i += 1
                continue
            y_vec = np.zeros(shape=(1, len(self.words)), dtype=np.bool)
            y_vec[0, self.word2numF(y)]  = 1.0
            x_vec = np.zeros(shape=(1, self.config.max_len), dtype=np.int32)
            for t, char in enumerate(x):
                x_vec[0,t] = self.word2numF(char)
            yield x_vec, y_vec
            i += 1


    def train(self):
        number_of_epoch = 10
        if not self.model:
            self.build_model()
        self.model.summary()
        self.model.fit_generator(
            generator=self.data_generator(),
            verbose=True,
            steps_per_epoch=self.config.batch_size,
            epochs=number_of_epoch,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(self.config.weight_file, save_weights_only=False),
                LambdaCallback(on_epoch_end=self.generate_sample_result)
            ]
        )




model = PoetryModel(Config)
# print(model.files_content)
# print(len(model.files_content))
text = input("text:")
sentence = model.predict(text)
print(sentence)



















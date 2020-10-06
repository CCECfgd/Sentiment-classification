#加入相关依赖
import os
import keras
import jieba
import logging
import re
import gensim
import pandas as pd
import numpy as np
from keras.layers import Bidirectional,LSTM,Dense,Embedding,Dropout,Activation,advanced_activations,Conv1D,MaxPooling1D,Flatten,BatchNormalization
from sklearn.model_selection import train_test_split

def word_cut(text):#结果为list
    """
    数据预处理：对传入的str进行切词
    :param text: 目标str
    :return: 切词后的list
    """
    jieba.setLogLevel(logging.INFO)  # 显示结巴分词日志
    regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')  # 正则表达式，返回的是一个匹配对象，https://www.runoob.com/regexp/regexp-syntax.html
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]

def comb_dataset(train,dev):
    """
    数据预处理：传入读取好的数据，将元组（x,y）存进一个列表中
    :param train: 训练集的原始表格
    :param dev: 验证集的原始表格
    :return: x,y
    """
    train_x = train['text']                                 #按行抽取
    train_y = train['label']
    dev_x = dev['text']
    dev_y = dev['label']

    X = list(train_x) + list(dev_x)                         #拼接
    Y = list(train_y) + list(dev_y)
    for i in range(len(X)):
        X[i] = word_cut(X[i])                               #执行切词
    return X,Y


def cut_dataset_word(x_all):
    """
    数据预处理：对传入的二维列表中的句子进行切词
    :param x_all: [[句子1],[句子2]]
    :return: [[分词后的句子1],[分词后的句子2]]
    """
    x_all_cut=[]#分词后的语料库
    for i in range(len(x_all)):
        x_all_cut.append(word_cut(x_all[i]))#切词后将元组添加至x_all_cut
    print(len(x_all_cut))
    return x_all_cut


def load_word_model(path):
    """
    读取词向量模型
    :return:词向量
    """
    word_model = gensim.models.Word2Vec.load(path)
    return word_model


def train_word_model(name):
    """
    根据语料库训练词向量并保存至本地
    :param name: 模型名
    :return: 词向量模型
    """
    x_all = list(pd.read_csv("data/ch_auto.csv")["text"])#读取原始语料库
    x_all_cut = cut_dataset_word(x_all)#对语料库进行切词
    word_model = gensim.models.Word2Vec(x_all_cut, min_count=1)#训练词向量模型
    path = "word_model/%s.model" % name
    word_model.save(path)#保存词向量模型
    return word_model


def generate_id2wec(model):
    """

    :param model:
    :return:
    """
    gensim_dict = gensim.corpora.dictionary.Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
    w2id = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
    w2vec = {word: model[word] for word in w2id.keys()}  # 词语的词向量
    n_vocabs = len(w2id) + 1
    embedding_weights = np.zeros((n_vocabs, 100))
    for w, index in w2id.items():  # 从索引为1的词语开始，用词向量填充矩阵
        embedding_weights[index, :] = w2vec[w]
    return w2id,embedding_weights


def text_to_array(w2index, senlist):  #
    """
    文本转为索引数字模式，将
    :param w2index:
    :param senlist:
    :return:
    """
    sentences_array = []
    for sen in senlist:
        new_sen = [ w2index.get(word,0) for word in sen]   # 单词转索引数字
        sentences_array.append(new_sen)
    return np.array(sentences_array)


def prepare_data(w2id,sentences,labels,max_len=100):
    """

    :param w2id:
    :param sentences:
    :param labels:
    :param max_len:
    :return:
    """
    X_train, X_val, y_train, y_val = train_test_split(sentences,labels, test_size=0.2)
    X_train = text_to_array(w2id, X_train)
    X_val = text_to_array(w2id, X_val)
    X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
    X_val = keras.preprocessing.sequence.pad_sequences(X_val, maxlen=max_len)
    return np.array(X_train), keras.utils.np_utils.to_categorical(y_train) ,np.array(X_val), keras.utils.np_utils.to_categorical(y_val)

def find_model(path,file_type):
    """
    查找指定文件夹下指定类型的文件
    :param path: 文件路径
    :param file_type: 文件类型
    :return: 返回包含文件名的list
    """
    model_dir = os.listdir(path)  # 获取模型文件夹下的文件
    model_list = []
    for i in model_dir:  # 查找有无模型文件
        if file_type in i:
            model_list.append(i)
        else:
            pass
    return model_list


class Sentiment:
    """
    神经网路的模型、训练及预测参数
    """
    def __init__(self, w2id, embedding_weights, Embedding_dim, maxlen, labels_category):
        self.Embedding_dim = Embedding_dim              #
        self.embedding_weights = embedding_weights      #
        self.vocab = w2id                               #
        self.labels_category = labels_category          # 类别数量
        self.maxlen = maxlen                            # 最大长度
        self.model = self.build_model_bilstm()                 # 模型通过build_model方法来构建

    def build_model_bilstm(self):
        """
        BILSTM
        """
        model = keras.Sequential()
        model.add(Embedding(output_dim=self.Embedding_dim,
                            input_dim=len(self.vocab) + 1,
                            weights=[self.embedding_weights],
                            input_length=self.maxlen))
        model.add(Bidirectional(LSTM(50), merge_mode='concat'))
        model.add(Dropout(0.5))
        #model.add(BatchNormalization())
        model.add(Dense(self.labels_category))
        model.add(Activation("sigmoid"))  # relu
        model.compile(loss='binary_crossentropy',#binary_crossentropy，categorical_crossentropy
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        return model

    def build_model_textcnn(self):
        """
        TEXTCNN
        :return:
        """
        model = keras.Sequential()
        model.add(Embedding(output_dim=self.Embedding_dim,
                            input_dim=len(self.vocab) + 1,
                            weights=[self.embedding_weights],
                            input_length=self.maxlen))
        model.add(Conv1D(256, 5, padding='same'))
        model.add(MaxPooling1D(3, 3, padding='same'))
        model.add(Conv1D(128, 5, padding='same'))
        model.add(MaxPooling1D(3, 3, padding='same'))
        model.add(Conv1D(64, 3, padding='same'))
        model.add(Flatten())
        model.add(Dropout(0.1))
        model.add(BatchNormalization())  # (批)规范化层
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(2, activation='relu'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def train(self, X_train, y_train, X_test, y_test, n_epoch,path):
        """
        通过训练集和训练，同时经过验证集验证，训练完成导出参数
        :param X_train: 训练集的样本
        :param y_train: 训练集的真实值
        :param X_test:
        :param y_test:
        :param n_epoch: 轮训次数
        :param path: 参数保存的路径及名字
        """
        self.model.fit(X_train, y_train, batch_size=200, epochs=7,
                       validation_data=(X_test, y_test))
        self.model.save(path)

    def predict(self, model_path, new_sen):
        """
        根据model进行预测输入str的结果
        :param model_path:权重保存的路径
        :param new_sen:待检测字符串
        :return:0/1
        """
        model = self.model
        model.load_weights(model_path)
        new_sen_list = jieba.lcut(new_sen)
        sen2id = [self.vocab.get(word, 0) for word in new_sen_list]
        sen_input = keras.preprocessing.sequence.pad_sequences([sen2id], maxlen=self.maxlen)
        res = model.predict(sen_input)[0]
        return np.argmax(res)


if __name__ =="__main__":
    #加载词向量
    index = input("是否使用已有词向量模型（y/n）：")
    if index == "y" or index == "Y":
        wmodel_list = find_model("word_model/",".model")

        if wmodel_list != []:
            print(wmodel_list)
            commd = int(input("请按照列表索引选择词向量模型："))
            wmodel_path = "word_model/" + wmodel_list[commd]  # 模型位置
            word_model = load_word_model(wmodel_path)  # 获取词向量
            w2id, embedding_weights = generate_id2wec(word_model)
        else:
            print("未找到词向量模型")

    elif index == "n" or index == "N":
        wmodel_name = input("保存此次训练的词向量名为：")
        word_model = train_word_model(wmodel_name)
        w2id, embedding_weights = generate_id2wec(word_model)
    else:
        print("输入错误，程序结束")

    test_str = "回头率还可以，无框门，上档次"#待检测文本

    #读取或训练网络模型
    index = ""
    index = input("是否使用已有网络模型进行预测（y/n）：")
    if index == "y" or index == "Y":
        model_list = find_model("model/",".h5")
        if model_list !=[]:
            print(model_list)
            commd = int(input("请按照列表索引选择网络模型："))
            model_path = "model/" + model_list[commd]  # 模型位置
            senti = Sentiment(w2id, embedding_weights, 100, 200, 2)
            pre_result = senti.predict(model_path,test_str)
            print(pre_result)

    elif index == "n" or index == "N":
        train = pd.read_csv('data/test.tsv', sep='\t', )  # 读取数据表
        dev = pd.read_csv('data/dev.tsv', sep='\t', )
        sentences, labels = comb_dataset(train, dev)  # 将读取的原始数据使用comb_dataset()进行切词并返回句子和label的列表
        x_train, y_train, x_val, y_val = prepare_data(w2id, sentences, labels, 200)  # 使用prepare_data()切分数据集并返回
        senti = Sentiment(w2id, embedding_weights, 100, 200, 2)
        model_name = input("保存此次训练的模型名为：")
        model_path = "model/%s.h5" % model_name
        senti.train(x_train, y_train, x_val, y_val, 5,model_path)
        pre_result = senti.predict(model_path,test_str)
        print(pre_result)
    else:
        print("输入错误")
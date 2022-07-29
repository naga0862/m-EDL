# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 08:56:36 2022

@author: a_nagahama_r
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from typing import Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns

import random

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import utils

import pandas
summary_folder = r'Z:/ENN_acc/1_LnuTnu/'

#For learning
class EDLLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        K: int,
        annealing: float,
        name: str = "custom_edl_loss",
        reduction="auto",
    ):
        super().__init__(name=name, reduction=reduction)
        self.K = K
        self.annealing = annealing
        pass

    def call(self, y_true, evidence):
        alpha = evidence + 1
        return self.loss_eq5(y_true=y_true, alpha=alpha)

    def KL(self, alpha):
        beta = tf.constant(np.ones((1, self.K)), dtype=tf.float32)
        S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)
        KL = (
            tf.reduce_sum(
                (alpha - beta)
                * (tf.math.digamma(alpha) - tf.math.digamma(S_alpha)),
                axis=1,
                keepdims=True,
            )
            + tf.math.lgamma(S_alpha)
            - tf.reduce_sum(tf.math.lgamma(alpha), axis=1, keepdims=True)
            #+ tf.reduce_sum(tf.math.lgamma(beta), axis=1, keepdims=True)#see notebook20220214
            - tf.math.lgamma(tf.reduce_sum(beta, axis=1, keepdims=True))
        )
        return KL

    def loss_eq5(self, y_true: np.ndarray, alpha: tf.Tensor):
        S = tf.reduce_sum(alpha, axis=1, keepdims=True)
        L_err = tf.reduce_sum(
            (y_true - (alpha / S)) ** 2, axis=1, keepdims=True
        )
        L_var = tf.reduce_sum(
            alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdims=True
        ) 
        KL_reg = self.KL((alpha - 1) * (1 - y_true) + 1)
        return L_err + L_var + self.annealing * KL_reg

    def get_config(self):
        config = super().get_config()
        config.update({"K": self.K, "annealing": self.annealing})
        return config


if __name__ == '__main__':
    #挙動テストの部
    #test_label = np.array([1,3,2,4])
    #onehot_testlabel = tf.one_hot(indices=test_label,depth=5)
    #onehot_testlabel_np = onehot_testlabel.numpy()
    #y_test_np = y_test.numpy()
    
    #arr = np.arange(25).reshape(5, 5)
    #dataset = tf.data.Dataset.from_tensor_slices(arr)#.shuffle(5)
    #dataset = tf.data.Dataset.from_tensor_slices(arr).shuffle(5)
    #dataset = tf.data.Dataset.from_tensor_slices(arr).shuffle(5).batch(2)
    #for item in dataset:
    #    print(item)
    
    #以下本文
    np.random.seed(123)
    tf.random.set_seed(123)
    
    
    for i_trial in range (0,100):#take mean of result
        print('1_LnuTnu {}th trial!!!!!!!!!!!!!'.format(i_trial+1))
        
        #"preparation of data"
        mnist = datasets.mnist
        (x_learn, y_learn), (x_test, y_test) = mnist.load_data()#y is output label data
        x_learn, y_learn = utils.shuffle(x_learn, y_learn) 
        x_test, y_test = utils.shuffle(x_test, y_test) 
        
        # MNISTデータの前処理関数（最大値で正規化&[0, 1]の2値に変換)．NOTE:one-hotに変換していることに注意
        #変数images, labelsはそれぞれnp.ndarray 出力は、np.ndarray, tf.Tensorのtuple型
        #imagesは28*28=784列のベクトルがデータ数分だけ並んだ2D arrayに変換される
        #出力tf.Tensorはone-hot横ベクトルがデータ数分だけ並んだtf.Tensor
        def preprocess_images(images: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, tf.Tensor]:
            images = images.reshape((images.shape[0], 28*28)) / 255.0
            #return np.where(images > 0.5, 1.0, 0.0).astype("float32"), tf.one_hot(indices=label, depth=10)
            return images.astype("float32"), tf.one_hot(indices=label, depth=10)
        
        #test, learn入出力データをpreprocess_imagesで変形する
        x_learn, y_learn = preprocess_images(x_learn, y_learn)
        x_test, y_test = preprocess_images(x_test, y_test)
        
        # learn_dataset = (
        # tf.data.Dataset.from_tensor_slices((x_learn, y_learn))
        # .shuffle(60000)
        # .batch(64))
        
        # #testデータもshuffleするの？？batch化するの？
        # test_dataset = (
        #     tf.data.Dataset.from_tensor_slices((x_test, y_test))
        #     .shuffle(10000)
        #     .batch(64))   
        # #for item in test_dataset:
        # #    print(item)
        
        # #一旦shuffleして規定のデータ数に揃える learn 60000 test 10000
        # no operation
        
        #"model development"
        model = Sequential()
        model.add(Dense(32,activation='relu',input_shape=(28*28,)))
        #model.add(Dense(32,activation='relu'))
        #input次元は省略可能　出力は32次元の層
        # model.add(tf.keras.layers.Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        # model.add(tf.keras.layers.Dropout(0.3))
        model.add(Dense(10, activation='relu'))
        # モデルのサマリ出力
        #print(model.summary())
        
        
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=EDLLoss(K=10, annealing=0.1), metrics=["accuracy", "mse"])
        #model.fit(x_learn, y_learn, batch_size=64, shuffle=True, validation_split=0.1, epochs=10)#epochs=10
        
        es = EarlyStopping(monitor='val_loss',patience=3,min_delta=0.001,verbose=1)
        model.fit(x_learn, y_learn, batch_size=64, shuffle=True, validation_split=0.1,callbacks=[es], epochs=100)#epochs=
        
        
        
        #"model evaluation"
        def calc_enn_output_from_evidence(evidence: tf.Tensor) -> Tuple[np.ndarray, np.ndarray, List[float], np.ndarray]:
            _, n_class = evidence.shape
            K = n_class
            alpha = evidence + 1
            S = tf.reduce_sum(alpha, axis=1, keepdims=True)
            bk = evidence[:,0:K] / S.numpy() #
            Phat_k = alpha / S
            #print(y_pred.numpy())
            Phat_k = Phat_k.numpy()
            y_pred = (
                np.argmax(Phat_k, axis=1)
            )
            
            #See notebook 20220214
            #if values in Phat_k are all the same
            #then select label from 0-9 randomly
            for i in range(len(Phat_k)):
                if len(np.unique(Phat_k[i])) == 1:
                    y_pred[i] = random.randint(0,9)
            
            uncertainty = n_class / tf.reduce_sum(alpha, axis=1, keepdims=True)
            #print(uncertainty)
            #uncertainty = tf.reshape(uncertainty, -1)#original code. change column vector to 
            uncertainty = tf.reshape(uncertainty, [-1])
            #print('!!!!!!')
            #print(uncertainty)
            uncertainty = [u.numpy() for u in uncertainty]
            return evidence, alpha, bk, uncertainty, y_pred, Phat_k
        
        #Model Evaluation!!!!!!!!!!!!!!!!!!!!!!!!!!!
        evidence = model.predict(x_test)
        evi, alp, belief, unc, y_pred, p_hat_k = calc_enn_output_from_evidence(evidence)
        p_hat_k_sum = p_hat_k.sum(axis=1)
        
        #add see notebook 20220301
        def remove_data_with_large_unc(u,uth,predlabel,correctlabel):
            return [u[np.where(u <= uth)], predlabel[np.where(u <= uth)], correctlabel[np.where(u <= uth)]]     
            
        y_test_label = np.argmax(y_test,axis=1)#labelに戻す
        label_list =[0,1,2,3,4,5,6,7,8,9]
        
        #定めたunc thresholdよりもuncが大きいデータを消す
        unc_ths = [i * 0.1 for i in range(0,11)]
        accs = []
        recalls = []
        precs = []
        for unc_th in unc_ths:
            unc_pick, y_pred_pick, y_test_label_pick = remove_data_with_large_unc(np.array(unc),unc_th,y_pred,y_test_label)
        
            conf_mat = confusion_matrix(y_test_label_pick,y_pred_pick,labels=label_list)
            acc = accuracy_score(y_test_label_pick,y_pred_pick)
            recall = recall_score(y_test_label_pick,y_pred_pick,average='micro')
            prec = precision_score(y_test_label_pick,y_pred_pick,average='micro')
            accs.append(acc)
            recalls.append(recall)
            precs.append(prec)
        
        accs_df = pandas.DataFrame(accs,index = unc_ths)
        accs_df.to_csv(summary_folder+r'{0}_accs_k_{1}.csv'.format('1_LnuTnu',i_trial))#index is unc_th
        recalls_df = pandas.DataFrame(recalls,index = unc_ths)
        recalls_df.to_csv(summary_folder+r'{0}_recalls_k_{1}.csv'.format('1_LnuTnu',i_trial))#index is unc_th
        precs_df = pandas.DataFrame(precs,index = unc_ths)
        precs_df.to_csv(summary_folder+r'{0}_precs_k_{1}.csv'.format('1_LnuTnu',i_trial))#index is unc_th
    
    

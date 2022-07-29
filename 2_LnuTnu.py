# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 08:56:36 2022

@author: a_nagahama_r
"""
from scipy import stats

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from typing import Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import cifar10

import tensorflow_datasets as tfds

from emnist import list_datasets
from emnist import extract_training_samples
from emnist import extract_test_samples

from sklearn import utils

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import pandas
summary_folder = r'Z:/ENN_acc/2_LnuTnu/'

# import requests
# import urllib3
# from urllib3.exceptions import InsecureRequestWarning
# urllib3.disable_warnings(InsecureRequestWarning)

# r = requests.get('https://www.itl.nist.gov',verify=False)

#For learning See notebook 20220202
class EDLLoss_p(tf.keras.losses.Loss):
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

    def call(self, y_true_p, evidence):
        tf.config.experimental_run_functions_eagerly(True)#modified 20220202 to add a column for "u" to evidence
        #print(evidence.numpy())
        alpha = evidence + 1
        alpha_e = alpha[:,0:(self.K)]#evidenceはすでにK+1次元なのでαuのために1列削除
        alpha_u = tf.constant([[self.K + 1 ] for i in range (len(alpha.numpy()))], dtype=tf.float32)#alpha uを用意
        alpha_p = tf.concat([alpha_e,alpha_u], 1)#削除した1列に追加
        #print(alpha_p.numpy())#ok
        #tf.config.experimental_run_functions_eagerly(False)
        return self.loss_eq5(y_true_p=y_true_p, alpha_p=alpha_p)

    def KL(self, alpha_t):
        beta = tf.constant(np.ones((1, self.K + 1)), dtype=tf.float32)###########original self.K
        S_alpha = tf.reduce_sum(alpha_t, axis=1, keepdims=True)#sum of alpha_tilde
        KL = (
            tf.reduce_sum(
                (alpha_t - beta)
                * (tf.math.digamma(alpha_t) - tf.math.digamma(S_alpha)),
                axis=1,
                keepdims=True,
            )
            + tf.math.lgamma(S_alpha)
            - tf.reduce_sum(tf.math.lgamma(alpha_t), axis=1, keepdims=True)
            #+ tf.reduce_sum(tf.math.lgamma(beta), axis=1, keepdims=True)#see notebook20220214
            - tf.math.lgamma(tf.reduce_sum(beta, axis=1, keepdims=True))
        )
        return KL

    def loss_eq5(self, y_true_p: np.ndarray, alpha_p: tf.Tensor):
        S_p = tf.reduce_sum(alpha_p, axis=1, keepdims=True)
        L_err = tf.reduce_sum(
            (y_true_p - (alpha_p / S_p)) ** 2, axis=1, keepdims=True
        )
        L_var = tf.reduce_sum(
            alpha_p * (S_p - alpha_p) / (S_p * S_p * (S_p + 1)), axis=1, keepdims=True
        ) 
        KL_reg = self.KL((alpha_p - 1) * (1 - y_true_p) + 1)
        #print(L_err + L_var + self.annealing * KL_reg)#ok
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
        print('2_LnuTnu {}th trial!!!!!!!!!!!!!'.format(i_trial+1))
        #"preparation of data"
        #mnist data
        mnist = datasets.mnist
        (x_learn_m, y_learn_m), (x_test_m, y_test_m) = mnist.load_data()#y is output label data    
        x_learn_m, y_learn_m = utils.shuffle(x_learn_m, y_learn_m) 
        x_test_m, y_test_m = utils.shuffle(x_test_m, y_test_m) 
        
        #emnist
        #emnist_train = tfds.load(name='emnist/balanced', split='train')
        #(x_learn_c,y_learn_c),(x_test_c,y_test_c) = cifar10.load_data()
        #x_learn_c = tf.image.rgb_to_grayscale(x_learn_c)
        # #x_test_c = (tf.image.rgb_to_grayscale(x_test_c)).numpy()
        # x_test_em, label_test_em = extract_test_samples('letters')
        # x_test_em, label_test_em = utils.shuffle(x_test_em, label_test_em)#randomize order of emnist data
        # y_test_em = np.array([10 for i in range(len(label_test_em))])#y_em are all 10 = U
            
        # #fashion mnist
        # (x_learn_f,y_learn_f),(x_test_f,y_test_f) = fashion_mnist.load_data()
        
        
        # MNISTデータの前処理関数（最大値で正規化&[0, 1]の2値に変換)．NOTE:one-hotに変換していることに注意
        #変数images, labelsはそれぞれnp.ndarray 出力は、np.ndarray, tf.Tensorのtuple型
        #imagesは28*28=784列のベクトルがデータ数分だけ並んだ2D arrayに変換される
        #出力tf.Tensorはone-hot横ベクトルがデータ数分だけ並んだtf.Tensor
        def preprocess_images(images: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, tf.Tensor]:
            images = images.reshape((images.shape[0], 28*28)) / 255.0
            #return np.where(images > 0.5, 1.0, 0.0).astype("float32"), tf.one_hot(indices=label, depth=10)
            #return np.where(images > 0.5, 1.0, 0.0).astype("float32"), tf.one_hot(indices=label, depth=11)#See notebook20220202
            return images.astype("float32"), tf.one_hot(indices=label, depth=11)#See notebook20220202
        
        #test, learn入出力データをpreprocess_imagesで変形する
        x_learn_m, y_learn_m = preprocess_images(x_learn_m, y_learn_m)
        x_test_m, y_test_m = preprocess_images(x_test_m, y_test_m)
        # x_test_em, y_test_em = preprocess_images(x_test_em, y_test_em)
        #y_test_m_array = y_test_m.numpy()#check
        #y_learn_array = y_learn.numpy()#check
        #y_test_em_array = y_test_em.numpy()#check
        
        #combine data: LnuTwu = Learning: mnist, Test: mnist 
        # #yu = 1 when U, 11D
        # mix_rate = 0.5
        # x_test_em_pick = x_test_em[:int(len(x_test_m)*mix_rate)]
        # y_test_em_pick = y_test_em[:int(len(x_test_m)*mix_rate)]
        # x_test_mix = np.concatenate([x_test_m,x_test_em_pick])
        # y_test_mix = np.concatenate([y_test_m,y_test_em_pick])
    
        # #shuffle and batch data
        # #xoriginal = x_learn_m.copy()#check
        # learn_dataset = (
        # tf.data.Dataset.from_tensor_slices((x_learn_m, y_learn_m))
        # .shuffle(60000)
        # .batch(64))
        
        # #Q1 testデータもshuffleするの？？batch化するの？Q1
        # test_dataset = (
        #     tf.data.Dataset.from_tensor_slices((x_test_mix, y_test_mix))
        #     .shuffle(10000)
        #     .batch(64))   
        # #for item in test_dataset:
        # #    print(item)
        
        #"model development"
        model = Sequential()
        model.add(Dense(32,activation='relu',input_shape=(28*28,)))
        # model.add(tf.keras.layers.Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        # model.add(tf.keras.layers.Dropout(0.3))
        model.add(Dense(11, activation='relu'))#original Q2
        #model.add(Dense(11, activation='relu'))#
        #model.add(Dense(10, activation='softmax'))
        # モデルのサマリ出力
        print(model.summary())
        
        
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=EDLLoss_p(K=10, annealing=0.1), metrics=["accuracy", "mse"])
        #model.fit(x_learn, y_learn, batch_size=64, shuffle=True, validation_split=0.1, epochs=10)#epochs=10
        
        es = EarlyStopping(monitor='val_loss',patience=3,min_delta=0.001,verbose=1)
        model.fit(x_learn_m, y_learn_m, batch_size=64, shuffle=True, validation_split=0.1,callbacks=[es], epochs=100)#epochs=
        
        #"model evaluation"
        def calc_enn_output_from_evidence(evidence: tf.Tensor) -> Tuple[np.ndarray, np.ndarray, List[float], np.ndarray]:
            _, n_class = evidence.shape
            K = n_class - 1#u以外のクラス数
            #print(evidence)
            alpha_k = evidence[:,0:K] + 1
            #print(alpha_k)
            S = tf.reduce_sum(alpha_k, axis=1, keepdims=True)
            
            bk = evidence[:,0:K] / S.numpy() #
            
            alpha_u = K + 1
            
            #bu = evidence[:,(K-1):K] / S.numpy() 
            bu = K / S#uncertainty see notebook20220126
            #print(bu)
            Phat_k = alpha_k / (S + alpha_u)#!!!!!!!!!!!!!!!!20220126変更点 see notebook 20220126
            
            #print(y_pred.numpy())
            Phat_k = Phat_k.numpy()#20220125変更点
           # _, n_class = Phat_k.shape
            #print(uncertainty)
            #bu = tf.reshape(bu, -1)#original code. change column vector to 
            bu = tf.reshape(bu, [-1])#20220125変更点
            bu = [u.numpy() for u in bu]
            Phat_u = alpha_u / (S + alpha_u)#!!!!!!!!!!!!!!!!20220126追加 see notebook 20220126
            #Phat_u = Phat_u.numpy()#[10000,1]array
            Phat_u = tf.reshape(Phat_u, [-1])#20220125変更点
            Phat_u = [[pu.numpy() for pu in Phat_u]]
            Phat_kp =  np.hstack((Phat_k,np.array(Phat_u).T))#20220127追加 see notebook 20220127
            
            Pbar_k = (alpha_k / S).numpy()#see notebook20220127
            
            # y_pred = (
            #     np.argmax(Phat_k, axis=1)
            # )#original code
            y_pred_Phat = (
                np.argmax(Phat_kp, axis=1)
            )#Phatを使った予測ラベル
            y_pred_Pbar_k = (
                np.argmax(Pbar_k, axis=1)
            )#Pbar kを使った予測ラベル
            
            return evidence, alpha_k, alpha_u, bk, bu, y_pred_Phat, y_pred_Pbar_k, Phat_k, Phat_u, Phat_kp, Pbar_k, S.numpy(), K
        
        
        #Model Evaluation!!!!!!!!!!!!!!!!!!!!!!!!!!!
        evidence = model.predict(x_test_m)#mnist only ekに相当
        evidence_sum = evidence.sum(axis=1)#ekは足してもSにならない　S-Kになるはず=OK
        
        #evidenceからalpha_k alpha_u, belief(bk), uncertainty(bk), pk,puの期待値,S, Kを計算する y_pred_phatはphatを使った予測ラベル、p_pred_pbarはpbarを使ったラベル
        evi, alp_k, alp_u, belief_k, belief_u, y_pred_phat, y_pred_pbar, p_hat_k, p_hat_u, p_hat_kp, p_bar_k, S, large_k = calc_enn_output_from_evidence(evidence) #see notebook 20220126
        
        bk_sum = belief_k.sum(axis=1) #
        sumOf_bk_and_unc = bk_sum + belief_u#1になるはず=ok
        
        p_hat_k_sum = p_hat_k.sum(axis=1)
        p_hat_sum = p_hat_k_sum + p_hat_u#1になるはず=ok
        
        p_hat_kp_sum = p_hat_kp.sum(axis=1)#1になるはずok
        p_bar_k_sum = p_bar_k.sum(axis=1)#1になるはずok
        
        #add see notebook 20220301
        def remove_data_with_large_unc(u,uth,predlabel,correctlabel):
            return [u[np.where(u <= uth)], predlabel[np.where(u <= uth)], correctlabel[np.where(u <= uth)]]     
            
        y_test_label_m = np.argmax(y_test_m,axis=1)#labelに戻す
        label_list =[0,1,2,3,4,5,6,7,8,9]
        
        #定めたunc thresholdよりもuncが大きいデータを消す
        unc_ths = [i * 0.1 for i in range(0,11)]
        acc_pbars = []
        acc_phats = []
        recall_pbars = []
        recall_phats = []
        prec_pbars = []
        prec_phats = []
        for unc_th in unc_ths:
            unc_pick, y_pred_pick, y_test_label_pick = remove_data_with_large_unc(np.array(belief_u),unc_th,y_pred_pbar,y_test_label_m)#pbarを使った予測ラベルの正解率 (i) notebook20220301
        
            conf_mat = confusion_matrix(y_test_label_pick,y_pred_pick,labels=label_list)
            acc_pbar = accuracy_score(y_test_label_pick,y_pred_pick)
            acc_pbars.append(acc_pbar)
            recall_pbar = recall_score(y_test_label_pick,y_pred_pick,average='micro')
            recall_pbars.append(recall_pbar)
            prec_pbar = precision_score(y_test_label_pick,y_pred_pick,average='micro')
            prec_pbars.append(prec_pbar)
            
            acc_phats.append(accuracy_score(y_test_label_m,y_pred_phat))#constant!!!!!!! for comparison i' see notebook 20220301
            recall_phats.append(recall_score(y_test_label_m,y_pred_phat,average='micro'))#constant!!!!!!! for comparison i' see notebook 20220301
            prec_phats.append(precision_score(y_test_label_m,y_pred_phat,average='micro'))#constant!!!!!!! for comparison i' see notebook 20220301
      
        acc_pbars_df = pandas.DataFrame(acc_pbars,index = unc_ths)
        acc_pbars_df.to_csv(summary_folder+r'{0}_acc_pbars_k_{1}.csv'.format('2_LnuTnu',i_trial))#index is unc_th
        
        acc_phats_df = pandas.DataFrame(acc_phats,index = unc_ths)
        acc_phats_df.to_csv(summary_folder+r'{0}_acc_phats_k_{1}.csv'.format('2_LnuTnu',i_trial))#index is unc_th
        
        recall_pbars_df = pandas.DataFrame(recall_pbars,index = unc_ths)
        recall_pbars_df.to_csv(summary_folder+r'{0}_recall_pbars_k_{1}.csv'.format('2_LnuTnu',i_trial))#index is unc_th
        
        recall_phats_df = pandas.DataFrame(recall_phats,index = unc_ths)
        recall_phats_df.to_csv(summary_folder+r'{0}_recall_phats_k_{1}.csv'.format('2_LnuTnu',i_trial))#index is unc_th
        
        prec_pbars_df = pandas.DataFrame(prec_pbars,index = unc_ths)
        prec_pbars_df.to_csv(summary_folder+r'{0}_prec_pbars_k_{1}.csv'.format('2_LnuTnu',i_trial))#index is unc_th
        
        prec_phats_df = pandas.DataFrame(prec_phats,index = unc_ths)
        prec_phats_df.to_csv(summary_folder+r'{0}_prec_phats_k_{1}.csv'.format('2_LnuTnu',i_trial))#index is unc_th
    
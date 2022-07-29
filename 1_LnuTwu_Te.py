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

from emnist import list_datasets
from emnist import extract_training_samples
from emnist import extract_test_samples
from tensorflow.keras.datasets import fashion_mnist

from sklearn import utils

import random

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import pandas


summary_folder = r'Z:/ENN_acc/1_LnuTwu_ltr/'

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


# MNISTデータの前処理関数（最大値で正規化&[0, 1]の2値に変換)．NOTE:one-hotに変換していることに注意
#変数images, labelsはそれぞれnp.ndarray 出力は、np.ndarray, tf.Tensorのtuple型
#imagesは28*28=784列のベクトルがデータ数分だけ並んだ2D arrayに変換される
#出力tf.Tensorはone-hot横ベクトルがデータ数分だけ並んだtf.Tensor
def preprocess_images(images: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, tf.Tensor]:
    images = images.reshape((images.shape[0], 28*28)) / 255.0
    #return np.where(images > 0.5, 1.0, 0.0).astype("float32"), tf.one_hot(indices=label, depth=10)
    return images.astype("float32"), tf.one_hot(indices=label, depth=10)

def remove_data_with_large_unc(u,uth,predlabel,correctlabel):
    class_k_data = [u[np.where(u <= uth)], predlabel[np.where(u <= uth)], correctlabel[np.where(u <= uth)]]  
    #pred label changed 10 where unc larger than uth
    class_U_data = [u[np.where(u > uth)], [10 for i in range(len(predlabel[np.where(u > uth)]))], correctlabel[np.where(u > uth)]]
    return [class_k_data, class_U_data]     

#show images
def show_true_image(x: tf.Tensor, y: np.ndarray, bK: np.ndarray, bU: np.ndarray, pHatK: np.ndarray):
    width, height = 4, 4
    select_label = 5#4 label to display
    #print(len(x))
    trueFalse_match = np.array(np.argmax(y, axis=1) == select_label)
    match_index = np.where(trueFalse_match == True)[0]
    #print(match_index)

    x = x[match_index]#画像の中からvector y のselect labelが一番大きいものだけ残す
    #print(x)
    #print(x.shape)
    pHatK = pHatK[match_index]
    #print(trueFalse_match)
    bK = bK[match_index]
    bU = np.array(bU)
    bU = bU[match_index]
    y = y[trueFalse_match]#画像の中からvector y のselect labelが一番大きいものだけ残す
    print(len(bK))
    print(len(bU))
    print(len(pHatK))
    print(len(y))
    #print(len(x))
    #print(x.shape)
    x = x.reshape(-1, 28, 28, 1)#change x vector to image (28x28)
    #print(x.shape)
    #fig = plt.figure(figsize=(width, height*1.2))
    fig = plt.figure(tight_layout=True)
    for i in range(width*height):
        ax = fig.add_subplot(4, 4, i+1)
        ax.imshow(x[i, :, :, 0], cmap="gray")
        #ax.set_title(f"pHatK:{belief_u[i]:.2f}")
        #ax.set_title(f"y:{y[i,select_label]:.2f}")
        #ax.set_title(f"bk:{bK[i,select_label]:.1f},bu:{bU[i]:.1f}")
        ax.set_title(f"pHatK:{pHatK[i,select_label]:.2f}")
        plt.axis("off")
    plt.show()
      

def make_histgram(uncertainty: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
    plt.style.use("default")
    sns.set()
    sns.set_style("whitegrid")
    sns.set_palette("Set1")
    true_array = uncertainty[y_true == y_pred]
    false_array = uncertainty[y_true != y_pred]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist([true_array, false_array], bins=20, label=["True", "False"], color=["blue", "red"], stacked=True)
    ax.set_xlabel("uncertainty")
    ax.set_ylabel("sample num")
    plt.legend()
    plt.show()

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
        print('1_LnuTwu_ltr {}th trial!!!!!!!!!!!!!'.format(i_trial+1))
        
        #"preparation of data"
        #mnist data
        mnist = datasets.mnist
        (x_learn_m, y_learn_m), (x_test_m, y_test_m) = mnist.load_data()#y is output label data   
        x_learn_m, y_learn_m = utils.shuffle(x_learn_m, y_learn_m) 
        x_test_m, y_test_m = utils.shuffle(x_test_m, y_test_m)    
        
        #emnistc
        x_test_em, label_test_em = extract_test_samples('letters')
        x_learn_em, label_learn_em = extract_training_samples('letters')
        x_test_em, label_test_em = utils.shuffle(x_test_em, label_test_em)#randomize order of emnist data
        x_learn_em, label_learn_em = utils.shuffle(x_learn_em, label_learn_em)#randomize order of emnist data
        y_test_em = np.array([10 for i in range(len(label_test_em))])#y_em are all 10 = U
        y_learn_em = np.array([10 for i in range(len(label_learn_em))])#y_em are all 10 = U
       
        #fashion mnist
        (x_learn_f,y_learn_f),(x_test_f,y_test_f) = fashion_mnist.load_data()
        x_learn_f, y_learn_f = utils.shuffle(x_learn_f, y_learn_f) 
        x_test_f, y_test_f = utils.shuffle(x_test_f, y_test_f) 
        y_learn_f = np.array([10 for i in range(len(y_learn_f))])#y_em are all 10 = U
        y_test_f = np.array([10 for i in range(len(y_test_f))])#y_em are all 10 = U
        
        
        #test, learn入出力データをpreprocess_imagesで変形する
        x_learn_m, y_learn_m = preprocess_images(x_learn_m, y_learn_m)
        x_learn_f, y_learn_f = preprocess_images(x_learn_f, y_learn_f)
        x_learn_em, y_learn_em = preprocess_images(x_learn_em, y_learn_em)
        x_test_m, y_test_m = preprocess_images(x_test_m, y_test_m)
        x_test_f, y_test_f = preprocess_images(x_test_f, y_test_f)
        x_test_em, y_test_em = preprocess_images(x_test_em, y_test_em)#onehot vector y_test_em includes class10 = U, 11D
        
        #combine data: LnuTwu = Learning: mnist&emnist, Test: mnist&fashion
        #yu = 1 when U, 11D
        mix_rate_ts = [i * 0.25 for i in range(1,4)]
        accs_k_forMixt = []
        accs_U_forMixt = []
        recalls_k_forMixt = []
        recalls_U_forMixt = []
        precs_k_forMixt = []
        precs_U_forMixt = []
        for mix_rate_t in mix_rate_ts:
            # x_test_em_pick = x_test_em[:int(len(x_test_m)*mix_rate_t)]
            # y_test_em_pick = y_test_em[:int(len(x_test_m)*mix_rate_t)]
            # x_test_mix = np.concatenate([x_test_m,x_test_em_pick])
            # y_test_mix = np.concatenate([y_test_m,y_test_em_pick])  
            x_test_f_pick = x_test_f[:int(len(x_test_m)*mix_rate_t)]
            y_test_f_pick = y_test_f[:int(len(x_test_m)*mix_rate_t)]
            x_test_mix = np.concatenate([x_test_m,x_test_f_pick])
            y_test_mix = np.concatenate([y_test_m,y_test_f_pick])  
        
            #一旦shuffleして規定のデータ数に揃える learn 60000 test 10000
            #x_learn_mix, y_learn_mix = utils.shuffle(x_learn_mix, y_learn_mix,n_samples=60000)
            x_test_mix, y_test_mix = utils.shuffle(x_test_mix, y_test_mix,n_samples=10000)
            
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
            #model.fit(x_learn_m, y_learn_m, batch_size=64, shuffle=True, validation_split=0.1, epochs=10)#epochs=10
            
            es = EarlyStopping(monitor='val_loss',patience=3,min_delta=0.001,verbose=1)
            model.fit(x_learn_m, y_learn_m, batch_size=64, shuffle=True, validation_split=0.1,callbacks=[es], epochs=100)#epochs=
            
            #Model Evaluation!!!!!!!!!!!!!!!!!!!!!!!!!!!
            evidence = model.predict(x_test_mix)
            evi, alp, belief, unc, y_pred, p_hat_k = calc_enn_output_from_evidence(evidence)
            p_hat_k_sum = p_hat_k.sum(axis=1)
            
            #add see notebook 20220301
            
            #y_test_mix[10] =#ytestmix のuのところにlabel 10を入れる
            ins_column = [[1] if len(np.unique(y_test_mix[i])) == 1 else [0] for i in range(len(y_test_mix))]
            y_test_mix = np.hstack((y_test_mix,ins_column))
            y_test_label = np.argmax(y_test_mix,axis=1)#labelに戻
            label_list =[0,1,2,3,4,5,6,7,8,9,10]#with U
            
            #定めたunc thresholdよりもuncが大きいデータを消す
            unc_ths = [i * 0.1 for i in range(0,11)]
            accs_k = []
            accs_U = []
            recalls_k = []
            recalls_U = []
            precs_k = []
            precs_U = []
            for unc_th in unc_ths:
                k_data, U_data = remove_data_with_large_unc(np.array(unc),unc_th,y_pred,y_test_label)
                unc_pick_k, y_pred_pick_k, y_test_label_pick_k = k_data#data classified class k from unc_th
                
                #y_pred labels where unc is large, y_test_label(correct label) where unc is large
                unc_pick_U, y_pred_pick_U, y_test_label_pick_U = U_data#data classified uncertain from unc_th
            
                conf_mat_k = confusion_matrix(y_test_label_pick_k,y_pred_pick_k,labels=label_list)
                conf_mat_U = confusion_matrix(y_test_label_pick_U,y_pred_pick_U,labels=label_list)
                acc_k = accuracy_score(y_test_label_pick_k,y_pred_pick_k)
                acc_U = accuracy_score(y_test_label_pick_U,y_pred_pick_U)
                accs_k.append(acc_k)
                accs_U.append(acc_U)
                recall_k = recall_score(y_test_label_pick_k,y_pred_pick_k,average='micro')
                recall_U = recall_score(y_test_label_pick_U,y_pred_pick_U,average='micro')
                recalls_k.append(recall_k)
                recalls_U.append(recall_U)
                prec_k = precision_score(y_test_label_pick_k,y_pred_pick_k,average='micro')
                prec_U = precision_score(y_test_label_pick_U,y_pred_pick_U,average='micro')
                precs_k.append(prec_k)
                precs_U.append(prec_U)
            
            accs_k_forMixt.append(accs_k)
            accs_U_forMixt.append(accs_U)
            recalls_k_forMixt.append(recalls_k)
            recalls_U_forMixt.append(recalls_U)
            precs_k_forMixt.append(precs_k)
            precs_U_forMixt.append(precs_U)
        
        accs_k_forMixt_df = pandas.DataFrame(accs_k_forMixt,columns =['{:.1f}'.format(x) for x in unc_ths], index = mix_rate_ts)
        accs_U_forMixt_df = pandas.DataFrame(accs_U_forMixt,columns =['{:.1f}'.format(x) for x in unc_ths], index = mix_rate_ts)
        
        accs_k_forMixt_df.to_csv(summary_folder+r'{0}_ltr_accs_k_{1}.csv'.format('1_LnuTwu',i_trial))#index is mix rate for test data, column is unc_th
        accs_U_forMixt_df.to_csv(summary_folder+r'{0}_ltr_accs_U_{1}.csv'.format('1_LnuTwu',i_trial))
        
        recalls_k_forMixt_df = pandas.DataFrame(recalls_k_forMixt,columns =['{:.1f}'.format(x) for x in unc_ths], index = mix_rate_ts)
        recalls_U_forMixt_df = pandas.DataFrame(recalls_U_forMixt,columns =['{:.1f}'.format(x) for x in unc_ths], index = mix_rate_ts)
        
        recalls_k_forMixt_df.to_csv(summary_folder+r'{0}_ltr_recalls_k_{1}.csv'.format('1_LnuTwu',i_trial))#index is mix rate for test data, column is unc_th
        recalls_U_forMixt_df.to_csv(summary_folder+r'{0}_ltr_recalls_U_{1}.csv'.format('1_LnuTwu',i_trial))
        
        precs_k_forMixt_df = pandas.DataFrame(precs_k_forMixt,columns =['{:.1f}'.format(x) for x in unc_ths], index = mix_rate_ts)
        precs_U_forMixt_df = pandas.DataFrame(precs_U_forMixt,columns =['{:.1f}'.format(x) for x in unc_ths], index = mix_rate_ts)
        
        precs_k_forMixt_df.to_csv(summary_folder+r'{0}_ltr_precs_k_{1}.csv'.format('1_LnuTwu',i_trial))#index is mix rate for test data, column is unc_th
        precs_U_forMixt_df.to_csv(summary_folder+r'{0}_ltr_precs_U_{1}.csv'.format('1_LnuTwu',i_trial))
                
        #make_histgram(uncertainty=np.array(unc), y_true=np.argmax(y_test_mix, axis=1), y_pred=y_pred)
        #show_true_image(x_test_mix, y_test_mix,belief,unc,p_hat_k)
        

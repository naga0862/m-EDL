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
summary_folder = r'Z:/ENN_acc/2_LwuTwu_iiX_f/'
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
    
    y_pred_Phat = (
    np.argmax(Phat_kp, axis=1)
    )#Phatを使った予測ラベル
    y_pred_Pbar_k = (
    np.argmax(Pbar_k, axis=1)
    )#Pbar kを使った予測ラベル
    
    return evidence, alpha_k, alpha_u, bk, bu, y_pred_Phat, y_pred_Pbar_k, Phat_k, Phat_u, Phat_kp, Pbar_k, S.numpy(), K
    


# MNISTデータの前処理関数（最大値で正規化&[0, 1]の2値に変換)．NOTE:one-hotに変換していることに注意
#変数images, labelsはそれぞれnp.ndarray 出力は、np.ndarray, tf.Tensorのtuple型
#imagesは28*28=784列のベクトルがデータ数分だけ並んだ2D arrayに変換される
#出力tf.Tensorはone-hot横ベクトルがデータ数分だけ並んだtf.Tensor
def preprocess_images(images: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, tf.Tensor]:
    images = images.reshape((images.shape[0], 28*28)) / 255.0
    #return np.where(images > 0.5, 1.0, 0.0).astype("float32"), tf.one_hot(indices=label, depth=10)
    #return np.where(images > 0.5, 1.0, 0.0).astype("float32"), tf.one_hot(indices=label, depth=11)#See notebook20220202
    return images.astype("float32"), tf.one_hot(indices=label, depth=11)#See notebook20220202

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
    
    for i_trial in range (0,100):
        print('2_LwuTwu_iiX_f {}th trial !!!!!!!!!!'.format(i_trial+1))
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
        #x_test_c = (tf.image.rgb_to_grayscale(x_test_c)).numpy()
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
        
        #combine data: LwuTwu = Learning: mnist&em, Test: mnist&fashion
        #yu = 1 when U, 11D
        mix_rate_ls = [i * 0.25 for i in range(1,4)] 
        accs_k_forMixlt = []
        accs_U_forMixlt = []
        accs_k_forMixlt_df = []
        accs_U_forMixlt_df = []
        
        recalls_k_forMixlt = []
        recalls_U_forMixlt = []
        recalls_k_forMixlt_df = []
        recalls_U_forMixlt_df = []
        
        precs_k_forMixlt = []
        precs_U_forMixlt = []
        precs_k_forMixlt_df = []
        precs_U_forMixlt_df = []
        for mix_rate_l in mix_rate_ls:#loop for mix rate for learn data
            mix_rate_ts = [j * 0.25 for j in range(1,4)]
            accs_k_forMixt = []
            accs_U_forMixt = []
            recalls_k_forMixt = []
            recalls_U_forMixt = []
            precs_k_forMixt = []
            precs_U_forMixt = []
            
            for mix_rate_t in mix_rate_ts:#loop for mix rate for test data
                #dataの混合learn
                #x_learn_em_pick = x_learn_em[:int(len(x_learn_m)*mix_rate_l)]
                #y_learn_em_pick = y_learn_em[:int(len(x_learn_m)*mix_rate_l)]
                x_learn_f_pick = x_learn_f[:int(len(x_learn_m)*mix_rate_l)]
                y_learn_f_pick = y_learn_f[:int(len(x_learn_m)*mix_rate_l)]
                x_learn_mix = np.concatenate([x_learn_m,x_learn_f_pick])
                y_learn_mix = np.concatenate([y_learn_m,y_learn_f_pick])
                #一旦shuffleして規定のデータ数に揃える learn 60000 test 10000
                x_learn_mix, y_learn_mix = utils.shuffle(x_learn_mix, y_learn_mix,n_samples=60000)
                
                #dataの混合test
                x_test_f_pick = x_test_f[:int(len(x_test_m)*mix_rate_t)]
                y_test_f_pick = y_test_f[:int(len(x_test_m)*mix_rate_t)]
                x_test_mix = np.concatenate([x_test_m,x_test_f_pick])
                y_test_mix = np.concatenate([y_test_m,y_test_f_pick])
                
                #一旦shuffleして規定のデータ数に揃える test 10000
                x_test_mix, y_test_mix = utils.shuffle(x_test_mix, y_test_mix,n_samples=10000)
                
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
                
                #model.compile(optimizer=tf.keras.optimizers.Adam(), loss=EDLLoss(K=10, annealing=0.1), metrics=["accuracy", "mse"])
                model.compile(optimizer=tf.keras.optimizers.Adam(), loss=EDLLoss_p(K=10, annealing=0.1), metrics=["accuracy", "mse"])#see notebook20220202
                #model.fit(x_learn_m, y_learn_m, batch_size=64, shuffle=True,validation_split=0.1, epochs=10)#epochs=10
                
                es = EarlyStopping(monitor='val_loss',patience=3,min_delta=0.001,verbose=1)
                model.fit(x_learn_mix, y_learn_mix, batch_size=64, shuffle=True, validation_split=0.1,callbacks=[es], epochs=100)#epochs=
                #"model evaluation"
                
                #Model Evaluation!!!!!!!!!!!!!!!!!!!!!!!!!!!
                #evidence = model.predict(x_test_m)#mnist only ekに相当
                evidence = model.predict(x_test_mix)#mnist + emnist ekに相当
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
                
                #unc_thを基にuncを決めるのではないので以下不要
                #y_test_mix[10] =#ytestmix のuのところにlabel 10を入れる
                # ins_column = [[1] if len(np.unique(y_test_mix[i])) == 1 else [0] for i in range(len(y_test_mix))]
                # y_test_mix = np.hstack((y_test_mix,ins_column))
                y_test_label = np.argmax(y_test_mix,axis=1)#labelに戻
                # label_list =[0,1,2,3,4,5,6,7,8,9,10]#with U
                
                # #定めたunc thresholdよりもuncが大きいデータを消す
                # unc_ths = [i * 0.1 for i in range(0,11)]
                # accs_k = []
                # accs_U = []
                # for unc_th in unc_ths:
                #     k_data, U_data = remove_data_with_large_unc(np.array(unc),unc_th,y_pred,y_test_label)
                #     unc_pick_k, y_pred_pick_k, y_test_label_pick_k = k_data#data classified class k from unc_th
                    
                #     #y_pred labels where unc is large, y_test_label(correct label) where unc is large
                #     unc_pick_U, y_pred_pick_U, y_test_label_pick_U = U_data#data classified uncertain from unc_th
                
                #     conf_mat_k = confusion_matrix(y_test_label_pick_k,y_pred_pick_k,labels=label_list)
                #     conf_mat_U = confusion_matrix(y_test_label_pick_U,y_pred_pick_U,labels=label_list)
                #     acc_k = accuracy_score(y_test_label_pick_k,y_pred_pick_k)
                #     acc_U = accuracy_score(y_test_label_pick_U,y_pred_pick_U)
                #     accs_k.append(acc_k)
                #     accs_U.append(acc_U)
                
                #むしろunc_thに依らないaccをp_hatに基づく予測ラベルから求める
                y_pred_p_hat_pick_k = y_pred_phat[np.where(y_test_label != 10)]#クラスkを予測した予測ラベルのみ抽出
                y_test_label_pick_k = y_test_label[np.where(y_test_label != 10)]#クラスkを予測したデータの正答ラベルのみ抽出
                y_pred_p_hat_pick_U = y_pred_phat[np.where(y_test_label == 10)]#クラスUを予測した予測ラベルのみ抽出
                y_test_label_pick_U = y_test_label[np.where(y_test_label == 10)]#クラスUを予測したデータの正答ラベルのみ抽出
    
                acc_k = accuracy_score(y_test_label_pick_k,y_pred_p_hat_pick_k)
                acc_U = accuracy_score(y_test_label_pick_U,y_pred_p_hat_pick_U)
                recall_k = recall_score(y_test_label_pick_k,y_pred_p_hat_pick_k,average='micro')
                recall_U = recall_score(y_test_label_pick_U,y_pred_p_hat_pick_U,average='micro')
                prec_k = precision_score(y_test_label_pick_k,y_pred_p_hat_pick_k,average='micro')
                prec_U = precision_score(y_test_label_pick_U,y_pred_p_hat_pick_U,average='micro')
                
                accs_k_forMixt.append([acc_k])
                accs_U_forMixt.append([acc_U])
                recalls_k_forMixt.append([recall_k])
                recalls_U_forMixt.append([recall_U])
                precs_k_forMixt.append([prec_k])
                precs_U_forMixt.append([prec_U])
            
    
            accs_k_forMixlt.append(accs_k_forMixt)
            accs_U_forMixlt.append(accs_U_forMixt)
            recalls_k_forMixlt.append(recalls_k_forMixt)
            recalls_U_forMixlt.append(recalls_U_forMixt)
            precs_k_forMixlt.append(precs_k_forMixt)
            precs_U_forMixlt.append(precs_U_forMixt)
            
            if len(accs_k_forMixlt_df) == 0:
                accs_k_forMixlt_df = pandas.DataFrame(accs_k_forMixt,index = mix_rate_ts, columns = [mix_rate_l])
                accs_U_forMixlt_df = pandas.DataFrame(accs_U_forMixt,index = mix_rate_ts, columns = [mix_rate_l])
                recalls_k_forMixlt_df = pandas.DataFrame(recalls_k_forMixt,index = mix_rate_ts, columns = [mix_rate_l])
                recalls_U_forMixlt_df = pandas.DataFrame(recalls_U_forMixt,index = mix_rate_ts, columns = [mix_rate_l])
                precs_k_forMixlt_df = pandas.DataFrame(precs_k_forMixt,index = mix_rate_ts, columns = [mix_rate_l])
                precs_U_forMixlt_df = pandas.DataFrame(precs_U_forMixt,index = mix_rate_ts, columns = [mix_rate_l])
            else:
                add_accs_k_forMixlt_df = pandas.DataFrame(accs_k_forMixt,index = mix_rate_ts, columns = [mix_rate_l])
                add_accs_U_forMixlt_df = pandas.DataFrame(accs_U_forMixt,index = mix_rate_ts, columns = [mix_rate_l])
                add_recalls_k_forMixlt_df = pandas.DataFrame(recalls_k_forMixt,index = mix_rate_ts, columns = [mix_rate_l])
                add_recalls_U_forMixlt_df = pandas.DataFrame(recalls_U_forMixt,index = mix_rate_ts, columns = [mix_rate_l])
                add_precs_k_forMixlt_df = pandas.DataFrame(precs_k_forMixt,index = mix_rate_ts, columns = [mix_rate_l])
                add_precs_U_forMixlt_df = pandas.DataFrame(precs_U_forMixt,index = mix_rate_ts, columns = [mix_rate_l])
                
                accs_k_forMixlt_df = pandas.concat([accs_k_forMixlt_df,add_accs_k_forMixlt_df],axis=1)
                accs_U_forMixlt_df = pandas.concat([accs_U_forMixlt_df,add_accs_U_forMixlt_df],axis=1)
                recalls_k_forMixlt_df = pandas.concat([recalls_k_forMixlt_df,add_recalls_k_forMixlt_df],axis=1)
                recalls_U_forMixlt_df = pandas.concat([recalls_U_forMixlt_df,add_recalls_U_forMixlt_df],axis=1)
                precs_k_forMixlt_df = pandas.concat([precs_k_forMixlt_df,add_precs_k_forMixlt_df],axis=1)
                precs_U_forMixlt_df = pandas.concat([precs_U_forMixlt_df,add_precs_U_forMixlt_df],axis=1)
                
        
        accs_k_forMixlt_df.to_csv(summary_folder+r'{0}_accs_k_{1}.csv'.format('2_LwuTwu_iiX_f',i_trial))#index=mix_rate_t, column = mix_rate_learn
        accs_U_forMixlt_df.to_csv(summary_folder+r'{0}_accs_U_{1}.csv'.format('2_LwuTwu_iiX_f',i_trial))
        
        recalls_k_forMixlt_df.to_csv(summary_folder+r'{0}_recalls_k_{1}.csv'.format('2_LwuTwu_iiX_f',i_trial))#index=mix_rate_t, column = mix_rate_learn
        recalls_U_forMixlt_df.to_csv(summary_folder+r'{0}_recalls_U_{1}.csv'.format('2_LwuTwu_iiX_f',i_trial))
        
        precs_k_forMixlt_df.to_csv(summary_folder+r'{0}_precs_k_{1}.csv'.format('2_LwuTwu_iiX_f',i_trial))#index=mix_rate_t, column = mix_rate_learn
        precs_U_forMixlt_df.to_csv(summary_folder+r'{0}_precs_U_{1}.csv'.format('2_LwuTwu_iiX_f',i_trial))
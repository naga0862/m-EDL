# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:14:21 2022

@author: a_nagahama_r
"""

import pandas
import matplotlib.pyplot as plt
import numpy as np
summary_folder = r'Z:/ENN_acc/'

def comp_iii(cond1_name,cond2_name,mode):
    accs_k_forMixt_1 = []
    accs_U_forMixt_1 = []
    accs_k_forMixt_2 = []
    accs_U_forMixt_2 = []
    
    for i_trial in range (0,100):
        if len(accs_k_forMixt_1) == 0:
            accs_k_forMixt_1 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}s_k_{2}.csv'.format(cond1_name,mode,i_trial),index_col=0)
            accs_U_forMixt_1 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}s_U_{2}.csv'.format(cond1_name,mode,i_trial),index_col=0)
            accs_k_forMixt_2 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}s_k_{2}.csv'.format(cond2_name,mode,i_trial),index_col=0)
            accs_U_forMixt_2 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}s_U_{2}.csv'.format(cond2_name,mode,i_trial),index_col=0)
            
            column_list_1 = accs_k_forMixt_1.columns
            index_list_1 = accs_k_forMixt_1.index
            column_list_2 = accs_k_forMixt_2.columns
            index_list_2 = accs_k_forMixt_2.index
            
            accs_k_forMixt_1 = accs_k_forMixt_1.values#change numpy ndarray
            accs_U_forMixt_1 = accs_U_forMixt_1.values#change numpy ndarray
            accs_k_forMixt_2 = accs_k_forMixt_2.values#change numpy ndarray
            accs_U_forMixt_2 = accs_U_forMixt_2.values#change numpy ndarray
            
            accs_k_forMixt_1 = accs_k_forMixt_1.reshape(3,11,1)#change 2D array (3x3) to 3D array (1x3x3)
            accs_U_forMixt_1 = accs_U_forMixt_1.reshape(3,11,1)
            accs_k_forMixt_2 = accs_k_forMixt_2.reshape(3,1,1)
            print(accs_k_forMixt_2)
            accs_U_forMixt_2 = accs_U_forMixt_2.reshape(3,1,1)
            
        else:
            add_accs_k_forMixt_1 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}s_k_{2}.csv'.format(cond1_name,mode,i_trial),index_col=0)
            add_accs_U_forMixt_1 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}s_U_{2}.csv'.format(cond1_name,mode,i_trial),index_col=0)
            add_accs_k_forMixt_2 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}s_k_{2}.csv'.format(cond2_name,mode,i_trial),index_col=0)
            add_accs_U_forMixt_2 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}s_U_{2}.csv'.format(cond2_name,mode,i_trial),index_col=0)
            
            add_accs_k_forMixt_1 = add_accs_k_forMixt_1.values.reshape(3,11,1)#change numpy ndarray
            add_accs_U_forMixt_1 = add_accs_U_forMixt_1.values.reshape(3,11,1)#change numpy ndarray
            add_accs_k_forMixt_2 = add_accs_k_forMixt_2.values.reshape(3,1,1)#change numpy ndarray
            add_accs_U_forMixt_2 = add_accs_U_forMixt_2.values.reshape(3,1,1)#change numpy ndarray
            
            accs_k_forMixt_1 = np.append(accs_k_forMixt_1, add_accs_k_forMixt_1,axis=2)
            accs_U_forMixt_1 = np.append(accs_U_forMixt_1, add_accs_U_forMixt_1,axis=2)
            accs_k_forMixt_2 = np.append(accs_k_forMixt_2, add_accs_k_forMixt_2,axis=2)
            accs_U_forMixt_2 = np.append(accs_U_forMixt_2, add_accs_U_forMixt_2,axis=2)
    print(accs_k_forMixt_2)
    mean_accs_k_forMixt_1 = np.mean(accs_k_forMixt_1,axis=2)
    mean_accs_U_forMixt_1 = np.mean(accs_U_forMixt_1,axis=2)
    mean_accs_k_forMixt_2 = np.mean(accs_k_forMixt_2,axis=2)
    print(mean_accs_k_forMixt_2)
    mean_accs_U_forMixt_2 = np.mean(accs_U_forMixt_2,axis=2)
    std_accs_k_forMixt_1 = np.std(accs_k_forMixt_1,axis=2)
    std_accs_U_forMixt_1 = np.std(accs_U_forMixt_1,axis=2)
    std_accs_k_forMixt_2 = np.std(accs_k_forMixt_2,axis=2)
    std_accs_U_forMixt_2 = np.std(accs_U_forMixt_2,axis=2)
    
    accs_k_forMixt_1 = pandas.DataFrame(data = mean_accs_k_forMixt_1, index = index_list_1, columns = column_list_1, dtype = 'float')
    accs_U_forMixt_1 = pandas.DataFrame(data = mean_accs_U_forMixt_1, index = index_list_1, columns = column_list_1, dtype = 'float')
    accs_k_forMixt_2 = pandas.DataFrame(data = mean_accs_k_forMixt_2, index = index_list_2, columns = column_list_2, dtype = 'float')
    accs_U_forMixt_2 = pandas.DataFrame(data = mean_accs_U_forMixt_2, index = index_list_2, columns = column_list_2, dtype = 'float')
    std_accs_k_forMixt_1 = pandas.DataFrame(data = std_accs_k_forMixt_1, index = index_list_1, columns = column_list_1, dtype = 'float')
    std_accs_U_forMixt_1 = pandas.DataFrame(data = std_accs_U_forMixt_1, index = index_list_1, columns = column_list_1, dtype = 'float')
    std_accs_k_forMixt_2 = pandas.DataFrame(data = std_accs_k_forMixt_2, index = index_list_2, columns = column_list_2, dtype = 'float')
    std_accs_U_forMixt_2 = pandas.DataFrame(data = std_accs_U_forMixt_2, index = index_list_2, columns = column_list_2, dtype = 'float')
    
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(1,2, sharex=True, sharey=True,figsize=[12, 8]) #...1

    mix_rate_ts = [i * 0.25 for i in range(1,4)]
    color_list = ['r','g','b']
    style_list = ['dotted','dashed','solid']
    
    
    for i_list in range(0,3):
        mix_rate_t = mix_rate_ts[i_list]
        acc_k_1 = accs_k_forMixt_1.loc[mix_rate_t,:]
        acc_U_1 = accs_U_forMixt_1.loc[mix_rate_t,:]
        acc_k_2 = accs_k_forMixt_2.loc[mix_rate_t,'0']
        acc_U_2 = accs_U_forMixt_2.loc[mix_rate_t,'0']
        std_k_1 = std_accs_k_forMixt_1.loc[mix_rate_t,:]
        std_U_1 = std_accs_U_forMixt_1.loc[mix_rate_t,:]
        std_k_2 = std_accs_k_forMixt_2.loc[mix_rate_t,'0']
        std_U_2 = std_accs_U_forMixt_2.loc[mix_rate_t,'0']
        
        ax[0].plot(accs_k_forMixt_1.columns, acc_k_1, label='{0}_{1}'.format(mix_rate_t,cond1_name),color=color_list[i_list], linestyle = style_list[i_list],linewidth = 2)
        ax[0].fill_between(accs_k_forMixt_1.columns, acc_k_1 - std_k_1, acc_k_1 + std_k_1,color=color_list[i_list],alpha=0.2)
        ax[0].axhline(y= acc_k_2, xmin=0, xmax=1.0,color=color_list[i_list], linestyle = style_list[i_list], label='{0}_{1}'.format(mix_rate_t,cond2_name),linewidth = 4)
        ax[0].axhspan(acc_k_2 - std_k_2, acc_k_2 + std_k_2, facecolor=color_list[i_list], alpha=0.2)
        
        #plt.title('For class ks')
        ax[0].set_xlim(0, 10)#unc_th 0-1 for 1th to 10th index
        ax[0].set_ylim(0,1)
        ax[0].xaxis.set_ticks(np.arange(0,11,2))
        ax[0].tick_params(axis = 'x', labelsize = 18)
        ax[0].tick_params(axis = 'y', labelsize = 18)
        #ax[0].grid(False)
        
        ax[1].plot(accs_U_forMixt_1.columns, acc_U_1, label='{0}_{1}'.format(mix_rate_t,cond1_name),color=color_list[i_list], linestyle = style_list[i_list],linewidth = 2)
        ax[1].fill_between(accs_U_forMixt_1.columns, acc_U_1 - std_U_1, acc_U_1 + std_U_1,color=color_list[i_list],alpha=0.2)
        ax[1].axhline(y= acc_U_2, xmin=0, xmax=1.0,color=color_list[i_list], linestyle = style_list[i_list],label='{0}_{1}'.format(mix_rate_t,cond2_name),linewidth = 4)
        ax[1].axhspan(acc_U_2 - std_U_2, acc_U_2 + std_U_2, facecolor=color_list[i_list], alpha=0.2)
        
        #ax[1].fill_between(accs_U_forMixt_2.columns, acc_U_2 - std_U_2, acc_U_2 + std_U_2,color=color_list[i_list],alpha=0.2)
        #ax[1].set_xlim(0, 10)#unc_th 0-1 for 1th to 10th index
        #ax[1].set_ylim(0,1)
        #plt.title('For class U')
        #ax[1].grid(False)
        ax[1].tick_params(axis = 'x', labelsize = 18)
        ax[1].tick_params(axis = 'y', labelsize = 18)
        
    #ax[0].set_xlim(0,1)    
    
    # Adding a plot in the figure which will encapsulate all the subplots with axis showing only
    fig.add_subplot(1, 1, 1, frame_on=False)
    plt.grid(False)

    # Hiding the axis ticks and tick labels of the bigger plot
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    
    # Adding the x-axis and y-axis labels for the bigger plot
    plt.xlabel('Uncertainty threshold (-)',loc='center',fontsize=20,labelpad=15)
    if mode == 'acc':
        plt.ylabel('Accuracy (-)',loc='center',fontsize=20,labelpad=15)
    elif mode == 'recall':
        plt.ylabel('Recall (-)',loc='center',fontsize=20,labelpad=15)
    elif mode == 'prec':
        plt.ylabel('Precision (-)',loc='center',fontsize=20,labelpad=15)
    
    # 凡例の表示
    lines, labels = ax[1].get_legend_handles_labels()
    fig.legend(lines, labels, bbox_to_anchor=(0,0),loc='lower left', borderaxespad=8, fontsize=17)

    #plt.tight_layout()
    # プロット表示(設定の反映)
    #plt.show()
    
    plt.savefig(summary_folder+r'comp_iii_{0}_{1}_{2}.png'.format(mode,cond1_name,cond2_name))

def comp_ii(cond1_name,cond2_name,mix_l,mode):
    
    accs_k_forMixtest_1 = []
    accs_U_forMixtest_1 = []
    accs_k_forMixtestlearn_2 = []
    accs_U_forMixtestlearn_2 = []
    
    #print(cond1_name)
    #print(cond2_name)
    #print(mode)
    
    for i_trial in range (0,100):
        if len(accs_k_forMixtest_1) == 0:
            accs_k_forMixtest_1 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}s_k_{2}.csv'.format(cond1_name,mode,i_trial),index_col=0)
            accs_U_forMixtest_1 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}s_U_{2}.csv'.format(cond1_name,mode,i_trial),index_col=0)
            accs_k_forMixtestlearn_2 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}s_k_{2}.csv'.format(cond2_name,mode,i_trial),index_col=0)
            accs_U_forMixtestlearn_2 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}s_U_{2}.csv'.format(cond2_name,mode,i_trial),index_col=0)
            column_list_1 = accs_k_forMixtest_1.columns
            index_list_1 = accs_k_forMixtest_1.index
            column_list_2 = accs_k_forMixtestlearn_2.columns
            index_list_2 = accs_k_forMixtestlearn_2.index
            
            accs_k_forMixtest_1 = accs_k_forMixtest_1.values#change numpy ndarray
            accs_U_forMixtest_1 = accs_U_forMixtest_1.values#change numpy ndarray
            accs_k_forMixtestlearn_2 = accs_k_forMixtestlearn_2.values#change numpy ndarray
            accs_U_forMixtestlearn_2 = accs_U_forMixtestlearn_2.values#change numpy ndarray
            
            accs_k_forMixtest_1 = accs_k_forMixtest_1.reshape(3,11,1)#change 2D array (3x3) to 3D array (1x3x3)
            #print(accs_k_forMixtestlearn_2)
            accs_U_forMixtest_1 = accs_U_forMixtest_1.reshape(3,11,1)
            accs_k_forMixtestlearn_2 = accs_k_forMixtestlearn_2.reshape(3,3,1)
            accs_U_forMixtestlearn_2 = accs_U_forMixtestlearn_2.reshape(3,3,1)
            #print(accs_k_forMixtestlearn_2)
            
            
            
        else:
            add_accs_k_forMixtest_1 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}s_k_{2}.csv'.format(cond1_name,mode,i_trial),index_col=0)
            add_accs_U_forMixtest_1 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}s_U_{2}.csv'.format(cond1_name,mode,i_trial),index_col=0)
            add_accs_k_forMixtestlearn_2 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}s_k_{2}.csv'.format(cond2_name,mode,i_trial),index_col=0)
            add_accs_U_forMixtestlearn_2 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}s_U_{2}.csv'.format(cond2_name,mode,i_trial),index_col=0)
            
            add_accs_k_forMixtest_1 = add_accs_k_forMixtest_1.values.reshape(3,11,1)
            add_accs_U_forMixtest_1 = add_accs_U_forMixtest_1.values.reshape(3,11,1)
            add_accs_k_forMixtestlearn_2 = add_accs_k_forMixtestlearn_2.values.reshape(3,3,1)
            add_accs_U_forMixtestlearn_2 = add_accs_U_forMixtestlearn_2.values.reshape(3,3,1)
            
            accs_k_forMixtest_1 = np.append(accs_k_forMixtest_1,add_accs_k_forMixtest_1,axis=2)
            accs_U_forMixtest_1 = np.append(accs_U_forMixtest_1,add_accs_U_forMixtest_1,axis=2)
            accs_k_forMixtestlearn_2 = np.append(accs_k_forMixtestlearn_2,add_accs_k_forMixtestlearn_2,axis=2)
            accs_U_forMixtestlearn_2 = np.append(accs_U_forMixtestlearn_2,add_accs_U_forMixtestlearn_2,axis=2)
            
            
    #print(accs_k_forMixtestlearn_2)
    
    mean_accs_k_forMixtest_1 = np.mean(accs_k_forMixtest_1,axis=2)
    mean_accs_U_forMixtest_1 = np.mean(accs_U_forMixtest_1,axis=2)
    mean_accs_k_forMixtestlearn_2 = np.mean(accs_k_forMixtestlearn_2,axis=2)
    mean_accs_U_forMixtestlearn_2 = np.mean(accs_U_forMixtestlearn_2,axis=2)
    std_accs_k_forMixtest_1 = np.std(accs_k_forMixtest_1,axis=2)
    std_accs_U_forMixtest_1 = np.std(accs_U_forMixtest_1,axis=2)
    std_accs_k_forMixtestlearn_2 = np.std(accs_k_forMixtestlearn_2,axis=2)
    std_accs_U_forMixtestlearn_2 = np.std(accs_U_forMixtestlearn_2,axis=2)
    #print(mean_accs_k_forMixtestlearn_2)
    
    accs_k_forMixtest_1 = pandas.DataFrame(data = mean_accs_k_forMixtest_1, index = index_list_1, columns = column_list_1, dtype = 'float')
    accs_U_forMixtest_1 = pandas.DataFrame(data = mean_accs_U_forMixtest_1, index = index_list_1, columns = column_list_1, dtype = 'float')
    accs_k_forMixtestlearn_2 = pandas.DataFrame(data = mean_accs_k_forMixtestlearn_2, index = index_list_2, columns = column_list_2, dtype = 'float')
    accs_U_forMixtestlearn_2 = pandas.DataFrame(data = mean_accs_U_forMixtestlearn_2, index = index_list_2, columns = column_list_2, dtype = 'float')
    std_accs_k_forMixtest_1 = pandas.DataFrame(data = std_accs_k_forMixtest_1, index = index_list_1, columns = column_list_1, dtype = 'float')
    std_accs_U_forMixtest_1 = pandas.DataFrame(data = std_accs_U_forMixtest_1, index = index_list_1, columns = column_list_1, dtype = 'float')
    std_accs_k_forMixtestlearn_2 = pandas.DataFrame(data = std_accs_k_forMixtestlearn_2, index = index_list_2, columns = column_list_2, dtype = 'float')
    std_accs_U_forMixtestlearn_2 = pandas.DataFrame(data = std_accs_U_forMixtestlearn_2, index = index_list_2, columns = column_list_2, dtype = 'float')
    
    
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(1,2, sharex=True, sharey=True,figsize=[12, 8]) #...1

    mix_rate_ts = [i * 0.25 for i in range(1,4)]
    color_list = ['r','g','b']
    style_list = ['dotted','dashed','solid']
    
    
    for i_list in range(0,3):
        mix_rate_t = mix_rate_ts[i_list]
        acc_k_1 = accs_k_forMixtest_1.loc[mix_rate_t,:]
        acc_U_1 = accs_U_forMixtest_1.loc[mix_rate_t,:]
        acc_k_2 = accs_k_forMixtestlearn_2.loc[mix_rate_t,'{}'.format(mix_l)]
        acc_U_2 = accs_U_forMixtestlearn_2.loc[mix_rate_t,'{}'.format(mix_l)]
        std_k_1 = std_accs_k_forMixtest_1.loc[mix_rate_t,:]
        std_U_1 = std_accs_U_forMixtest_1.loc[mix_rate_t,:]
        std_k_2 = std_accs_k_forMixtestlearn_2.loc[mix_rate_t,'{}'.format(mix_l)]
        std_U_2 = std_accs_U_forMixtestlearn_2.loc[mix_rate_t,'{}'.format(mix_l)]

        ax[0].plot(accs_k_forMixtest_1.columns, acc_k_1, label='{0}_{1}'.format(mix_rate_t,cond1_name),color=color_list[i_list], linestyle = style_list[i_list],linewidth = 2)
        ax[0].axhline(y= acc_k_2, xmin=0, xmax=1.0,color=color_list[i_list], linestyle = style_list[i_list], label='{0}_{1}'.format(mix_rate_t,cond2_name),linewidth = 4)
        #plt.title('For class ks')
        ax[0].fill_between(accs_k_forMixtest_1.columns, acc_k_1 - std_k_1, acc_k_1 + std_k_1,color=color_list[i_list],alpha=0.2)
        #ax[0].fill_between(accs_k_forMixtestlearn_2.columns, acc_k_2 - std_k_2, acc_k_2 + std_k_2)
        ax[0].axhspan(acc_k_2 - std_k_2, acc_k_2 + std_k_2, facecolor=color_list[i_list], alpha=0.2)
        
        ax[0].set_xlim(0, 10)#unc_th 0-1 for 1th to 10th index
        ax[0].set_ylim(0,1)
        ax[0].xaxis.set_ticks(np.arange(0,11,2))
        ax[0].tick_params(axis = 'x', labelsize = 18)
        ax[0].tick_params(axis = 'y', labelsize = 18)
        #ax[0].grid(False)
        
        ax[1].plot(accs_U_forMixtest_1.columns, acc_U_1, label='{0}_{1}'.format(mix_rate_t,cond1_name),color=color_list[i_list], linestyle = style_list[i_list],linewidth = 2)
        ax[1].axhline(y= acc_U_2, xmin=0, xmax=1.0,color=color_list[i_list], linestyle = style_list[i_list],label='{0}_{1}'.format(mix_rate_t,cond2_name),linewidth = 4)
        ax[1].fill_between(accs_U_forMixtest_1.columns, acc_U_1 - std_U_1, acc_U_1 + std_U_1,color=color_list[i_list],alpha=0.2)
        #ax[1].fill_between(accs_U_forMixtestlearn_2.columns, acc_U_2 - std_U_2, acc_U_2 + std_U_2)
        ax[1].axhspan(acc_U_2 - std_U_2, acc_U_2 + std_U_2, facecolor=color_list[i_list], alpha=0.2)
        
        #ax[1].set_xlim(0, 10)#unc_th 0-1 for 1th to 10th index
        #ax[1].set_ylim(0,1)
        #plt.title('For class U')
        #ax[1].grid(False)
        ax[1].tick_params(axis = 'x', labelsize = 18)
        ax[1].tick_params(axis = 'y', labelsize = 18)
        
    #ax[0].set_xlim(0,1)    
    
    # Adding a plot in the figure which will encapsulate all the subplots with axis showing only
    fig.add_subplot(1, 1, 1, frame_on=False)
    plt.grid(False)

    # Hiding the axis ticks and tick labels of the bigger plot
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    
    # Adding the x-axis and y-axis labels for the bigger plot
    plt.xlabel('Uncertainty threshold (-)',loc='center',fontsize=20,labelpad=15)
    if mode == 'acc':
        plt.ylabel('Accuracy (-)',loc='center',fontsize=20,labelpad=15)
    elif mode == 'recall':
        plt.ylabel('Recall (-)',loc='center',fontsize=20,labelpad=15)
    elif mode == 'prec':
        plt.ylabel('Precision (-)',loc='center',fontsize=20,labelpad=15)
    
    # 凡例の表示
    lines, labels = ax[1].get_legend_handles_labels()
    fig.legend(lines, labels, bbox_to_anchor=(0,0),loc='lower left', borderaxespad=8, fontsize=17)

    #plt.tight_layout()
    # プロット表示(設定の反映)
    #plt.show()
    
    plt.savefig(summary_folder+r'comp_ii_{0}_{1}_{2}_mixl{3}.png'.format(mode,cond1_name,cond2_name,mix_l))


def comp_i(cond1_name,cond2_name,mode):
    
    accs_k_1 = []
    accs_pbars_k_2 = []
    accs_phats_k_2 = []
    
    for i_trial in range (0,100):
        if len(accs_k_1) == 0:
            accs_k_1 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}s_k_{2}.csv'.format(cond1_name,mode,i_trial),index_col=0)
            #accs_U_1 = pandas.read_csv(summary_folder+r'{0}_accs_U.csv'.format(cond1_name),index_col=0)
            accs_pbars_k_2 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}_pbars_k_{2}.csv'.format(cond2_name,mode,i_trial),index_col=0)
            accs_phats_k_2 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}_phats_k_{2}.csv'.format(cond2_name,mode,i_trial),index_col=0)
            
            column_list = accs_k_1.columns
            index_list = accs_k_1.index
            #accs_U_2 = pandas.read_csv(summary_folder+r'{0}_accs_U.csv'.format(cond2_name),index_col=0)
            accs_k_1 = accs_k_1.values#change numpy ndarray
            accs_pbars_k_2 = accs_pbars_k_2.values#change numpy ndarray
            accs_phats_k_2 = accs_phats_k_2.values#change numpy ndarray
    
            print(accs_k_1)
            accs_k_1 = accs_k_1.reshape(11,1,1)#change 2D array (11x1) to 3D array (11x1x1)
            print(accs_k_1)
            accs_pbars_k_2 = accs_pbars_k_2.reshape(11,1,1)#change 2D array (11x1) to 3D array (11x1x1)
            accs_phats_k_2 = accs_phats_k_2.reshape(11,1,1)#change 2D array (11x1) to 3D array (11x1x1)
            
            
            
        else:
            add_accs_k_1 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}s_k_{2}.csv'.format(cond1_name,mode,i_trial),index_col=0)
            #accs_U_1 = pandas.read_csv(summary_folder+r'{0}_accs_U.csv'.format(cond1_name),index_col=0)
            add_accs_pbars_k_2 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}_pbars_k_{2}.csv'.format(cond2_name,mode,i_trial),index_col=0)
            add_accs_phats_k_2 = pandas.read_csv(summary_folder+r'{0}/{0}_{1}_phats_k_{2}.csv'.format(cond2_name,mode,i_trial),index_col=0)
            
            add_accs_k_1 = add_accs_k_1.values.reshape(11,1,1)
            add_accs_pbars_k_2 = add_accs_pbars_k_2.values.reshape(11,1,1)
            add_accs_phats_k_2 = add_accs_phats_k_2.values.reshape(11,1,1)
            
            accs_k_1 = np.append(accs_k_1,add_accs_k_1,axis=2)
            accs_pbars_k_2 = np.append(accs_pbars_k_2,add_accs_pbars_k_2,axis=2)
            accs_phats_k_2 = np.append(accs_phats_k_2,add_accs_phats_k_2,axis=2)
    #print(accs_k_1)
    mean_accs_k_1 = np.mean(accs_k_1,axis=2)
    mean_accs_pbars_k_2 = np.mean(accs_pbars_k_2,axis=2)
    mean_accs_phats_k_2 = np.mean(accs_phats_k_2,axis=2)
    std_accs_k_1 = np.std(accs_k_1,axis=2)
    std_accs_pbars_k_2 = np.std(accs_pbars_k_2,axis=2)
    std_accs_phats_k_2 = np.std(accs_phats_k_2,axis=2)
    #print(mean_accs_k_1)
    
    accs_k_1 = pandas.DataFrame(data = mean_accs_k_1, index = index_list, columns = column_list, dtype = 'float')
    accs_pbars_k_2 = pandas.DataFrame(data = mean_accs_pbars_k_2, index = index_list, columns = column_list, dtype = 'float')
    accs_phats_k_2 = pandas.DataFrame(data = mean_accs_phats_k_2, index = index_list, columns = column_list, dtype = 'float')
    std_accs_k_1 = pandas.DataFrame(data = std_accs_k_1, index = index_list, columns = column_list, dtype = 'float')
    std_accs_pbars_k_2 = pandas.DataFrame(data = std_accs_pbars_k_2, index = index_list, columns = column_list, dtype = 'float')
    std_accs_phats_k_2 = pandas.DataFrame(data = std_accs_phats_k_2, index = index_list, columns = column_list, dtype = 'float')
    
    
    #For comparsion i################################################################################
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(1,1, sharex=True, sharey=True,figsize=[12, 8]) #...1

    #mix_rate_ts = [i * 0.25 for i in range(1,4)]
    color_list = ['r','g','b']
    style_list = ['dotted','dashed','solid']
    
    ax.plot(accs_k_1.index, accs_k_1.loc[:,'0'], label='{0}'.format(cond1_name),color=color_list[0], linestyle = style_list[2],linewidth = 2)
    ax.fill_between(accs_k_1.index, accs_k_1.loc[:,'0'] - std_accs_k_1.loc[:,'0'], accs_k_1.loc[:,'0'] + std_accs_k_1.loc[:,'0'], alpha=0.2,color=color_list[0])
    #ax.fill_between(accs_k_1.index, accs_k_1.loc[:,'0'] - 10, accs_k_1.loc[:,'0'] + 10, alpha=0.2, color='b')
    ax.plot(accs_pbars_k_2.index, accs_pbars_k_2.loc[:,'0'], label='{0}'.format(cond2_name),color=color_list[2], linestyle = style_list[2],linewidth = 4)
    ax.fill_between(accs_pbars_k_2.index, accs_pbars_k_2.loc[:,'0'] - std_accs_pbars_k_2.loc[:,'0'], accs_pbars_k_2.loc[:,'0'] + std_accs_pbars_k_2.loc[:,'0'], alpha=0.2,color=color_list[2])
    plt.title('For class ks')
    ax.set_xlim(0, 1)#unc_th 0-1 for 1th to 10th index
    ax.set_ylim(0,1)
    ax.xaxis.set_ticks(np.arange(0,1.1,0.2))
    ax.tick_params(axis = 'x', labelsize = 18)
    ax.tick_params(axis = 'y', labelsize = 18)
    #ax[0].grid(False)

    # Adding a plot in the figure which will encapsulate all the subplots with axis showing only
    fig.add_subplot(1, 1, 1, frame_on=False)
    plt.grid(False)

    # Hiding the axis ticks and tick labels of the bigger plot
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    
    # Adding the x-axis and y-axis labels for the bigger plot
    plt.xlabel('Uncertainty threshold (-)',loc='center',fontsize=20,labelpad=15)
    if mode == 'acc':
        plt.ylabel('Accuracy (-)',loc='center',fontsize=20,labelpad=15)
    elif mode == 'recall':
        plt.ylabel('Recall (-)',loc='center',fontsize=20,labelpad=15)
    elif mode == 'prec':
        plt.ylabel('Precision (-)',loc='center',fontsize=20,labelpad=15)
    
    # 凡例の表示
    lines, labels = ax.get_legend_handles_labels()
    fig.legend(lines, labels, bbox_to_anchor=(0,0),loc='lower left', borderaxespad=8, fontsize=17)

    #plt.tight_layout()
    # プロット表示(設定の反映)
    #plt.show()
    
    plt.savefig(summary_folder+r'comp_i_{0}_pbars_{1}_{2}.png'.format(mode,cond1_name,cond2_name))

#For comparsion i' see notebook 20220301################################################################################   
    fig2, ay = plt.subplots(1,1, sharex=True, sharey=True,figsize=[12, 8]) #...1

    #mix_rate_ts = [i * 0.25 for i in range(1,4)]
    color_list = ['r','g','b']
    style_list = ['dotted','dashed','solid']
    
    ay.plot(accs_k_1.index, accs_k_1.loc[:,'0'], label='{0}'.format(cond1_name),color=color_list[0], linestyle = style_list[2],linewidth = 2)
    ay.fill_between(accs_k_1.index, accs_k_1.loc[:,'0'] - std_accs_k_1.loc[:,'0'], accs_k_1.loc[:,'0'] + std_accs_k_1.loc[:,'0'], alpha=0.2,color=color_list[0])
    ay.plot(accs_phats_k_2.index, accs_phats_k_2.loc[:,'0'], label='{0}'.format(cond2_name),color=color_list[2], linestyle = style_list[2],linewidth = 4)
    ay.fill_between(accs_phats_k_2.index, accs_phats_k_2.loc[:,'0'] - std_accs_phats_k_2.loc[:,'0'], accs_phats_k_2.loc[:,'0'] + std_accs_phats_k_2.loc[:,'0'], alpha=0.2,color=color_list[2])
    
    plt.title('For class ks')
    ay.set_xlim(0, 1)#unc_th 0-1 for 1th to 10th index
    ay.set_ylim(0,1)
    ay.xaxis.set_ticks(np.arange(0,1.1,0.2))
    ay.tick_params(axis = 'x', labelsize = 18)
    ay.tick_params(axis = 'y', labelsize = 18)
    #ay[0].grid(False)
    
        
    #ay[0].set_xlim(0,1)    
    
    # Adding a plot in the fig2ure which will encapsulate all the subplots with ayis showing only
    fig2.add_subplot(1, 1, 1, frame_on=False)
    plt.grid(False)

    # Hiding the ayis ticks and tick labels of the bigger plot
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    
    # Adding the x-ayis and y-ayis labels for the bigger plot
    plt.xlabel('Uncertainty threshold (-)',loc='center',fontsize=20,labelpad=15)
    if mode == 'acc':
        plt.ylabel('Accuracy (-)',loc='center',fontsize=20,labelpad=15)
    elif mode == 'recall':
        plt.ylabel('Recall (-)',loc='center',fontsize=20,labelpad=15)
    elif mode == 'prec':
        plt.ylabel('Precision (-)',loc='center',fontsize=20,labelpad=15)
    
    # 凡例の表示
    lines, labels = ay.get_legend_handles_labels()
    fig2.legend(lines, labels, bbox_to_anchor=(0,0),loc='lower left', borderaxespad=8, fontsize=17)

    #plt.tight_layout()
    # プロット表示(設定の反映)
    #plt.show()
    
    plt.savefig(summary_folder+r'comp_i_{0}_phats_{1}_{2}.png'.format(mode,cond1_name,cond2_name))


if __name__ == '__main__':
    comp_i('1_LnuTnu','2_LnuTnu','acc')
    comp_i('1_LnuTnu','2_LnuTnu','recall')
    comp_i('1_LnuTnu','2_LnuTnu','prec')
    
    mix_rate_ls = [i * 0.25 for i in range(1,4)]
    for j_list in range (0,3):
        mix_rate_l = mix_rate_ls[j_list]
        comp_ii('1_LnuTwu','2_LwuTwu',mix_rate_l,'acc')
        comp_ii('1_LnuTwu_ltr','2_LwuTwu_ltr',mix_rate_l,'acc')
        comp_ii('1_LnuTwu','2_LwuTwu_iiX_em',mix_rate_l,'acc')#see notebook 20220317
        comp_ii('1_LnuTwu_ltr','2_LwuTwu_iiX_f',mix_rate_l,'acc')#see notebook 20220317
        
        comp_ii('1_LnuTwu','2_LwuTwu',mix_rate_l,'recall')
        comp_ii('1_LnuTwu_ltr','2_LwuTwu_ltr',mix_rate_l,'recall')
        comp_ii('1_LnuTwu','2_LwuTwu_iiX_em',mix_rate_l,'recall')#see notebook 20220317
        comp_ii('1_LnuTwu_ltr','2_LwuTwu_iiX_f',mix_rate_l,'recall')#see notebook 20220317
        
        comp_ii('1_LnuTwu','2_LwuTwu',mix_rate_l,'prec')
        comp_ii('1_LnuTwu_ltr','2_LwuTwu_ltr',mix_rate_l,'prec')
        comp_ii('1_LnuTwu','2_LwuTwu_iiX_em',mix_rate_l,'prec')#see notebook 20220317
        comp_ii('1_LnuTwu_ltr','2_LwuTwu_iiX_f',mix_rate_l,'prec')#see notebook 20220317
    
    comp_iii('1_LnuTwu','2_LnuTwu','acc')
    comp_iii('1_LnuTwu_ltr','2_LnuTwu_ltr','acc')
    comp_iii('1_LnuTwu','2_LnuTwu','recall')
    comp_iii('1_LnuTwu_ltr','2_LnuTwu_ltr','recall')
    comp_iii('1_LnuTwu','2_LnuTwu','prec')
    comp_iii('1_LnuTwu_ltr','2_LnuTwu_ltr','prec')
    
    
    #############Comparison of (ii)
    
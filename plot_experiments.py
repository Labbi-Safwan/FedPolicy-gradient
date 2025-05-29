import pickle

import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import sys
from matplotlib.lines import Line2D
import seaborn as sns



def load_results_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

if __name__ == '__main__':
    font = {'family' : 'serif'
         }
    R = 1000
    runs = 4
    # First experiment: speedup homogenous case
 
    number_round_first_experiment = 2000    
    H_first_experiment = 5
    epsilons_p_first =  0.0 
    M_first_experiment = [2,10, 50]
    step_first =[0.01, 0.05, 0.25]
    
    # Second experiment: speedup heteregenous case
    
    H_second_experiment = 10
    epsilons_p_second =  0.01 
    M_second_experiment = 10
    step_second = [0.01]
    
    # Third experiment: Comparaison fedQ heteregenous
    number_round_third_experiment = 6000   
    epsilons_p_third =  1
    M_third_experiment = 2
    #H_third =  [1, 10, 100]
    #step_third = [0.01, 0.001, 0.0001]
    H_third =  5
    step_third = 0.01
    
    parser = argparse.ArgumentParser(description='plotting the experiments')
    parser.add_argument("--experiment", type=int, default=1, help="between 1 and 4")
    parser.add_argument("--environment", type=int, default=0, help="0 is for the synthetic environment and 1 is for Gridword")
    parser.add_argument("--runs", type=int, default=1, help="number of runs")

    args = parser.parse_args()
    args_dict = vars(args)
    if args.environment ==1:
        environment = "gridword"
    else:
        environment = "synthetic"
        
    if args.experiment ==1:
        experiment = "experiment1"
    elif args.experiment ==2:
        experiment = "experiment2"
    elif args.experiment ==3:
        experiment = "experiment3"
    elif args.experiment ==4:
        experiment = "experiment4"
    read_folder1 = './experiments/' + 'SoftFedPG' + '/'+ environment
    read_folder2 = './experiments/' + 'RegSoftFedPG' + '/'+ environment
    read_folder3 = './experiments/' + 'BitRegSoftFedPG' +'/'+ environment
    read_folder4 = './experiments/' + 'FedQlearning' +'/'+ environment
    save_plot_folder = './plots/' + experiment
    create_folder_if_not_exists(save_plot_folder)
    markers1 = [ '+','.', ',', 'o', 'v', '^', '<', '>']
    markers2 = ['x','s', 'D', 'h', 'p', '+', 'x', '|', '_']
    colors =  sns.color_palette("colorblind")
    fig = plt.figure(figsize=(4, 3))
    if args.experiment ==1: # Speedup homogenous case
        counter_marker = -1
        legend_elements = []
        for k in range(len(M_first_experiment)):
            counter_marker+=1
            average_values_SoftFedPG = []
            average_values_RegSoftFedPG = []
            average_values_BitRegSoftFedPG = []
            smaller_vales_SoftFedPG = []
            smaller_vales_RegSoftFedPG  = []
            smaller_vales_BitRegSoftFedPG = []
            larger_vales_SoftFedPG =[]
            larger_vales_RegSoftFedPG  = []
            larger_vales_BitRegSoftFedPG = []
            std_values_SoftFedPG = []
            std_values_RegSoftFedPG = []
            std_values_BitRegSoftFedPG = []
            true_objective_SoftFedPG= np.zeros((runs, number_round_first_experiment))
            true_objective_RegSoftFedPG = np.zeros((runs, number_round_first_experiment))
            true_objective_BitRegSoftFedPG = np.zeros((runs, number_round_first_experiment))
            for run in range(runs):
                true_objective_SoftFedPG_file = read_folder1 + '/' + 'N_'+ str(M_first_experiment[k]) +',R_'+ str(number_round_first_experiment) + ',H_'+ str(H_first_experiment) + ',step_'+ str(step_first[k]) + ',ep_'+ str(epsilons_p_first)+ ',run_'+str(run) +',true_objective.pkl'
                true_objective_RegSoftFedPG_file = read_folder2 + '/' 'N_'+ str(M_first_experiment[k]) +',R_'+ str(number_round_first_experiment) + ',H_'+ str(H_first_experiment) + ',step_'+ str(step_first[k]) + ',ep_'+ str(epsilons_p_first)+ ',run_'+str(run) +',true_objective.pkl'
                true_objective_BitRegSoftFedPG_file = read_folder3 + '/' 'N_'+ str(M_first_experiment[k]) +',R_'+ str(number_round_first_experiment) + ',H_'+ str(H_first_experiment) + ',step_'+ str(step_first[k]) + ',ep_'+ str(epsilons_p_first)+ ',run_'+str(run) +',true_objective.pkl'
                true_objective_SoftFedPG[run,:] = load_results_from_pickle(true_objective_SoftFedPG_file)
                true_objective_RegSoftFedPG[run,:] = load_results_from_pickle(true_objective_RegSoftFedPG_file)
                true_objective_BitRegSoftFedPG[run,:] = load_results_from_pickle(true_objective_BitRegSoftFedPG_file)
            smaller_vales_SoftFedPG.append(np.min(true_objective_SoftFedPG, axis=0))
            smaller_vales_RegSoftFedPG.append(np.min(true_objective_RegSoftFedPG, axis=0))
            smaller_vales_BitRegSoftFedPG.append(np.min(true_objective_BitRegSoftFedPG, axis=0))
            larger_vales_SoftFedPG.append(np.max(true_objective_SoftFedPG, axis=0))
            larger_vales_RegSoftFedPG.append(np.max(true_objective_RegSoftFedPG, axis=0))
            larger_vales_BitRegSoftFedPG.append(np.max(true_objective_BitRegSoftFedPG, axis=0))
            average_vales_SoftFedPG = np.mean(true_objective_SoftFedPG, axis=0)
            average_vales_RegSoftFedPG = np.mean(true_objective_RegSoftFedPG, axis=0)
            average_vales_BitRegSoftFedPG = np.mean(true_objective_BitRegSoftFedPG, axis=0)
            std_vales_SoftFedPG = np.std(true_objective_SoftFedPG, axis=0)
            std_vales_RegSoftFedPG = np.std(true_objective_RegSoftFedPG, axis=0)
            std_vales_BitRegSoftFedPG = np.std(true_objective_BitRegSoftFedPG, axis=0)
            legend_elements.append(Line2D([0],[0], color = colors[counter_marker],label=r"$M = {:.3g}$".format(M_first_experiment[k])))
            plt.plot(average_vales_SoftFedPG, label = r"Soft-FedPG$".format(M_first_experiment[k]), marker = 'x', color = colors[counter_marker],markevery=(0.1, 0.35),
                markersize = 8, linewidth = 0.8)
            plt.plot(average_vales_RegSoftFedPG, label = r"Reg-Soft-FedPG $\epsilon_p = {:.3g}$".format(M_first_experiment[k]), marker = 'o',color = colors[counter_marker],markevery=(0.1, 0.33),
                markersize = 8, linewidth = 0.8)
            plt.plot(average_vales_BitRegSoftFedPG, label = r"Bit-Reg-Soft-FedPG $\epsilon_p = {:.3g}$".format(M_first_experiment[k]), marker = 'v',color = colors[counter_marker],markevery=(0.1, 0.3),
                markersize = 8, linewidth = 0.8)
            plt.fill_between(range(number_round_first_experiment), np.array(average_vales_SoftFedPG) - np.array(std_vales_SoftFedPG), np.array(average_vales_SoftFedPG) + np.array(std_vales_SoftFedPG),color = colors[counter_marker] ,alpha=0.3)
            plt.fill_between(range(number_round_first_experiment), np.array(average_vales_RegSoftFedPG) - np.array(std_vales_RegSoftFedPG), np.array(average_vales_RegSoftFedPG) + np.array(std_vales_RegSoftFedPG), color = colors[counter_marker] , alpha=0.3)
            plt.fill_between(range(number_round_first_experiment), np.array(average_vales_BitRegSoftFedPG) - np.array(std_vales_BitRegSoftFedPG), np.array(average_vales_BitRegSoftFedPG) + np.array(std_vales_BitRegSoftFedPG), color = colors[counter_marker] , alpha=0.3)
            print(average_vales_SoftFedPG)
            print(average_vales_RegSoftFedPG)
        fontsize = 17
        #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.xlabel(r" Round R", fontsize=fontsize, **font)
        plt.ylabel(r"Objective $J(\theta_r)$", fontsize=fontsize, **font)
        #if args.environment ==1:
        #    plt.title("GridWorld environment", fontsize=fontsize)
        #else:
        #    plt.title("Synthetic environment", fontsize=fontsize)
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(fontsize=fontsize-3)
        plt.yticks(fontsize=fontsize-3)
        # Add the custom legend to the plot
        plt.legend(handles=legend_elements, fontsize=14, loc='upper right')
        plt.grid(linestyle = '--', alpha = 0.6) 
        plt.tight_layout()
        if args.environment ==1:
            plt.savefig(save_plot_folder +"gridword_experiment_1.pdf", bbox_inches="tight")
        else:
            plt.savefig(save_plot_folder+ "synthetic_experiment_1.pdf", bbox_inches="tight")

    elif args.experiment ==2: # Speedup heteregenous case
        counter_marker = -1
        legend_elements = []
        for M in M_first_experiment:
            counter_marker+=1
            average_values_SoftFedPG = []
            average_values_RegSoftFedPG = []
            average_values_BitRegSoftFedPG = []
            smaller_vales_SoftFedPG = []
            smaller_vales_RegSoftFedPG  = []
            smaller_vales_BitRegSoftFedPG = []
            larger_vales_SoftFedPG =[]
            larger_vales_RegSoftFedPG  = []
            larger_vales_BitRegSoftFedPG = []
            std_values_SoftFedPG = []
            std_values_RegSoftFedPG = []
            std_values_BitRegSoftFedPG = []
            true_objective_SoftFedPG= np.zeros((runs, R))
            true_objective_RegSoftFedPG = np.zeros((runs, R))
            true_objective_BitRegSoftFedPG = np.zeros((runs, R))
            for run in range(runs):
                true_objective_SoftFedPG_file = read_folder1 + '/' + 'N_'+ str(M) +',R_'+ str(number_round_first_experiment) + ',H_'+ str(H_first_experiment) + ',step_'+ str(step_first) + ',ep_'+ str(epsilons_p_first)+ ',run_'+str(run) +',true_objective.pkl'
                #true_objective_RegSoftFedPG_file = read_folder2 + '/' 'N_'+ str(M) +',R_'+ str(number_round_first_experiment) + ',H_'+ str(H_first_experiment) + ',step_'+ str(step_first) + ',ep_'+ str(epsilons_p_first)+ ',run_'+str(run) +',true_objective.pkl'
                #true_objective_BitRegSoftFedPG_file = read_folder3 + '/' 'N_'+ str(M) +',R_'+ str(number_round_first_experiment) + ',H_'+ str(H_first_experiment) + ',step_'+ str(step_first) + ',ep_'+ str(epsilons_p_first)+ ',run_'+str(run) +',true_objective.pkl'
                true_objective_RegSoftFedPG_file = true_objective_SoftFedPG_file
                true_objective_BitRegSoftFedPG_file = true_objective_SoftFedPG_file
                true_objective_SoftFedPG[run,:] = load_results_from_pickle(true_objective_SoftFedPG_file)
                true_objective_RegSoftFedPG[run,:] = load_results_from_pickle(true_objective_RegSoftFedPG_file)
                true_objective_BitRegSoftFedPG[run,:] = load_results_from_pickle(true_objective_BitRegSoftFedPG_file)
            smaller_vales_SoftFedPG.append(np.min(true_objective_SoftFedPG, axis=0))
            smaller_vales_RegSoftFedPG.append(np.min(true_objective_RegSoftFedPG, axis=0))
            smaller_vales_BitRegSoftFedPG.append(np.min(true_objective_BitRegSoftFedPG, axis=0))
            larger_vales_SoftFedPG.append(np.max(true_objective_SoftFedPG, axis=0))
            larger_vales_RegSoftFedPG.append(np.max(true_objective_RegSoftFedPG, axis=0))
            larger_vales_BitRegSoftFedPG.append(np.max(true_objective_BitRegSoftFedPG, axis=0))
            average_vales_SoftFedPG = np.mean(true_objective_SoftFedPG, axis=0)
            average_vales_RegSoftFedPG = np.mean(true_objective_RegSoftFedPG, axis=0)
            average_vales_BitRegSoftFedPG = np.mean(true_objective_BitRegSoftFedPG, axis=0)
            std_vales_SoftFedPG = np.std(true_objective_SoftFedPG, axis=0)
            std_vales_RegSoftFedPG = np.std(true_objective_RegSoftFedPG, axis=0)
            std_vales_BitRegSoftFedPG = np.std(true_objective_BitRegSoftFedPG, axis=0)
            legend_elements.append(Line2D([0],[0], color = colors[counter_marker],label=r"$\epsilon_p = {:.3g}$".format(M)))
            plt.plot(average_vales_SoftFedPG, label = r"Soft-FedPG$".format(M), marker = 'x', color = colors[counter_marker],
                markersize = 8, linewidth = 0.8)
            plt.plot(average_vales_RegSoftFedPG, label = r"Reg-Soft-FedPG $\epsilon_p = {:.3g}$".format(M), marker = 'o',color = colors[counter_marker],
                markersize = 8, linewidth = 0.8)
            plt.plot(average_vales_BitRegSoftFedPG, label = r"Bit-Reg-Soft-FedPG $\epsilon_p = {:.3g}$".format(M), marker = 'v',color = colors[counter_marker],
                markersize = 8, linewidth = 0.8)
            print(average_vales_SoftFedPG)
            print(average_vales_RegSoftFedPG)
            
            plt.fill_between(np.array(average_vales_SoftFedPG), np.maximum(np.array(average_vales_SoftFedPG) - np.array(std_vales_SoftFedPG),np.array(smaller_vales_SoftFedPG)), np.minimum(np.array(average_vales_SoftFedPG) + np.array(std_vales_SoftFedPG),np.array(larger_vales_SoftFedPG)),color = colors[counter_marker] ,alpha=0.3)
            plt.fill_between(np.array(average_vales_RegSoftFedPG), np.array(average_vales_RegSoftFedPG) - np.array(std_vales_RegSoftFedPG), np.array(average_vales_RegSoftFedPG) + np.array(std_vales_RegSoftFedPG), color = colors[counter_marker] , alpha=0.3)
            plt.fill_between(np.array(average_vales_BitRegSoftFedPG), np.array(average_vales_BitRegSoftFedPG) - np.array(std_vales_BitRegSoftFedPG), np.array(average_vales_BitRegSoftFedPG) + np.array(std_vales_BitRegSoftFedPG), color = colors[counter_marker] , alpha=0.3)

        fontsize = 17
        #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.xlabel(r"Communication Round R", fontsize=fontsize, **font)
        plt.ylabel(r"Objective $J(\theta_r)$", fontsize=fontsize, **font)
        #if args.environment ==1:
        #    plt.title("GridWorld environment", fontsize=fontsize)
        #else:
        #    plt.title("Synthetic environment", fontsize=fontsize)
        plt.xscale('log')
        plt.yscale('log')
        plt.ticklabel_format(style='plain', axis='y')
        plt.xticks(fontsize=fontsize-3)
        plt.yticks(fontsize=fontsize-3)
        # Add the custom legend to the plot
        if args.environment ==1:
            plt.legend(handles=legend_elements, fontsize=14, loc='upper right')
        plt.grid(linestyle = '--', alpha = 0.6) 
        plt.tight_layout()
        if args.environment ==1:
            plt.savefig(save_plot_folder +"gridword_experiment_2.pdf", bbox_inches="tight")
        else:
            plt.savefig(save_plot_folder+ "synthetic_experiment_2.pdf", bbox_inches="tight")
            
    elif args.experiment ==3: # comparaison fed q learning heteregenous
        counter_marker = -1
        legend_elements = []
        counter_marker+=1
        average_values_SoftFedPG = []
        average_values_RegSoftFedPG = []
        average_values_BitRegSoftFedPG = []
        average_values_FedQ = []
        smaller_vales_SoftFedPG = []
        smaller_vales_RegSoftFedPG  = []
        smaller_vales_BitRegSoftFedPG = []
        smaller_vales_FedQ= []
        larger_vales_SoftFedPG =[]
        larger_vales_RegSoftFedPG  = []
        larger_vales_BitRegSoftFedPG = []
        larger_vales_FedQ = []
        std_values_SoftFedPG = []
        std_values_RegSoftFedPG = []
        std_values_BitRegSoftFedPG = []
        std_values_FedQ = []
        true_objective_SoftFedPG= np.zeros((runs, number_round_third_experiment))
        true_objective_RegSoftFedPG = np.zeros((runs, number_round_third_experiment))
        true_objective_BitRegSoftFedPG = np.zeros((runs, number_round_third_experiment))
        true_objective_FedQ = np.zeros((runs, number_round_third_experiment))
        for run in range(runs):
            true_objective_SoftFedPG_file = read_folder1 + '/' + 'N_'+ str(M_third_experiment) +',R_'+ str(number_round_third_experiment) + ',H_'+ str(H_third) + ',step_'+ str(step_third) + ',ep_'+ str(epsilons_p_third)+ ',run_'+str(run) +',true_objective.pkl'
            true_objective_RegSoftFedPG_file = read_folder2 + '/' 'N_'+ str(M_third_experiment) +',R_'+ str(number_round_third_experiment) + ',H_'+ str(H_third) + ',step_'+ str(step_third) + ',ep_'+ str(epsilons_p_third)+ ',run_'+str(run) +',true_objective.pkl'
            true_objective_BitRegSoftFedPG_file = read_folder3 + '/' 'N_'+ str(M_third_experiment) +',R_'+ str(number_round_third_experiment) + ',H_'+ str(H_third) + ',step_'+ str(step_third) + ',ep_'+ str(epsilons_p_third)+ ',run_'+str(run) +',true_objective.pkl'
            true_objective_FedQ_file = read_folder4 + '/' 'N_'+ str(M_third_experiment) +',R_'+ str(number_round_third_experiment) + ',H_'+ str(H_third) + ',step_'+ str(step_third) + ',ep_'+ str(epsilons_p_third)+ ',run_'+str(run) +',true_objective.pkl'
            true_objective_SoftFedPG[run,:] = load_results_from_pickle(true_objective_SoftFedPG_file)
            true_objective_RegSoftFedPG[run,:] = load_results_from_pickle(true_objective_RegSoftFedPG_file)
            true_objective_BitRegSoftFedPG[run,:] = load_results_from_pickle(true_objective_BitRegSoftFedPG_file)
            true_objective_FedQ[run,:] = load_results_from_pickle(true_objective_FedQ_file)
        smaller_vales_SoftFedPG.append(np.min(true_objective_SoftFedPG, axis=0))
        smaller_vales_RegSoftFedPG.append(np.min(true_objective_RegSoftFedPG, axis=0))
        smaller_vales_BitRegSoftFedPG.append(np.min(true_objective_BitRegSoftFedPG, axis=0))
        smaller_vales_FedQ.append(np.min(true_objective_FedQ, axis=0))
        larger_vales_SoftFedPG.append(np.max(true_objective_SoftFedPG, axis=0))
        larger_vales_RegSoftFedPG.append(np.max(true_objective_RegSoftFedPG, axis=0))
        larger_vales_BitRegSoftFedPG.append(np.max(true_objective_BitRegSoftFedPG, axis=0))
        larger_vales_FedQ.append(np.max(true_objective_FedQ, axis=0))
        average_vales_SoftFedPG = np.mean(true_objective_SoftFedPG, axis=0)
        average_vales_RegSoftFedPG = np.mean(true_objective_RegSoftFedPG, axis=0)
        average_vales_BitRegSoftFedPG = np.mean(true_objective_BitRegSoftFedPG, axis=0)
        average_vales_FedQ = np.mean(true_objective_FedQ, axis=0)
        std_vales_SoftFedPG = np.std(true_objective_SoftFedPG, axis=0)
        std_vales_RegSoftFedPG = np.std(true_objective_RegSoftFedPG, axis=0)
        std_vales_BitRegSoftFedPG = np.std(true_objective_BitRegSoftFedPG, axis=0)
        std_vales_FedQ = np.std(true_objective_FedQ, axis=0)
        legend_elements.append(Line2D([0],[0], color = colors[counter_marker],label=r"$M = {:.3g}$".format(M_third_experiment)))
        plt.plot(average_vales_SoftFedPG, label = r"Soft-FedPG$".format(M_third_experiment), marker = 'x', color = colors[counter_marker],markevery=(0.1, 0.35),
            markersize = 8, linewidth = 0.8)
        plt.plot(average_vales_RegSoftFedPG, label = r"Reg-Soft-FedPG $\epsilon_p = {:.3g}$".format(M_third_experiment), marker = 'o',color = colors[counter_marker],markevery=(0.1, 0.33),
            markersize = 8, linewidth = 0.8)
        plt.plot(average_vales_BitRegSoftFedPG, label = r"Bit-Reg-Soft-FedPG $\epsilon_p = {:.3g}$".format(M_third_experiment), marker = 'v',color = colors[counter_marker],markevery=(0.1, 0.3),
            markersize = 8, linewidth = 0.8)
        plt.plot(average_vales_FedQ, label = r"FedQ-learning $\epsilon_p = {:.3g}$".format(M_third_experiment), marker = 's',linestyle = '--',color = colors[7],markevery=1400,
            markersize = 8, linewidth = 2)
        plt.fill_between(range(number_round_third_experiment), np.array(average_vales_SoftFedPG) - np.array(std_vales_SoftFedPG), np.array(average_vales_SoftFedPG) + np.array(std_vales_SoftFedPG),color = colors[counter_marker] ,alpha=0.3)
        plt.fill_between(range(number_round_third_experiment), np.array(average_vales_RegSoftFedPG) - np.array(std_vales_RegSoftFedPG), np.array(average_vales_RegSoftFedPG) + np.array(std_vales_RegSoftFedPG), color = colors[counter_marker] , alpha=0.3)
        plt.fill_between(range(number_round_third_experiment), np.array(average_vales_BitRegSoftFedPG) - np.array(std_vales_BitRegSoftFedPG), np.array(average_vales_BitRegSoftFedPG) + np.array(std_vales_BitRegSoftFedPG), color = colors[counter_marker] , alpha=0.3)
        plt.fill_between(range(number_round_third_experiment), np.array(average_vales_FedQ) - np.array(std_vales_FedQ), np.array(average_vales_FedQ) + np.array(std_vales_FedQ), color = colors[7] , alpha=0.3)
        print(true_objective_FedQ)
        fontsize = 17
        #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.xlabel(r" Round R", fontsize=fontsize, **font)
        plt.ylabel(r"Objective $J(\theta_r)$", fontsize=fontsize, **font)
        #if args.environment ==1:
        #    plt.title("GridWorld environment", fontsize=fontsize)
        #else:
        #    plt.title("Synthetic environment", fontsize=fontsize)
        #plt.xscale('log')
        #plt.yscale('log')
        plt.xticks(fontsize=fontsize-3)
        plt.yticks(fontsize=fontsize-3)
        # Add the custom legend to the plot
        #plt.legend(handles=legend_elements, fontsize=14, loc='upper right')
        plt.grid(linestyle = '--', alpha = 0.6) 
        plt.tight_layout()
        if args.environment ==1:
            plt.savefig(save_plot_folder +"gridword_experiment_3.pdf", bbox_inches="tight")
        else:
            plt.savefig(save_plot_folder+ "synthetic_experiment_3.pdf", bbox_inches="tight")

import numpy as np
from  synthetic_env.Env import FiniteStateFiniteActionMDP, Challengin_MDP , Challengin_Gridword
import pickle
import matplotlib.pyplot as plt
import os
import argparse
from gridword_env.gridworld import GridWorld
import concurrent.futures
from algorithms import SoftfedPG, RegSoftfedPG, BitRegSoftfedPG, FedQ

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
            
def save_results_to_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def run_softfedpg(argument):
    full_path = argument[0]
    run = argument[1]
    envs = argument[2]
    number_rounds = argument[3]
    number_local_steps = argument[4]
    step_size = argument[5]
    ep  = argument[6]
    dict  = argument[7]
    N = len(envs)
    true_objective_values_file =  full_path +  '/' + 'N_'+ str(N) +',R_'+ str(number_rounds) + ',H_'+ str(number_local_steps) + ',step_'+ str(step_size) + ',ep_'+ str(ep)+ ',run_'+str(run) +',true_objective.pkl'
    np.random.seed(run)
    federation_SoftfedPG = SoftfedPG(envs, number_rounds, number_local_steps, step_size,  **dict)
    true_objective_values = federation_SoftfedPG.train()
    save_results_to_pickle(true_objective_values_file, true_objective_values)
    return 

def run_regsoftfedpg(argument):
    full_path = argument[0]
    run = argument[1]
    envs = argument[2]
    number_rounds = argument[3]
    number_local_steps = argument[4]
    step_size = argument[5]
    ep  = argument[6]
    dict  = argument[7]
    N = len(envs)
    true_objective_values_file =  full_path +  '/' + 'N_'+ str(N) +',R_'+ str(number_rounds) + ',H_'+ str(number_local_steps) + ',step_'+ str(step_size) + ',ep_'+ str(ep)+ ',run_'+str(run) +',true_objective.pkl'
    np.random.seed(run)
    federation_RegSoftfedPG = RegSoftfedPG(envs, number_rounds, number_local_steps, step_size,  **dict)
    true_objective_values = federation_RegSoftfedPG.train()
    save_results_to_pickle(true_objective_values_file, true_objective_values)
    return 

def run_bitregsoftfedpg(argument):
    full_path = argument[0]
    run = argument[1]
    envs = argument[2]
    number_rounds = argument[3]
    number_local_steps = argument[4]
    step_size = argument[5]
    ep  = argument[6]
    dict  = argument[7]
    N = len(envs)
    true_objective_values_file =  full_path +  '/' + 'N_'+ str(N) +',R_'+ str(number_rounds) + ',H_'+ str(number_local_steps) + ',step_'+ str(step_size) + ',ep_'+ str(ep)+ ',run_'+str(run) +',true_objective.pkl'
    np.random.seed(run)
    federation_BiRegSoftfedPG = BitRegSoftfedPG(envs, number_rounds, number_local_steps, step_size,  **dict)
    true_objective_values = federation_BiRegSoftfedPG.train()
    save_results_to_pickle(true_objective_values_file, true_objective_values)
    return 

def run_fedqlearning(argument):
    full_path = argument[0]
    run = argument[1]
    envs = argument[2]
    number_rounds = argument[3]
    number_local_steps = argument[4]
    step_size = argument[5]
    ep  = argument[6]
    dict  = argument[7]
    N = len(envs)
    true_objective_values_file =  full_path +  '/' + 'N_'+ str(N) +',R_'+ str(number_rounds) + ',H_'+ str(number_local_steps) + ',step_'+ str(step_size) + ',ep_'+ str(ep)+ ',run_'+str(run) +',true_objective.pkl'
    np.random.seed(run)
    federation_fedQ = FedQ(envs, number_rounds, number_local_steps,  **dict)
    true_objective_values = federation_fedQ.train()
    save_results_to_pickle(true_objective_values_file, true_objective_values)
    return 

if __name__ == '__main__':
    epsilons_p =  [1]
    #step = [0.1, 0.1, 0.1,0.1, 0.1, 0.1,0.1, 0.1, 0.1,0.1, 0.1, 0.1,0.1, 0.1, 0.1]
    #H =  [1, 1, 1, 1, 10, 10, 10, 10, 50, 50,50,50, 100, 100, 100, 100]
    #N = [2,10, 50, 100, 2,10, 50, 100, 2,10, 50, 100, 2,10, 50, 100]
    #R = [2000, 2000, 2000,2000, 2000, 2000,2000, 2000, 2000,2000, 2000, 2000,2000, 2000, 2000,2000]
    H =  [5]
    N = [2]
    step = [0.01]
    R = [6000, 2000, 2000, 2000]
    print(type(len(H)))
    nb_experiments = np.min([len(H), len(step), len(R)])
    parser = argparse.ArgumentParser(description='launching the experiment')
    parser.add_argument("--alg", type=int, default=0, help="0 is for SoftFedPG, 1 is for RegSoftFedPG, 2 for BitRegSoftFedPG, and 3 for FedQlearning ")
    parser.add_argument("--environment", type=int, default=0, help="0 is for the synthetic environment and 1 is for Gridword")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--temperature", type=float, default=0.05, help="temperature")
    parser.add_argument("--runs", type=int, default=4, help="number of runs")
    #parser.add_argument("--len_truncation", type=int, default=20, help="lenght  of the truncation T")
    parser.add_argument("--len_truncation", type=int, default=20, help="lenght  of the truncation T")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size per iteration")
    parser.add_argument("--verbose", type=bool, default=True, help="verbose")

    args = parser.parse_args()
    args_dict = vars(args)
    runs = args.runs
    if args.alg == 0:
        algorithm = "SoftFedPG"
    elif args.alg ==1:
        algorithm = "RegSoftFedPG"
    elif args.alg ==2:
        algorithm = "BitRegSoftFedPG"
    else:
        algorithm = "FedQlearning"
    if args.environment ==1:
        environment = "gridword"
    else:
        environment = "synthetic"
        
    parent_directory = './experiments/' + algorithm  +'/'+ environment
    create_folder_if_not_exists(parent_directory)
    np.random.seed(0)
    if args.environment ==0:
        for i, ep in enumerate(epsilons_p):
            if ep == 1:
                env_type_1 = Challengin_MDP(k=1)
                env_type_2 = Challengin_MDP(k=2)
                for k in range(nb_experiments):
                    full_path = parent_directory
                    seeds = [k for k in range(args.runs)]
                    envs_1 = [env_type_1  for _ in range(N[k]//2)]
                    envs_2 = [env_type_2 for _ in range(N[k]//2)]
                    envs = envs_1 + envs_2
                    with concurrent.futures.ProcessPoolExecutor(max_workers=runs) as executor:
                        if args.alg ==0:
                            arguments = [[full_path, seed, envs, R[k], H[k], step[k],ep, args_dict] for seed in seeds]
                            #run_softfedpg(arguments[0])
                            results = list(executor.map(run_softfedpg, arguments))   
                        elif args.alg ==1:
                            arguments = [[full_path, seed, envs, R[k], H[k], step[k], ep, args_dict] for seed in seeds]
                            results = list(executor.map(run_regsoftfedpg, arguments))
                        elif args.alg ==2:
                            arguments = [[full_path, seed, envs, R[k], H[k], step[k], ep, args_dict] for seed in seeds]
                            results = list(executor.map(run_bitregsoftfedpg, arguments))
                            #run_bitregsoftfedpg(arguments[0])
                        elif args.alg ==3:
                            arguments = [[full_path, seed, envs, R[k], H[k], step[k], ep, args_dict] for seed in seeds]
                            results = list(executor.map(run_fedqlearning, arguments))
                            #run_fedqlearning(arguments[0])
                        else:
                            print("The algorithm shoud be either 0, 1, 2 or 3: 0 is for softfedpg, 1 is for regsoftfedpg, 2 for bitregsoftfedpg and 3 for fed q learning")
            else:
                S, A = 5, 4
                env = FiniteStateFiniteActionMDP(S=S, A=A)
                common_P = env.get_P()
                common_r = env.get_r()
                env_type_1 = FiniteStateFiniteActionMDP(S=S, A=A, epsilon_p=ep, common_transition=common_P, common_reward=common_r)
                env_type_2 = FiniteStateFiniteActionMDP(S=S, A=A, epsilon_p=ep, common_transition=common_P, common_reward=common_r)
                #env_type_1 = FiniteStateFiniteActionMDP(S=S, A=A)
                #env_type_2 = FiniteStateFiniteActionMDP(S=S, A=A)
                for k in range(nb_experiments):
                    full_path = parent_directory
                    seeds = [k for k in range(args.runs)]
                    envs_1 = [env_type_1  for _ in range(N[k]//2)]
                    envs_2 = [env_type_2 for _ in range(N[k]//2)]
                    envs = envs_1 + envs_2
                    with concurrent.futures.ProcessPoolExecutor(max_workers=runs) as executor:
                        if args.alg ==0:
                            arguments = [[full_path, seed, envs, R[k], H[k], step[k],ep, args_dict] for seed in seeds]
                            #run_softfedpg(arguments[0])
                            results = list(executor.map(run_softfedpg, arguments))   
                        elif args.alg ==1:
                            arguments = [[full_path, seed, envs, R[k], H[k], step[k], ep, args_dict] for seed in seeds]
                            results = list(executor.map(run_regsoftfedpg, arguments))
                        elif args.alg ==2:
                            arguments = [[full_path, seed, envs, R[k], H[k], step[k], ep, args_dict] for seed in seeds]
                            results = list(executor.map(run_bitregsoftfedpg, arguments))
                            #run_bitregsoftfedpg(arguments[0])
                        elif args.alg ==3:
                            arguments = [[full_path, seed, envs, R[k], H[k], step[k], ep, args_dict] for seed in seeds]
                            results = list(executor.map(run_fedqlearning, arguments))
                        else:
                            print("The algorithm shoud be either 0, 1, 2 or 3: 0 is for softfedpg, 1 is for regsoftfedpg, 2 for bitregsoftfedpg and 3 for fed q learning")
    elif args.environment ==1:
        for i, ep in enumerate(epsilons_p):
            if ep ==1:
                env_type_1 = Challengin_Gridword(k=1)
                env_type_2 = Challengin_Gridword(k=2)
                for k in range(nb_experiments):
                    full_path = parent_directory
                    seeds = [k for k in range(args.runs)] 
                    envs_1 = [env_type_1  for _ in range(N[k]//2)]
                    envs_2 = [env_type_2 for _ in range(N[k]//2)]
                    envs = envs_1 + envs_2
                    with concurrent.futures.ProcessPoolExecutor(max_workers=runs) as executor:
                        if args.alg ==0:
                            arguments = [[full_path, seed, envs, R[k], H[k], step[k], ep, args_dict] for seed in seeds]
                            results = list(executor.map(run_softfedpg, arguments))   
                        elif args.alg ==1:
                            arguments = [[full_path, seed, envs, R[k], H[k], step[k], ep, args_dict] for seed in seeds]
                            results = list(executor.map(run_regsoftfedpg, arguments)) 
                        elif args.alg ==2:
                            arguments = [[full_path, seed, envs, R[k], H[k], step[k],ep,  args_dict] for seed in seeds]
                            results = list(executor.map(run_bitregsoftfedpg, arguments))
                        elif args.alg ==3:
                            arguments = [[full_path, seed, envs, R[k], H[k], step[k], ep, args_dict] for seed in seeds]
                            results = list(executor.map(run_fedqlearning, arguments))
                        else:
                            print("The algorithm shoud be either 0, 1, 2, 3: 0 is for softfedpg, 1 is for regsoftfedpg, 2 for bitregsoftfedpg and 3 for feq q learning") 
            else:
                env = GridWorld(3, 3, walls=((1, 1),(1,1)), success_probability=0.8)
                common_P = env.get_P() # get the common transition kernel
                common_r = env.get_r() # get the common reward function
                env_type_1 = GridWorld(3, 3, walls=((1, 1),(1,1)), success_probability=1.0, common=common_P, epsilon_p=ep)
                env_type_2 = GridWorld(3, 3, walls=((1, 1),(1,1)), success_probability=1.0, common=common_P, epsilon_p=ep)
                for k in range(nb_experiments):
                    full_path = parent_directory
                    seeds = [k for k in range(args.runs)] 
                    envs_1 = [env_type_1  for _ in range(N[k]//2)]
                    envs_2 = [env_type_2 for _ in range(N[k]//2)]
                    envs = envs_1 + envs_2
                    with concurrent.futures.ProcessPoolExecutor(max_workers=runs) as executor:
                        if args.alg ==0:
                            arguments = [[full_path, seed, envs, R[k], H[k], step[k], ep, args_dict] for seed in seeds]
                            results = list(executor.map(run_softfedpg, arguments))   
                        elif args.alg ==1:
                            arguments = [[full_path, seed, envs, R[k], H[k], step[k], ep, args_dict] for seed in seeds]
                            results = list(executor.map(run_regsoftfedpg, arguments)) 
                        elif args.alg ==2:
                            arguments = [[full_path, seed, envs, R[k], H[k], step[k],ep,  args_dict] for seed in seeds]
                            results = list(executor.map(run_bitregsoftfedpg, arguments))
                        elif args.alg ==3:
                            arguments = [[full_path, seed, envs, R[k], H[k], step[k], ep, args_dict] for seed in seeds]
                            results = list(executor.map(run_fedqlearning, arguments))
                        else:
                            print("The algorithm shoud be either 0, 1, 2, 3: 0 is for softfedpg, 1 is for regsoftfedpg, 2 for bitregsoftfedpg and 3 for feq q learning") 
                
    else:
        print("The environnement shoud be either 0 or 1: 0 is for the synthetic environment and 1 is for Gridword")
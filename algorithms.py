import numpy as np
import matplotlib.pyplot as plt
import math


import argparse
import os

class SoftfedPG:
    def __init__(self, envs , number_rounds, number_local_steps, step_size,  **kwargs):
        self.kwargs = kwargs
        # get the environments 
        self.envs = envs
        self.S = envs[0].S
        self.A = envs[0].A
        self.gamma = kwargs.get('discount')
        # get number of rounds, number of local steps and nthe lenght of the truncation
        self.R = number_rounds
        self.T = kwargs.get('len_truncation')
        self.step = step_size
        self.H  = number_local_steps
        self.environnement = kwargs.get('environment')
        self.B = kwargs.get('batch_size')
        # set number of agents
        self.M = len(envs)
        self.N = len(envs)
        self.verbose = kwargs.get('verbose')
        self.thetas = np.zeros((self.S, self.A, self.M))  # shape: (S, A, M)
        self.init_dist = np.ones(self.S)/self.S


    def softmax_policy(self, theta, state):
        """Compute action probabilities from theta for a given state."""
        logits = theta[state]
        exp_logits = np.exp(logits - np.max(logits))  
        probs = exp_logits / np.sum(exp_logits)
        return probs
    
    def bit_softmax_policy(self, theta, state):
        """Compute action probabilities from theta for a given state."""
        logits = theta[state]
        exp_logits = np.exp(logits - np.max(logits))  
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def sample_action(self, probs):
        """Sample an action according to given action probabilities."""
        return np.random.choice(len(probs), p=probs)
    
    def compute_grad_log_pi(self, theta, state, action):
        probs = self.softmax_policy(theta, state)  # shape (A,)
        grad = np.zeros((self.S, self.A))           # initialize (S,A) matrix
        grad[state, :] = -probs                      # for all actions at that state
        grad[state, action] += 1                     # plus 1 at (state, action)
        return grad

    def train(self):
        true_objectives = []
        local_thetas = np.zeros((self.S, self.A, self.M))
        for r in range(self.R):
            avg_return = self.compute_objective(local_thetas[:, :, 0])
            true_objectives.append(avg_return)
            if self.verbose:
                print('Round:',r)
                print(avg_return)
            for m in range(self.M):  # For each client
                #print('agent',m)
                env = self.envs[m]
                theta_m = np.copy(local_thetas[:, :, m])  # local copy

                for h in range(self.H):  # Local steps
                    state = env.reset()
                    trajectories = []
                    for _ in range(self.B):  # default batch size B=10 if not specified
                        state = env.reset()
                        trajectory = []
                        for _ in range(self.T):
                            probs = self.softmax_policy(theta_m, state)
                            action = self.sample_action(probs)
                            next_state, reward = env.step(action)
                            trajectory.append((state, action, reward))
                            state = next_state
                        trajectories.append(trajectory)
                        #print(trajectory)

                    # Now, estimate the gradient over the whole batch
                    grads = np.zeros_like(theta_m)

                    for trajectory in trajectories:
                        cumulative_grad_log_pi = np.zeros_like(theta_m)

                        for t, (s_t, a_t, r_t) in enumerate(trajectory):
                            # Accumulate gradient sum up to time t
                            grad_log_pi_t = self.compute_grad_log_pi(theta_m, s_t, a_t)
                            cumulative_grad_log_pi += grad_log_pi_t

                            # Add contribution to total gradient
                            grads += (self.gamma ** t) * cumulative_grad_log_pi * r_t

                    # Average over B trajectories
                    grads /= self.B
                    #print(grads)

                    # Update theta
                    theta_m += self.step * grads
                    #print(theta_m)

                    local_thetas[:, :, m] = theta_m  # save updated theta for this client

            # Federated Averaging: aggregate the local_thetas
            local_thetas = np.mean(local_thetas, axis=2, keepdims=True).repeat(self.M, axis=2)
            # Optional: track true objective

        return true_objectives

    def compute_mrp_transition(self, agent, policy):
        transition_kernel =  self.envs[agent].get_P()
        mrp_transition = np.sum(policy[:, :, np.newaxis] * transition_kernel, axis=1)
        return mrp_transition

    def compute_mrp_reward(self, agent, policy):
        reward = self.envs[agent].get_r()
        # Element-wise multiplication and sum along the actions axis
        mrp_reward = np.sum(policy * reward, axis=1)
        return mrp_reward

    def compute_stationnary_distribution(self,agent, policy):
        mrp_transition = self.compute_mrp_transition(agent, policy)
        stationnary_distribution = (1- self.gamma) * self.init_dist.T @ np.linalg.inv(np.eye(self.S) - self.gamma *mrp_transition)
        return stationnary_distribution

    def compute_value_function(self,agent, policy):
        mrp_transition = self.compute_mrp_transition(agent, policy)
        mrp_reward = self.compute_mrp_reward(agent, policy)
        return np.linalg.inv(np.eye(self.S) -self.gamma *mrp_transition) @ mrp_reward

    def compute_qfunction(self,agent, policy):
        reward = self.envs[agent].get_r()
        transitions = self.envs[agent].get_P()
        value_function = self.compute_value_function(agent, policy)
        expected_future_rewards = np.sum(transitions * value_function[np.newaxis, np.newaxis, :], axis=2)
        Q_function = reward + self.gamma * expected_future_rewards
        return Q_function

    def compute_policy(self,logits):
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def compute_objective(self,logits):
        policy = self.compute_policy(logits)
        #print(policy)
        objective = 0.0
        for agent in range(self.N):
            statinnary_distrubtion_agent = self.compute_stationnary_distribution(agent, policy)
            reward_mrp = self.compute_mrp_reward(agent, policy)
            objective += (1 / (1 - self.gamma))*np.dot(statinnary_distrubtion_agent,reward_mrp)
        return(objective/self.N)
    
class RegSoftfedPG:
    def __init__(self, envs , number_rounds, number_local_steps, step_size,  **kwargs):
        self.kwargs = kwargs
        # get the environments 
        self.temperature = kwargs.get('temperature')
        self.envs = envs
        self.S = envs[0].S
        self.A = envs[0].A
        self.gamma = kwargs.get('discount')
        # get number of rounds, number of local steps and nthe lenght of the truncation
        self.R = number_rounds
        self.T = kwargs.get('len_truncation')
        self.step = step_size
        self.H  = number_local_steps
        self.environnement = kwargs.get('environment')
        self.B = kwargs.get('batch_size')
        # set number of agents
        self.M = len(envs)
        self.N = len(envs)
        self.verbose = kwargs.get('verbose')
        self.thetas = np.zeros((self.S, self.A, self.M))  # shape: (S, A, M)
        self.init_dist = np.ones(self.S)/self.S


    def softmax_policy(self, theta, state):
        """Compute action probabilities from theta for a given state."""
        logits = theta[state]
        exp_logits = np.exp(logits - np.max(logits))  
        probs = exp_logits / np.sum(exp_logits)
        return probs
    
    def sample_action(self, probs):
        """Sample an action according to given action probabilities."""
        return np.random.choice(len(probs), p=probs)
    
    def compute_grad_log_pi(self, theta, state, action):
        probs = self.softmax_policy(theta, state)  # shape (A,)
        grad = np.zeros((self.S, self.A))           # initialize (S,A) matrix
        grad[state, :] = -probs                      # for all actions at that state
        grad[state, action] += 1                     # plus 1 at (state, action)
        return grad

    def train(self):
        true_objectives = []
        local_thetas = np.zeros((self.S, self.A, self.M))
        for r in range(self.R):
            for m in range(self.M):  # For each client
                env = self.envs[m]
                theta_m = np.copy(local_thetas[:, :, m])  # local copy

                for h in range(self.H):  # Local steps
                    state = env.reset()
                    trajectories = []

                    for _ in range(self.B):  # default batch size B=10 if not specified
                        state = env.reset()
                        trajectory = []
                        for _ in range(self.T):
                            probs = self.softmax_policy(theta_m, state)
                            action = self.sample_action(probs)
                            next_state, reward = env.step(action)
                            trajectory.append((state, action, reward))
                            state = next_state
                        trajectories.append(trajectory)

                    # Now, estimate the gradient over the whole batch
                    grads = np.zeros_like(theta_m)

                    for trajectory in trajectories:
                        cumulative_grad_log_pi = np.zeros_like(theta_m)

                        for t, (s_t, a_t, r_t) in enumerate(trajectory):
                            # Accumulate gradient sum up to time t
                            grad_log_pi_t = self.compute_grad_log_pi(theta_m, s_t, a_t)
                            cumulative_grad_log_pi += grad_log_pi_t

                            # Add contribution to total gradient
                            prob = self.softmax_policy(theta_m, s_t)
                            grads += (self.gamma ** t) * cumulative_grad_log_pi *[r_t - self.temperature *np.log(prob[a_t])]

                    # Average over B trajectories
                    grads /= self.B

                    # Update theta
                    theta_m += self.step * grads
                    #print(theta_m)

                    local_thetas[:, :, m] = theta_m  # save updated theta for this client

            # Federated Averaging: aggregate the local_thetas
            local_thetas = np.mean(local_thetas, axis=2, keepdims=True).repeat(self.M, axis=2)
            # Optional: track true objective
            avg_return = self.compute_objective(local_thetas[:, :, 0])
            true_objectives.append(avg_return)

            if self.verbose:
                print('Round:',r)
                print(avg_return)
        return true_objectives

    def compute_mrp_transition(self, agent, policy):
        transition_kernel =  self.envs[agent].get_P()
        mrp_transition = np.sum(policy[:, :, np.newaxis] * transition_kernel, axis=1)
        return mrp_transition

    def compute_mrp_reward(self, agent, policy):
        reward = self.envs[agent].get_r()
        # Element-wise multiplication and sum along the actions axis
        mrp_reward = np.sum(policy * reward, axis=1)
        return mrp_reward

    def compute_stationnary_distribution(self,agent, policy):
        mrp_transition = self.compute_mrp_transition(agent, policy)
        stationnary_distribution = (1- self.gamma) * self.init_dist .T @ np.linalg.inv(np.eye(self.S) - self.gamma *mrp_transition)
        return stationnary_distribution

    def compute_value_function(self,agent, policy):
        mrp_transition = self.compute_mrp_transition(agent, policy)
        mrp_reward = self.compute_mrp_reward(agent, policy)
        return np.linalg.inv(np.eye(self.S) -self.gamma *mrp_transition) @ mrp_reward

    def compute_qfunction(self,agent, policy):
        reward = self.envs[agent].get_r()
        transitions = self.envs[agent].get_P()
        value_function = self.compute_value_function(agent, policy)
        expected_future_rewards = np.sum(transitions * value_function[np.newaxis, np.newaxis, :], axis=2)
        Q_function = reward + self.gamma * expected_future_rewards
        return Q_function

    def compute_policy(self,logits):
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def compute_objective(self,logits):
        policy = self.compute_policy(logits)
        objective = 0.0
        for agent in range(self.N):
            statinnary_distrubtion_agent = self.compute_stationnary_distribution(agent, policy)
            reward_mrp = self.compute_mrp_reward(agent, policy)
            objective += (1 / (1 - self.gamma))*np.dot(statinnary_distrubtion_agent,reward_mrp)
        return(objective/self.N)
    
class BitRegSoftfedPG:
    def __init__(self, envs , number_rounds, number_local_steps, step_size,  **kwargs):
        self.kwargs = kwargs
        self.temperature = kwargs.get('temperature')

        # get the environments 
        self.envs = envs
        self.S = envs[0].S
        self.A = envs[0].A
        self.k = int(math.log2(self.A))
        self.gamma = kwargs.get('discount')
        # get number of rounds, number of local steps and nthe lenght of the truncation
        self.R = number_rounds
        self.T = kwargs.get('len_truncation')
        self.step = step_size
        self.H  = number_local_steps
        self.environnement = kwargs.get('environment')
        self.B = kwargs.get('batch_size')
        # set number of agents
        self.M = len(envs)
        self.N = len(envs)
        self.verbose = kwargs.get('verbose')
        self.thetas = np.zeros((self.S, self.A, self.M))  # shape: (S, A, M)
        self.init_dist = np.ones(self.S)/self.S

    def prefix_to_index(self, w_prefix):
        """
        Converts a prefix like [1, 0, 1] to a unique index among all possible prefixes.
        Prefixes are mapped by interpreting the bits as binary and summing previous prefix counts.
        """
        index = 0
        for i in range(len(w_prefix)):
            index += 2**i  # sum of sizes of shorter prefix lengths
        binary_value = int("".join(str(b) for b in w_prefix), 2) if w_prefix else 0
        return index + binary_value

    def pi_theta_np(self, a_w, s, theta):
        """
        Computes π_θ(a_w | s) using tabular autoregressive policy with θ as a NumPy array.
        
        Args:
            a_w: list of bits, e.g., [0, 1, 0]
            s: integer state id
            theta: np.array of shape (num_states, 2**k - 1, 2)
        
        Returns:
            float: probability π_θ(a_w | s)
        """
        prob = 1.0
        w_prefix = []
        for p in range(len(a_w)):
            idx = self.prefix_to_index(w_prefix)
            logits = theta[s, idx]  # shape (2,)
            exp_logits = np.exp(logits - np.max(logits))  # numerical stability
            probs = exp_logits / np.sum(exp_logits)
            prob *= probs[a_w[p]]
            w_prefix.append(a_w[p])
        return prob

    def compute_gradient(self, a_w, s, reward, theta, k):
        grad = np.zeros_like(theta)
        w_prefix = []
        for p in range(k):
            idx = self.prefix_to_index(w_prefix)
            logits = theta[s, idx]
            probs = np.exp(logits - np.max(logits))
            probs /= np.sum(probs)

            one_hot = np.zeros(2)
            one_hot[a_w[p]] = 1.0
            grad[s, idx] += reward * (one_hot - probs)

            w_prefix.append(a_w[p])
        return grad

    def sample_bit_action(self, s, theta):
        """Sample a bit-level action (list of bits) using the autoregressive policy."""
        action = []
        for p in range(self.k):
            idx = self.prefix_to_index(action)
            logits = theta[s, idx]
            probs = np.exp(logits - np.max(logits))
            probs /= np.sum(probs)
            bit = np.random.choice([0, 1], p=probs)
            action.append(bit)
        return action

    def bit_entropy_regularizer(self, theta, s, a_bits):
        """
        Computes h_b^θ(s, a) = ∑_{p=0}^{k-1} γ^{p/(k-1)} log π(w(a)_p | s, w(a)_{:p})
        """
        h_b = 0.0
        w_prefix = []
        for p in range(self.k):
            idx = self.prefix_to_index(w_prefix)
            logits = theta[s, idx]
            logits = logits - np.max(logits)  # for stability
            probs = np.exp(logits) / np.sum(np.exp(logits))
            prob = probs[a_bits[p]]
            #print(prob)
            h_b += self.gamma**(p / ((self.k )- 1)) * np.log(prob + 1e-8)  # add ε for numerical stability
            w_prefix.append(a_bits[p])
        return h_b

    def sample_action(self, probs):
        """Sample an action according to given action probabilities."""
        return np.random.choice(len(probs), p=probs)

    def train(self):
        true_objectives = []
        num_prefixes = 2**self.k - 1
        local_thetas = np.zeros((self.S, num_prefixes, 2, self.M))

        for r in range(self.R):
            for m in range(self.M):
                env = self.envs[m]
                theta_m = np.copy(local_thetas[:, :, :, m])

                for h in range(self.H):
                    trajectories = []
                    for _ in range(self.B):
                        state = env.reset()
                        trajectory = []
                        for _ in range(self.T):
                            action = self.sample_bit_action(state, theta_m)
                            next_state, reward = env.step(int("".join(str(b) for b in action), 2))
                            trajectory.append((state, action, reward))
                            state = next_state
                        trajectories.append(trajectory)

                    grads = np.zeros_like(theta_m)
                    for trajectory in trajectories:
                        cumulative_grad_log_pi = np.zeros_like(theta_m)
                        for t, (s_t, a_t, r_t) in enumerate(trajectory):
                            grad_log_pi_t = self.compute_gradient(a_t, s_t, 1.0, theta_m, self.k)  # r handled outside
                            cumulative_grad_log_pi += grad_log_pi_t
                            a_bits = a_t  # already a list of bits
                            h_bit = self.bit_entropy_regularizer(theta_m, s_t, a_bits)
                            grads += (self.gamma ** t) * cumulative_grad_log_pi * [r_t - self.temperature * h_bit]

                    grads /= self.B
                    theta_m += self.step * grads
                    local_thetas[:, :, :, m] = theta_m

            # Federated averaging
            local_thetas = np.mean(local_thetas, axis=3, keepdims=True).repeat(self.M, axis=3)
            avg_return = self.compute_objective(local_thetas[:, :, :, 0])
            true_objectives.append(avg_return)

            if self.verbose:
                print(f"Round {r}: Average Return = {avg_return:.3f}")

        return true_objectives


    def compute_mrp_transition(self, agent, policy):
        transition_kernel =  self.envs[agent].get_P()
        mrp_transition = np.sum(policy[:, :, np.newaxis] * transition_kernel, axis=1)
        return mrp_transition

    def compute_mrp_reward(self, agent, policy):
        reward = self.envs[agent].get_r()
        # Element-wise multiplication and sum along the actions axis
        mrp_reward = np.sum(policy * reward, axis=1)
        return mrp_reward

    def compute_stationnary_distribution(self,agent, policy):
        mrp_transition = self.compute_mrp_transition(agent, policy)
        stationnary_distribution = (1- self.gamma) * self.init_dist .T @ np.linalg.inv(np.eye(self.S) - self.gamma *mrp_transition)
        return stationnary_distribution

    def compute_value_function(self,agent, policy):
        mrp_transition = self.compute_mrp_transition(agent, policy)
        mrp_reward = self.compute_mrp_reward(agent, policy)
        return np.linalg.inv(np.eye(self.S) -self.gamma *mrp_transition) @ mrp_reward

    def compute_qfunction(self,agent, policy):
        reward = self.envs[agent].get_r()
        transitions = self.envs[agent].get_P()
        value_function = self.compute_value_function(agent, policy)
        expected_future_rewards = np.sum(transitions * value_function[np.newaxis, np.newaxis, :], axis=2)
        Q_function = reward + self.gamma * expected_future_rewards
        return Q_function

    def compute_policy_from_theta_bitwise(self, theta):
        """
        Compute π(a|s) for all actions a (in [0, 2^k - 1]) and states s, using bit-level autoregressive policy.
        Returns:
            policy: np.array of shape (S, A)
        """
        policy = np.zeros((self.S, 2**self.k))
        for s in range(self.S):
            for a in range(2**self.k):
                bits = [int(b) for b in format(a, f'0{self.k}b')]
                policy[s, a] = self.pi_theta_np(bits, s, theta)
        return policy

    def compute_objective(self, theta):
        """
        Computes the expected discounted reward using the autoregressive bit-level policy θ.
        """
        policy = self.compute_policy_from_theta_bitwise(theta)
        objective = 0.0
        for agent in range(self.N):
            stat_dist = self.compute_stationnary_distribution(agent, policy)
            reward_mrp = self.compute_mrp_reward(agent, policy)
            objective += (1 / (1 - self.gamma)) * np.dot(stat_dist, reward_mrp)
        return objective / self.N


class FedQ:
    def __init__(self, envs, number_rounds, number_local_steps,  **kwargs):
        self.envs = envs
        self.S = envs[0].S
        self.A = envs[0].A
        self.M = len(envs)  # number of clients
        self.N = self.M
        self.B = kwargs.get('batch_size')
        self.T = kwargs.get('len_truncation')
        self.R = number_rounds
        self.eta_schedule = lambda t: 0.01 / np.sqrt(t)
        self.gamma = kwargs.get('discount')
        self.H = number_local_steps
        self.verbose = kwargs.get('verbose', False)
        self.init_dist = np.ones(self.S)/self.S
        # Initialize Q-tables for all clients
        self.Q_clients = np.zeros((self.M, self.S, self.A))

    def sample_empirical_TQ(self, Q, env):
        """
        Empirical Bellman update using random sampling of (s, a) pairs with averaging.

        Parameters:
            Q (np.ndarray): Current Q-value estimate, shape (S, A)
            env: Environment with sample_specific_triplet(s, a)
            num_samples (int): Number of (s, a) pairs to sample

        Returns:
            TQ (np.ndarray): Empirical Bellman operator estimate
        """
        TQ = np.zeros_like(Q, dtype=np.float64)
        counts = np.zeros_like(Q, dtype=np.int32)

        for _ in range(self.B):
            s = np.random.choice(self.S)
            a = np.random.choice(self.A)
            next_state, reward = env.sample_specific_triplet(s, a)
            target = reward + self.gamma * np.max(Q[next_state])

            # Incremental average update
            counts[s, a] += 1
            TQ[s, a] += (target - TQ[s, a]) / counts[s, a]

        return TQ
    def train(self):
        true_objectives = []
        for r in range(1, self.R + 1):
            #eta = self.eta_schedule(r)
            eta = 0.001
            updated_Qs = []

            for i in range(self.M):
                Q_local = self.Q_clients[i].copy()
                env = self.envs[i]

                # Perform H local Q-learning steps
                for _ in range(self.H):
                    TQ = self.sample_empirical_TQ(Q_local, env)
                    Q_local = (1 - eta) * Q_local + eta * TQ

                updated_Qs.append(Q_local)

            # Federated averaging
            Q_avg = np.mean(updated_Qs, axis=0)
            #print(Q_avg)
            policy_round  = self.get_policy(Q_avg)
            self.Q_clients = np.array([Q_avg.copy() for _ in range(self.M)])
            true_value = self.compute_objective(policy_round)
            true_objectives.append(true_value)
            if self.verbose:
                print(f"Round {r}, True value: {true_value:.4f}")
        return true_objectives

    def get_policy(self, Q):
        one_hot = np.zeros_like(Q)
        one_hot[np.arange(Q.shape[0]), np.argmax(Q, axis=1)] = 1
        return one_hot  # Greedy policy
    
    def compute_mrp_transition(self, agent, policy):
        transition_kernel =  self.envs[agent].get_P()
        mrp_transition = np.sum(policy[:, :, np.newaxis] * transition_kernel, axis=1)
        return mrp_transition

    def compute_mrp_reward(self, agent, policy):
        reward = self.envs[agent].get_r()
        # Element-wise multiplication and sum along the actions axis
        mrp_reward = np.sum(policy * reward, axis=1)
        return mrp_reward

    def compute_stationnary_distribution(self,agent, policy):
        mrp_transition = self.compute_mrp_transition(agent, policy)
        stationnary_distribution = (1- self.gamma) * self.init_dist .T @ np.linalg.inv(np.eye(self.S) - self.gamma *mrp_transition)
        return stationnary_distribution

    def compute_value_function(self,agent, policy):
        mrp_transition = self.compute_mrp_transition(agent, policy)
        mrp_reward = self.compute_mrp_reward(agent, policy)
        return np.linalg.inv(np.eye(self.S) -self.gamma *mrp_transition) @ mrp_reward

    def compute_qfunction(self,agent, policy):
        reward = self.envs[agent].get_r()
        transitions = self.envs[agent].get_P()
        value_function = self.compute_value_function(agent, policy)
        expected_future_rewards = np.sum(transitions * value_function[np.newaxis, np.newaxis, :], axis=2)
        Q_function = reward + self.gamma * expected_future_rewards
        return Q_function

    def compute_objective(self,policy):
        objective = 0.0
        #print(policy)
        for agent in range(self.N):
            statinnary_distrubtion_agent = self.compute_stationnary_distribution(agent, policy)
            reward_mrp = self.compute_mrp_reward(agent, policy)
            objective += (1 / (1 - self.gamma))*np.dot(statinnary_distrubtion_agent,reward_mrp)
        return(objective/self.N)
    



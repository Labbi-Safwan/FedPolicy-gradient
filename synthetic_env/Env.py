import numpy as np

from .utils import RandomSimplexVector

class FiniteStateFiniteActionMDP(object):
    def __init__(self, S=50, A=50, epsilon_p = 0., epsilon_r = 0., common_transition=None, common_reward=None):
        super().__init__()
        self.S = S
        self.A = A
        self.common_transition = common_transition
        self.common_reward = common_reward
        self.epsilon_p = epsilon_p
        self.epsilon_r = epsilon_r

        # transition kernel
        if self.common_transition is not None:
            self.local_P = RandomSimplexVector(d = self.S, size=[self.S, self.A])
            #self.P = RandomSimplexVector(d = self.S, size=[self.H, self.S, self.A]) # transition probability shape [H S A S']
            self.P = (1 - self.epsilon_p)*self.common_transition + self.epsilon_p*self.local_P
        else: self.P = RandomSimplexVector(d = self.S, size=[self.S, self.A])
        
        if self.common_reward is not None:
            self.local_R = np.random.uniform(0., 1., size=[self.S, self.A])
            self.R = (1-self.epsilon_r)*self.common_reward + self.epsilon_r*self.local_R 
        else:
            self.R = np.random.uniform(0., 1., size=[self.S, self.A]) # reward between [0, 1],  shape [H, S, A]
        

    def get_P(self):
        return self.P
    
    def get_r(self):
        return self.R
    
    def reset(self,):
        self.t = 0
        self.state = np.random.randint(self.S)
        #self.state = 0
        return self.state

    def step(self, action):
        r = self.R[self.t, self.state, action]
        p = self.P[self.t, self.state, action]
        s = np.random.choice(self.S, 1, p=p)
        self.state = s.item()
        self.t += 1
        return self.state, r

    def save_env(self, dir='01'):
        np.save('envs/' + dir + '_transition.npy', self.P)
        np.save('envs/' + dir + '_reward.npy', self.R)
        return

    def load_env(self, dir='01'):
        self.P = np.load('envs/' + dir + '_transition.npy')
        self.R = np.load('envs/' + dir + '_reward.npy')
        return
    
    def observation_dim(self):
        return self.S
    
    def action_dim(self):
        return self.A
    
    def best_gen(self,):
        Q = np.zeros([self.H, self.S, self.A])
        V = np.zeros([self.H + 1, self.S])
        actions = np.zeros([self.H, self.S, self.A])
        for h in range(self.H - 1, -1, -1):
            for s in range(self.S):
                for a in range(self.A):
                    p = self.P[h, s, a]
                    EV = np.dot(p, V[h+1])
                    Q[h, s, a] = self.R[h, s, a] + EV
                actions[h, s, np.argmax(Q[h, s])] = 1
                V[h, s] = np.max(Q[h, s])
        return V[0], actions, Q

    def value_gen(self, actions):
        Q = np.zeros([self.H, self.S, self.A])
        V = np.zeros([self.H + 1, self.S])
        for h in range(self.H - 1, -1, -1):
            for s in range(self.S):
                for a in range(self.A):
                    p = self.P[h, s, a]
                    EV = np.dot(p, V[h+1])
                    Q[h, s, a] = self.R[h, s, a] + EV
                p = actions[h][s, :]
                V[h, s] = np.dot(p, Q[h, s])
        return V[0]

    def full_value_gen(self, actions):
        Q = np.zeros([self.H, self.S, self.A])
        V = np.zeros([self.H + 1, self.S])
        for h in range(self.H - 1, -1, -1):
            for s in range(self.S):
                for a in range(self.A):
                    p = self.P[h, s, a]
                    EV = np.dot(p, V[h+1])
                    Q[h, s, a] = self.R[h, s, a] + EV
                p = actions[h, s]
                V[h, s] = np.dot(p, Q[h, s])
        return Q, V

if __name__ == '__main__':
    Env = FiniteStateFiniteActionMDP()
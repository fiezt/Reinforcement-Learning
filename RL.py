import numpy as np


class grid_world_mdp(object):
    def __init__(self, grid_cols, grid_rows, actions, terminal_states=None, 
                 prob_func=None, reward_func=None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        self.n = grid_cols * grid_rows
        self.m = actions
        
        self.states = range(self.n)
        self.actions = range(self.m)
        
        grid = np.zeros((self.grid_rows, self.grid_cols))

        if self.m == 4:
            
            # Action of N, S, E, W.
            self.actions_to_idx = {(-1,0): 0, (1,0): 1, (0,1): 2, (0,-1): 3}            
            self.idx_to_actions = {v:k for k,v in self.actions_to_idx.items()}   
            
            self.action_names_to_idx = {'N':0, 'S':1, 'E':2, 'W':3}
            self.idx_to_action_names = {v:k for k,v in self.action_names_to_idx.items()}

            
        elif self.m == 8:
            
            # Action of N, NE, NW, S, SE, SW, E, W 
            self.actions_to_idx = {(-1,0): 0, (-1,1): 1, (-1,-1): 2, (1,0): 3, 
                              (1,1): 4, (1,-1): 5, (0,1): 6, (0,-1): 7}
            
            self.idx_to_actions = {v:k for k,v in self.actions_to_idx.items()}
            
            self.action_names_to_idx = {'N':0, 'NE':1, 'NW':2 , 'S':3, 'SE':4, 
                                        'SW':5, 'E':6, 'W':7}
            self.idx_to_action_names = {v:k for k,v in self.action_names_to_idx.items()}
            

        self.idx_to_states = {self.states[i]:(i/self.grid_rows, i%self.grid_rows) for i in range(self.n)}
        self.states_to_idx = {v:k for k,v in self.idx_to_states.items()}
        
        
        if terminal_states is None:
            self.terminal_states = []
        else:
            self.terminal_states = [0,15]
        
        self.create_prob_dist(prob_func)
        
        self.create_rewards(reward_func)
        
        self.check_valid_dist()
        
    
    def create_prob_dist(self, prob_func=None):
        """
        
        """
        
        if prob_func is None:
            self.get_prob_dist()
        else:
            self.P = prob_func(self)
        
    
    def get_prob_dist(self):
        """
        
        """
        
        self.P = np.zeros((self.n, self.m, self.n))
        
        for state in self.states: 
            for action in self.actions: 
                
                if state in self.terminal_states:
                    self.P[state, action, state] = 1
                    continue

                curr_pos = self.idx_to_states[state]

                new_pos = (curr_pos[0] + self.idx_to_actions[action][0], 
                           curr_pos[1] + self.idx_to_actions[action][1])

                if new_pos in self.states_to_idx:
                    new_state = self.states_to_idx[new_pos]
                    self.P[state, action, new_state] = 1

                else:
                    self.P[state, action, state] = 1
        
    
    def create_rewards(self, reward_func=None):
        """
        
        """
        
        if reward_func is None:
            self.get_rewards()
        else:
            self.R = reward_func(self)
        
    
    def get_rewards(self):
        """
        
        """
        
        self.R = -1*np.ones((self.n, self.m, self.n))

        for state in self.terminal_states:
            self.R[state] = 0
    
    
    def check_valid_dist(self):
        """
        
        """
        
        for state in xrange(self.n):
            for action in xrange(self.m):
                assert abs(sum(self.P[state, action, :]) - 1) < 1e-3, 'Transitions do not sum to 1'


class RL(object):
    def __init__(self, mdp):
        """
        
        """

        self.mdp = mdp
        
    
    def iterative_policy_evaluation(self, pi=None, gamma=1):
        """
        
        """
        
        # Random policy if not provided.
        if pi is None:
            pi = 1/float(self.mdp.m) * np.ones((self.mdp.n, self.mdp.m))
        
        v = np.zeros(self.mdp.n)

        while True:
            
            delta = 0

            for s in self.mdp.states:            
                temp = v[s].copy()       
                                
                v[s] = sum(pi[s, a]*sum(self.mdp.P[s, a, s_new]*(self.mdp.R[s, a, s_new] + gamma*v[s_new]) 
                           for s_new in self.mdp.states) for a in self.mdp.actions)

                delta = max(delta, abs(temp - v[s]))

            if delta < 1e-10:
                break
                
        policy = self.get_iterative_policy(v)
        
        return v, policy
        
    
    def get_iterative_policy(self, v):
        """
        
        """
        
        policy = np.zeros(self.mdp.n)
        
        action_vals = np.zeros(self.mdp.m)

        for s in self.mdp.states:
            
            for a in self.mdp.actions:
                action_vals[a] = sum(self.mdp.P[s,a,s_new]*v[s_new] for s_new in self.mdp.states)
            
            best_action = np.argmax(action_vals)
            
            policy[s] = best_action
            
        return policy
    
    
    def get_named_policy(self, policy):
        """
        
        """
        
        if len(policy.shape) == 1 or (policy.shape[0] == 1 and policy.shape[1] != 1) \
            or (policy.shape[0] != 1 and policy.shape[1] == 1):
                
            return [self.mdp.idx_to_action_names[a] for a in policy.tolist()]
        
        else:
            return [[self.mdp.idx_to_action_names[col] for col in row] for row in policy.tolist()]
            













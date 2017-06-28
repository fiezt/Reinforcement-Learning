import numpy as np
from matplotlib.cbook import MatplotlibDeprecationWarning
import warnings
import os
import string
from pylab import rc
from itertools import izip
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches


class grid_world_mdp(object):
    def __init__(self, grid_rows, grid_cols, actions, terminal_states=[0,15], 
                 prob_func=None, reward_func=None):
        """Creating the grid world MDP. Note that everything is indexed by 
        row, column, not x, y coordinates.

        :param grid_rows: Integer number of rows for which to make the grid.
        :param grid_cols: Integer number of cols for which to make the grid.
        :param actions: Integer that must be 4 or 8. These will be compass directions.
        :param terminal_states: List or None, of the terminal states as integers.
        :param prob_func: Function to create a different probability distribution if needed.
        :param reward_func: Function to create a different reward structure if needed.
        """

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        # Number of states.
        self.n = grid_cols * grid_rows

        # Number of actions.
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

        else:
            print('The number of actions must be 4 or 8: Exiting')
            return            

        self.idx_to_states = {self.states[i]:(i/self.grid_cols, i%self.grid_cols) for i in range(self.n)}
        self.states_to_idx = {v:k for k,v in self.idx_to_states.items()}
        
        if terminal_states is None:
            self.terminal_states = []
        else:
            self.terminal_states = terminal_states
        
        self.create_prob_dist(prob_func)
        
        self.create_rewards(reward_func)
        
        self.check_valid_dist()
        
    
    def create_prob_dist(self, prob_func=None):
        """Creating the probability distribution for the MDP.

        :param prob_func: Optional alternative function to create prob distribution.
        """
        
        if prob_func is None:
            self.get_prob_dist()
        else:
            self.P = prob_func(self)
        
    
    def get_prob_dist(self):
        """
        Default function to create a probability distribution for the MDP.
        This is a noiseless distribution. If the agent takes action 0, it will
        go to the desired location with probability 1, if the action is legal, 
        else if will have probability 0 of going to it and 1 of staying in its state.
        """
        
        self.P = np.zeros((self.n, self.m, self.n))
        
        for state in self.states: 
            for action in self.actions: 
                
                # If in terminal state will stay in terminal state.
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
        """Creating the reward structure for the MDP.

        :param reward_func: Optional alternative function to create reward distribution.
        """
        
        if reward_func is None:
            self.get_rewards()
        else:
            self.R = reward_func(self)
        
    
    def get_rewards(self):
        """
        The default reward structure. -1 for all actions, except 0 if 
        going to terminal state.
        """
        
        self.R = -1*np.ones((self.n, self.m, self.n))

        for state in self.terminal_states:
            self.R[state] = 0
    
    
    def check_valid_dist(self):
        """Checking the probability distribution sums to 1 for each action."""
        
        for state in xrange(self.n):
            for action in xrange(self.m):
                assert abs(sum(self.P[state, action, :]) - 1) < 1e-3, 'Transitions do not sum to 1'


class RL(object):
    def __init__(self, mdp):
        self.mdp = mdp
        
    
    def iterative_policy_evaluation(self, pi=None, gamma=1):
        """Iterative policy evaluation finds the state value function for a policy.

        :param pi: Probability distribution of actions given states.
        :param gamma: Float discount factor in (0,1]
        """
        
        # Random policy if a policy is not provided.
        if pi is None:
            self.mdp.pi = 1/float(self.mdp.m) * np.ones((self.mdp.n, self.mdp.m))
        else:
            self.mdp.pi = pi
        
        self.mdp.v = np.zeros(self.mdp.n)

        max_iter = 1000

        for iteration in xrange(max_iter):
            
            delta = 0

            for s in self.mdp.states:            
                v_temp = self.mdp.v[s].copy()       
                
                # Bellman equation to back up.
                self.mdp.v[s] = sum(self.mdp.pi[s, a] * sum(self.mdp.P[s, a, s_new] 
                                    * (self.mdp.R[s, a, s_new] + gamma*self.mdp.v[s_new]) 
                                    for s_new in self.mdp.states) for a in self.mdp.actions)

                delta = max(delta, abs(v_temp - self.mdp.v[s]))

            if delta < 1e-10:
                break
        
        self.get_iterative_policy()
        
        self.get_named_policy()

    
    def get_iterative_policy(self):
        """Given the value function find the policy for actions in states. """
        
        self.mdp.policy = np.zeros(self.mdp.n)
        
        self.mdp.action_vals = np.zeros((self.mdp.n, self.mdp.m))

        for s in self.mdp.states:
            
            for a in self.mdp.actions:
                self.mdp.action_vals[s,a] = sum(self.mdp.P[s,a,s_new] * self.mdp.v[s_new] 
                                                for s_new in self.mdp.states)
            
            # This break policies ties by taking the first index occurrence of the max.
            best_action = np.argmax(self.mdp.action_vals[s])
            
            self.mdp.policy[s] = best_action
            
    
    def get_named_policy(self):
        """Get the named action for each action in the policy."""
        
        self.mdp.named_policy = [self.mdp.idx_to_action_names[a] for a in self.mdp.policy]    


    def policy_iteration(self, gamma=1.):
        """Finds optimal policy and the value function for that policy.
        
        :param gamma: Float discounting factor for rewards.
        """
        
        # Initializing the value and policy function.
        self.mdp.v = np.zeros(self.mdp.n)
        self.mdp.policy = np.zeros(self.mdp.n, dtype=int)
        self.mdp.action_vals = np.zeros((self.mdp.n, self.mdp.m))

        max_iter = 1000
        max_eval = 100

        # Policy evaluation followed by policy improvement until convergence.
        for iteration in xrange(max_iter):

            # Policy evaluation.
            for evaluation in xrange(max_eval):
                
                delta = 0

                for s in self.mdp.states:            
                    v_temp = self.mdp.v[s].copy()       
                    
                    a = self.mdp.policy[s]
                    self.mdp.v[s] = sum(self.mdp.P[s, a, s_new] 
                                        * (self.mdp.R[s, a, s_new] + gamma*self.mdp.v[s_new]) 
                                        for s_new in self.mdp.states)

                    delta = max(delta, abs(v_temp - self.mdp.v[s]))

                if delta < 1e-10:
                    break

            # Policy improvement.
            stable = True

            for s in self.mdp.states:

                old_policy = self.mdp.policy[s].copy()

                self.mdp.action_vals[s] = [sum(self.mdp.P[s, a, s_new] 
                                               * (self.mdp.R[s, a, s_new] + gamma*self.mdp.v[s_new]) 
                                               for s_new in self.mdp.states) for a in self.mdp.actions]

                self.mdp.policy[s] = np.argmax(self.mdp.action_vals[s])

                if self.mdp.policy[s] != old_policy and stable:
                    stable = False

            # Policy convergence check.
            if stable:
                break
        
        self.get_named_policy()

        # Policy probability distribution.
        self.mdp.pi = np.zeros((self.mdp.n, self.mdp.m))
        self.mdp.pi[np.arange(self.mdp.pi.shape[0]), self.mdp.policy] = 1. 


class grid_display(object):
    def __init__(self, rl, title='Grid World', fig_path=None, fig_name=None, savefig=False):
            """Create and plot the grid with the values and policy.

            :param rl: RL object with needed information.
            :param title: Title for the figure.
            """

            # Flipping rows for plotting reasons.
            self.values = np.flipud(rl.mdp.v.reshape((rl.mdp.grid_rows, rl.mdp.grid_cols)))

            # Converting terminal state locs to the flipped orientation.
            state_locs = np.flipud(np.arange(rl.mdp.n).reshape((rl.mdp.grid_rows, rl.mdp.grid_cols)))
            state_locs = state_locs.reshape((-1)).tolist()
            self.terminal_states = [state_locs.index(state) for state in rl.mdp.terminal_states]

            self.named_policy = zip(*[iter(rl.mdp.named_policy)]*4)[::-1]
            self.named_policy = [item for row in self.named_policy for item in row]
            
            self.title = title

            if savefig:
                if fig_path is None:
                    fig_path = os.getcwd() + '/../figs'

                if fig_name is None:
                    title = title.translate(None, string.punctuation)
                    fig_name = '_'.join(title.split()) + '.png'

                full_path = os.path.join(fig_path, fig_name)
            else:
                full_path = None
                        
            self.create_grid(savefig, full_path)
            plt.show()


    def show_values(self):
        """Add in the values for each square the policy."""
        
        warnings.simplefilter('ignore', MatplotlibDeprecationWarning)
        ax = self.grid.get_axes()

        orient_dict = {'N':0, 'NE':7*np.pi/4., 'NW':np.pi/4., 'S':np.pi, 
                       'SE':5*np.pi/4., 'SW':3*np.pi/4., 'E':3*np.pi/2., 'W':np.pi/2.}
        
        dist = 0.42
        arrow_loc = {'N':(0, dist), 'NE':(dist, dist), 'NW':(-dist, dist),
                     'S':(0, -dist), 'SE':(dist, -dist), 'SW':(-dist, -dist),
                     'E':(dist, 0), 'W':(-dist, 0)}

        count = 0

        for p, value, choice in izip(self.grid.get_paths(), self.grid.get_array(), self.named_policy):
            x, y = p.vertices[:-2, :].mean(0)

            ax.text(x, y, "%.2f" % value, ha="center", va="center", color='white', 
                    fontweight='bold', fontsize='24')

            if count in self.terminal_states:
                pass
            else:            
                orient = orient_dict[choice]
                direct = arrow_loc[choice]
                
                ax.add_patch(patches.RegularPolygon((x + direct[0], y + direct[1]), 
                                                    3, .05, color='white', orientation=orient))
            
            count += 1
        
    
    def create_grid(self, savefig, full_path):
        """Create and plot the grid with the values and policy.

        :param savefig: Bool indicating whether or not to save the figure.
        :param full_path: String with combined file path and figure name.
        """
        
        cmap = mcolors.LinearSegmentedColormap.from_list('cmap', ['red', 'black', 'limegreen'])

        rc('axes', linewidth=4)

        fig, ax = plt.subplots(facecolor='black', edgecolor='white', linewidth=4)    

        self.grid = ax.pcolor(self.values, edgecolors='white', linewidths=4, cmap=cmap, 
                              vmin=self.values.min(), vmax=self.values.max())

        self.show_values()

        for spine in ax.spines.values():
            spine.set_edgecolor('white')
                
        x_axis_size = self.values.shape[1]
        y_axis_size = self.values.shape[0]

        xlabels = [str(val) for val in range(0, x_axis_size)]
        ylabels = [str(val) for val in range(y_axis_size-1, -1, -1)]

        ax.set_xticks(np.arange(0.5, len(xlabels)))
        ax.set_yticks(np.arange(0.5, len(ylabels)))

        ax.set_xticklabels(xlabels)                                                       
        ax.set_yticklabels(ylabels) 

        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(24)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(24)
            tick.label1.set_fontweight('bold')
        
        plt.title(self.title, color='white', fontsize='24', fontweight='bold')
            
        fig.set_size_inches((self.values.shape[1]*4, self.values.shape[0]*4))

        if savefig:
            plt.savefig(full_path, facecolor='black')


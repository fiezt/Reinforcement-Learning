import numpy as np
from matplotlib.cbook import MatplotlibDeprecationWarning
import warnings
import os
import string
import itertools
from pylab import rc
from itertools import izip
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib as mpl


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
        
    
    def get_prob_dist(self, noise=0.0):
        """
        Default function to create a probability distribution for the MDP.
        This is a noiseless distribution. If the agent takes action 0, it will
        go to the desired location with probability 1, if the action is legal, 
        else if will have probability 0 of going to it and 1 of staying in its state.
        Noise can be added to create probability of not taking the selected action.

        :param noise: Float in (0,1) indicating the probability of not taking 
        the intended action, this probability will be uniformly distributed 
        over all other actions.
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
                    self.P[state, action, new_state] = 1 - noise
                else:
                    self.P[state, action, state] = 1 - noise

                # Adding probability of taking an action that is not chosen.
                for noisy_action in actions:
                    if noisy_action == action:
                        continue

                    new_pos = (curr_pos[0] + self.idx_to_actions[noisy_action][0], 
                               curr_pos[1] + self.idx_to_actions[noisy_action][1])

                    if new_pos in self.states_to_idx:
                        new_state = self.states_to_idx[new_pos]
                        self.P[state, noisy_action, new_state] = noise/float(self.mdp.m - 1)
                    else:
                        self.P[state, noisy_action, state] = noise/float(self.mdp.m - 1)
        
    
    def create_rewards(self, reward_func=None):
        """Creating the reward structure for the MDP.

        :param reward_func: Optional alternative function to create reward distribution.
        """
        
        if reward_func is None:
            self.get_rewards()
        else:
            self.R = reward_func(self)
        
    
    def get_rewards(self, noise=0.0):
        """
        The default reward structure. -1 for all actions, except 0 if 
        going to terminal state. The noise parameter can be used to make the 
        rewards not uniform by adding in Gaussian noise with mean 0 and standard
        deviation of the noise to each of the state, action, next state rewards.

        :param noise: Float indicating the standard deviation for 0 mean Gaussian
        noise to add into each reward for a state, action, next state tuple.
        """
        
        noise_array = np.random.normal(0, noise, (self.n, self.m, self.m))
        self.R = -1*np.ones((self.n, self.m, self.n))

        # Adding noise selected so not all rewards are equal.
        self.R = noise_array + self.R

        for state in self.terminal_states:
            self.R[state] = 0

    
    def sample_transition(self, s, a, sigma=0.0):
        """Sample the transition probability from the defined probability distribution.

        This function samples the transition from the defined transition 
        probability distribution in the form of a multinomial distribution in 
        which the probability of each resulting state has probability equal to 
        that of the true distribution of being sampled. The observed reward of 
        this transition is sampled from the true reward distribution with 
        Gaussian noise added.

        :param s: Integer index of the current state index the agent is in.
        :param a: Integer index of the action index the agent is taking.
        :param sigma: The standard deviation of the noise to add in when sampling
        a reward for the transition.

        :return s_new: Integer index of the resulting new state for the transition.
        :return reward: Float of the sampled reward of the transition.
        """

        sample = np.random.multinomial(1, self.P[s, a]).tolist()
        s_new = sample.index(1)

        reward = np.random.normal(self.R[s, a, s_new], sigma)

        return s_new, reward
    

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
        :param gamma: Float discounting factor for the rewards in (0,1].
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
                self.mdp.action_vals[s, a] = sum(self.mdp.P[s, a, s_new] * self.mdp.v[s_new] 
                                                 for s_new in self.mdp.states)
                        
        self.mdp.policy = random_argmax(self.mdp.action_vals)
            
    
    def get_named_policy(self):
        """Get the named action for each action in the policy."""
        
        self.mdp.named_policy = [self.mdp.idx_to_action_names[a] for a in self.mdp.policy]    


    def policy_iteration(self, gamma=1., max_iter=1000, max_eval=100):
        """Finds optimal policy and the value function for that policy.
        
        :param gamma: Float discounting factor for rewards in (0,1].
        :param max_iter: Integer max number of iterations to run policy iteration.
        :param max_eval: Integer max number of evaluations to run policy evaluation.
        """
        
        # Initializing the value and policy function.
        self.mdp.v = np.zeros(self.mdp.n)
        self.mdp.policy = np.zeros(self.mdp.n, dtype=int)
        self.mdp.action_vals = np.zeros((self.mdp.n, self.mdp.m))

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

                self.mdp.policy[s] = random_argmax(self.mdp.action_vals[s])

                if self.mdp.policy[s] != old_policy and stable:
                    stable = False

            # Policy convergence check.
            if stable:
                break
        
        self.get_named_policy()

        # Policy probability distribution.
        self.mdp.pi = np.zeros((self.mdp.n, self.mdp.m))
        self.mdp.pi[np.arange(self.mdp.pi.shape[0]), self.mdp.policy] = 1. 


    def value_iteration(self, gamma=1., max_eval=1000):
        """Find the optimal value function and policy with value iteration.
        
        :param gamma: Float discounting factor for rewards in (0,1].
        :param max_eval: Integer max number of evaluations to do for value iteration.
        """

        # Initializing the value and policy function.
        self.mdp.v = np.zeros(self.mdp.n)
        self.mdp.policy = np.zeros(self.mdp.n, dtype=int)
        self.mdp.action_vals = np.zeros((self.mdp.n, self.mdp.m))

        # Value iteration step which effectively combines evaluation and improvement.
        for evaluation in xrange(max_eval):
            
            delta = 0

            for s in self.mdp.states:            
                v_temp = self.mdp.v[s].copy()       
                
                self.mdp.v[s] = max([sum(self.mdp.P[s, a, s_new] 
                                     * (self.mdp.R[s, a, s_new] + gamma*self.mdp.v[s_new]) 
                                     for s_new in self.mdp.states) for a in self.mdp.actions])

                delta = max(delta, abs(v_temp - self.mdp.v[s]))

            if delta < 1e-10:
                break

        # Finding the deterministic policy for the value function.
        for s in self.mdp.states:
            self.mdp.action_vals[s] = [sum(self.mdp.P[s, a, s_new] 
                                           * (self.mdp.R[s, a, s_new] + gamma*self.mdp.v[s_new]) 
                                           for s_new in self.mdp.states) for a in self.mdp.actions]

        self.mdp.policy = random_argmax(self.mdp.action_vals)
    
        self.get_named_policy()

        # Policy probability distribution.
        self.mdp.pi = np.zeros((self.mdp.n, self.mdp.m))
        self.mdp.pi[np.arange(self.mdp.pi.shape[0]), self.mdp.policy] = 1. 


    def q_value_iteration(self, gamma=1., max_eval=1000):
        """Find the optimal q function using q value iteration.

        The optimal value function and policy are also updated using the optimal
        q function that is found.

        :param gamma: Float discounting factor for rewards in (0,1].
        :param max_eval: Integer max number of evaluations to do for q value iteration.
        """

        self.mdp.q = np.zeros((self.mdp.n, self.mdp.m))

        for evaluation in xrange(max_eval):

            delta = 0

            for state_action in itertools.product(self.mdp.states, self.mdp.actions):
                s = state_action[0]
                a = state_action[1]

                q_temp = self.mdp.q[s, a].copy()

                self.mdp.q[s, a] = sum(self.mdp.P[s, a, s_new] 
                                       * (self.mdp.R[s, a, s_new] 
                                          + gamma*self.mdp.q[s_new, :].max()) 
                                       for s_new in self.mdp.states)

                delta = max(delta, abs(self.mdp.q[s, a] - q_temp))

            if delta < 1e-10:
                break

        self.mdp.v = self.mdp.q.max(axis=1)
        self.mdp.policy = random_argmax(self.mdp.q)
        self.get_named_policy()


    def one_step_temporal_difference(self, policy=None, gamma=1, num_episodes=100):
        """Finding the value function for a policy using temporal difference.

        :param policy: If using a new policy to evaluate pass as array.
        :param gamma: Float discounting factor for rewards in (0,1].
        :param num_episodes: Integer number of episodes to run one step TD.
        """

        if policy is None:
            pass
        else:
            self.mdp.policy = policy
        
        self.mdp.v = np.zeros(self.mdp.n)

        if not self.mdp.terminal_states:
            print('Need to add terminal states: Exiting')
            return

        for episode in xrange(num_episodes):

            s = np.random.choice(self.mdp.states)

            while s not in self.mdp.terminal_states:
                a = self.mdp.policy[s]
                s_new, reward = self.mdp.sample_transition(s, a)

                self.mdp.v[s] = self.mdp.v[s] + gamma*(reward + gamma*self.mdp.v[s_new] - self.mdp.v[s])

                s = s_new


    def epsilon_greedy(self, s):
        """Epsilon greedy exploration-exploitation strategy.

        This policy strategy selects the current best action with probability
        of 1 - epsilon, and a random action with probability epsilon.
        
        :param s: Integer index of the current state index the agent is in.

        :return a: Integer index of the chosen index for the agent to take.
        """

        if not np.random.binomial(1, self.epsilon):
            a = random_argmax(self.mdp.q[s])
        else:
            a = np.random.choice(self.mdp.actions)

        return a


    def softmax(self, s):
        """Softmax exploration-exploitation strategy.

        This policy strategy uses a boltzman distribution with a temperature 
        parameter tau, to assign the probabilities of choosing an action based
        off of the current q value of the state and action.

        :param s: Integer index of the current state index the agent is in.

        :return a: Integer index of the chosen index for the agent to take.
        """

        exp = lambda s, a: np.exp(self.mdp.q[s, a]/self.tau) 
        
        total = sum(exp(s, a) for a in self.mdp.actions)

        probs = [exp(s, a)/total for a in self.mdp.actions]

        sample = np.random.multinomial(1, probs).tolist()
        a = sample.index(1)

        return a


    def choose_action(self, s):
        """Choose action for a TD algorithm that is updating using q values.

        The policy strategy for choosing an action is either chosen using a
        softmax strategy, epsilon greedy strategy, greedy strategy, or a random strategy.

        :param s: Integer index of the current state index the agent is in.

        :return a: Integer index of the chosen index for the action to take.
        """

        if self.policy_strategy == 'softmax':
            a = self.softmax(s)
        elif self.policy_strategy == 'e-greedy':
            a = self.epsilon_greedy(s)
        elif self.policy_strategy == 'greedy':
            a = random_argmax(self.mdp.q[s])
        else:
            a = np.random.choice(self.mdp.actions)

        return a


    def sarsa(self, policy_strategy='softmax', epsilon=.2, tau=100, gamma=1, 
              alpha=.5, num_episodes=1000):
        """Finding the q function using the on policy TD method SARSA.

        :param policy_strategy: String indicating policy strategy to choose actions with.
        :param epsilon: Float epsilon value in (0, 1) indicating probability of 
        taking random action with the epsilon greedy policy strategy.
        :param tau: Float value for temperature parameter to use in the softmax
        policy strategy.
        :param gamma: Float discounting factor for rewards in (0,1].
        :param alpha: Float step size parameter for TD step. Typically in (0,1].
        :param num_episodes: Integer number of episodes to run algorithm.
        """

        self.epsilon = epsilon
        self.tau = float(tau)
        self.policy_strategy = policy_strategy

        self.mdp.q = np.zeros((self.mdp.n, self.mdp.m))

        if not self.mdp.terminal_states:
            print('Need to add terminal states: Exiting')
            return

        for episode in xrange(num_episodes):

            s = np.random.choice(self.mdp.states)
            a = np.random.choice(self.mdp.actions)

            while s not in self.mdp.terminal_states:
                s_new, reward = self.mdp.sample_transition(s, a)
                a_new = self.choose_action(s_new)
                
                self.mdp.q[s, a] = self.mdp.q[s, a] + alpha*(reward 
                                                            + gamma*self.mdp.q[s_new, a_new] 
                                                            - self.mdp.q[s, a])

                s = s_new
                a = a_new

        self.mdp.v = self.mdp.q.max(axis=1)
        self.mdp.policy = random_argmax(self.mdp.q)
        self.get_named_policy()


    def q_learning(self, policy_strategy='softmax', epsilon=.2, tau=100, gamma=1, 
                   alpha=.5, num_episodes=1000):
        """Finding the q function using the off policy TD method q-learning.

        :param policy_strategy: String indicating policy strategy to choose actions with.
        :param epsilon: Float epsilon value in (0, 1) indicating probability of 
        taking random action with the epsilon greedy policy strategy.
        :param tau: Float value for temperature parameter to use in the softmax
        policy strategy.
        :param gamma: Float discounting factor for rewards in (0,1].
        :param alpha: Float step size parameter for TD step. Typically in (0,1].
        :param num_episodes: Integer number of episodes to run algorithm.
        """

        self.epsilon = epsilon
        self.tau = float(tau)
        self.policy_strategy = policy_strategy

        self.mdp.q = np.zeros((self.mdp.n, self.mdp.m))

        if not self.mdp.terminal_states:
            print('Need to add terminal states: Exiting')
            return

        for episode in xrange(num_episodes):

            s = np.random.choice(self.mdp.states)

            while s not in self.mdp.terminal_states:
                a = self.choose_action(s)
                s_new, reward = self.mdp.sample_transition(s, a)

                self.mdp.q[s, a] = self.mdp.q[s, a] + alpha*(reward
                                                           + gamma*self.mdp.q[s_new].max()
                                                           - self.mdp.q[s, a])

                s = s_new

        self.mdp.v = self.mdp.q.max(axis=1)
        self.mdp.policy = random_argmax(self.mdp.q)
        self.get_named_policy()


class grid_display(object):
    def __init__(self, rl, title='Grid World', fig_path=None, fig_name=None, savefig=False):
            """Create and plot the grid with the values and policy.

            :param rl: RL object with needed information.
            :param title: Title for the figure.
            """

            # Flipping rows for plotting reasons.
            self.values = np.flipud(rl.mdp.v.reshape((rl.mdp.grid_rows, rl.mdp.grid_cols)))

            self.rl = rl

            # Converting terminal state locs to the flipped orientation.
            state_locs = np.flipud(np.arange(self.rl.mdp.n).reshape((self.rl.mdp.grid_rows, self.rl.mdp.grid_cols)))
            state_locs = state_locs.reshape((-1)).tolist()
            self.terminal_states = [state_locs.index(state) for state in self.rl.mdp.terminal_states]

            self.named_policy = zip(*[iter(self.rl.mdp.named_policy)]*self.rl.mdp.m)[::-1]
            self.named_policy = [item for row in self.named_policy for item in row]
            
            self.title = title
            self.savefig = savefig

            if self.savefig:
                if fig_path is None:
                    fig_path = os.getcwd() + '/../figs'

                if fig_name is None:
                    title = title.translate(None, string.punctuation)
                    fig_name = '_'.join(title.split()) + '.png'

                self.full_path = os.path.join(fig_path, fig_name)
            else:
                self.full_path = None


    def setup_grid(self):

        self.cmap = mcolors.LinearSegmentedColormap.from_list('cmap', ['red', 'black', 'limegreen'])

        rc('axes', linewidth=4)

        self.fig, self.ax = plt.subplots(facecolor='black', edgecolor='white', linewidth=4)    

        self.grid = self.ax.pcolor(self.values, edgecolors='white', linewidths=4, cmap=self.cmap, 
                                   vmin=self.values.min(), vmax=self.values.max())


    def finish_grid(self):

        for spine in self.ax.spines.values():
            spine.set_edgecolor('white')
                
        x_axis_size = self.values.shape[1]
        y_axis_size = self.values.shape[0]

        xlabels = [str(val) for val in range(0, x_axis_size)]
        ylabels = [str(val) for val in range(y_axis_size-1, -1, -1)]

        self.ax.set_xticks(np.arange(0.5, len(xlabels)))
        self.ax.set_yticks(np.arange(0.5, len(ylabels)))

        self.ax.set_xticklabels(xlabels)                                                       
        self.ax.set_yticklabels(ylabels) 

        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')

        for tick in self.ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(24)
            tick.label1.set_fontweight('bold')
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(24)
            tick.label1.set_fontweight('bold')
        
        plt.title(self.title, color='white', fontsize='24', fontweight='bold')
            
        self.fig.set_size_inches((self.values.shape[1]*4, self.values.shape[0]*4))

        if self.savefig:
            plt.savefig(self.full_path, facecolor='black')

        plt.show()


    def show_v(self):
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


    def show_q(self):
        """

        """

        warnings.simplefilter('ignore', MatplotlibDeprecationWarning)
        ax = self.grid.get_axes()

        count = 0

        for p, value, choice in izip(self.grid.get_paths(), self.grid.get_array(), self.named_policy):
            x, y = p.vertices[:-2, :].mean(0)
  
            j = 0       
            colors = {0:'red', 1:'blue', 2:'orange', 3:'white'}   
            min_v = self.values.min()
            max_v = self.values.max()

            verts = [[] for a in range(self.rl.mdp.m)]
            verts[0] = [x, y]
            verts[-1] = [0, 0]
            codes = [mpl.path.Path.MOVETO, mpl.path.Path.LINETO, mpl.path.Path.LINETO, mpl.path.Path.CLOSEPOLY]

            # W, N, E, S
            dist_change = {0:(-.3,.0), 1:(0.,.3), 2:(.3,0), 3:(0,-.3)}

            # N, S, E, W.
            a_change = {0: 3, 1:0, 2:2, 3:1}
            q_ = 0

            for v in range(self.rl.mdp.m):
  
                if v == len(p.vertices) - 1:
                    verts[1] = p.vertices[v]
                    verts[2] = p.vertices[0]
                else:
                    verts[1] = p.vertices[v]
                    verts[2] = p.vertices[v+1]

                path = mpl.path.Path(verts, codes)

                patch = patches.PathPatch(path, edgecolor='white', 
                                          facecolor=self.cmap((self.q_values[count][a_change[v]] - min_v)/(max_v-min_v)), lw=4)
                ax.add_patch(patch)

                ax.text(x+dist_change[v][0], y+dist_change[v][1], "%.2f" % self.q_values[count][a_change[v]], 
                        ha="center", va="center", color='white', fontweight='bold', fontsize='24')
            
            count += 1

    
    
    def show_q_values(self):
        """Create and plot the grid with the values and policy.

        :param savefig: Bool indicating whether or not to save the figure.
        :param full_path: String with combined file path and figure name.
        """
        
        self.setup_grid()

        self.q_values = tuple(map(tuple, self.rl.mdp.q))
        self.q_values = zip(*[iter(self.q_values)]*self.rl.mdp.m)[::-1]
        self.q_values = [item for row in self.q_values for item in row]

        self.show_q()

        self.finish_grid()



    def show_values(self):
        """Create and plot the grid with the values and policy.

        :param savefig: Bool indicating whether or not to save the figure.
        :param full_path: String with combined file path and figure name.
        """
        
        self.setup_grid()

        self.show_v()

        self.finish_grid()


def random_argmax(arr):
    """Helper functio to get the argmax of an array breaking ties randomly."""

    if len(arr.shape) == 1:
        choice = np.random.choice(np.flatnonzero(arr == arr.max()))
        return choice
    else:
        N = arr.shape[0]
        argmax_array = np.zeros(N)

        for i in xrange(N):
            choice = np.random.choice(np.flatnonzero(arr[i] == arr[i].max()))
            argmax_array[i] = choice

        return argmax_array.astype(int)
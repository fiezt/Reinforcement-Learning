import numpy as np
import itertools
import sys
import os
import string
import matplotlib.pyplot as plt
from matplotlib.cbook import MatplotlibDeprecationWarning
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib as mpl
from pylab import rc
from itertools import izip
import seaborn as sns
sns.reset_orig()
import warnings


def random_argmax(arr):
    """Helper function to get the argmax of an array breaking ties randomly.
    
    :param arr: 1D or 1D numpy array to find the argmax for.

    :return choice or argmax_array: Choice is integer index of array with 
    the max value, argmax_array is array of integer index of max value in each 
    row of the original array.
    """

    if len(arr.shape) == 1:
        choice = np.random.choice(np.flatnonzero(arr == arr.max()))
        return choice
    else:
        N = arr.shape[0]
        argmax_array = np.zeros(N)

        for i in xrange(N):
            choice = np.random.choice(np.flatnonzero(arr[i] == arr[i].max()))
            argmax_array[i] = choice

        argmax_array = argmax_array.astype(int)

        return argmax_array


class RLBase(object):
    def __init__(self):
        pass


    def simulate_policy(self, env, num_episodes=1000, render_final=True):
        """Interact in the environment using the learned policy.

        This function is used to interact in the environment and get the average
        reward over of number of episodes and show the last episode if specified.
        The trajectories are also stored of the state action pairs in each episode.

        :param env: environment class which the algorithm will attempt to learn.
        :param num_episodes: Integer number of episodes to run in environment for.
        :param render_final: Bool indicator of whether to show the last episode.

        :return t
        """

        # List tracking rewards of learning algorithm over episodes.
        self.episode_rewards = []

        # List of episode trajectories containing state action pairs.
        self.trajectories = []

        total_reward = 0

        for episode in xrange(num_episodes):

            epsiode_trajectory = []

            done = False
            episode_reward = 0
            s = env.reset()

            while done != True:
                # Show the environment to the screen.
                if episode == num_episodes - 1 and render_final:
                    env.render()

                a = self.policy[s]

                epsiode_trajectory.append((s, a))

                s, reward, done, info = env.step(a) 

                episode_reward += reward

            self.trajectories.append(epsiode_trajectory)
            self.episode_rewards.append(episode_reward)
            total_reward += episode_reward

        avg_reward = total_reward/float(num_episodes)

        print('\n\nAverage reward over %d episodes is %f\n\n' % (num_episodes, avg_reward))


    def plot_epsiode_returns(self, title='Episode Returns', fig_path=None, 
                             fig_name=None, save_fig=True):
        """Plotting the reward returns over episodes.
        
        :param title: String title for figure.
        :param fig_path: File path to save figure to.
        :param fig_name: File name to save figure as.
        :param save_fig: Bool indicating whether to save the figure.
        """

        sns.set()
        sns.set_style("whitegrid")

        plt.figure()

        plt.plot(self.episode_rewards, color='red', lw=2)

        plt.title(title, fontsize=22)
        plt.xlabel('Episode Number', fontsize=20)
        plt.ylabel('Episode Return', fontsize=20)

        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tick_params(axis='both', which='minor', labelsize=18)
        plt.xlim([0, len(self.episode_rewards)])

        plt.tight_layout()

        if save_fig:
            # Default figure path.
            if fig_path is None:
                fig_path = os.getcwd() + '/../figs'

            # Default figure name.
            if fig_name is None:
                title = title.translate(None, string.punctuation)
                fig_name = '_'.join(title.split()) + '.png'

            plt.savefig(os.path.join(fig_path, fig_name), bbox_inches='tight')

        sns.reset_orig()

        plt.show()


    def scatter_epsiode_returns(self, title='Episode Returns', fig_path=None, 
                                fig_name=None, save_fig=True):
        """Scatter plotting the reward returns over episodes.
        
        :param title: String title for figure.
        :param fig_path: File path to save figure to.
        :param fig_name: File name to save figure as.
        :param save_fig: Bool indicating whether to save the figure.
        """

        sns.set()
        sns.set_style("whitegrid")

        plt.figure()

        plt.scatter(range(len(self.episode_rewards)), self.episode_rewards, color='red', lw=2)

        plt.title(title, fontsize=22)
        plt.xlabel('Episode Number', fontsize=20)
        plt.ylabel('Episode Return', fontsize=20)

        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tick_params(axis='both', which='minor', labelsize=18)
        plt.xlim([0, len(self.episode_rewards)])

        plt.tight_layout()

        if save_fig:
            # Default figure path.
            if fig_path is None:
                fig_path = os.getcwd() + '/../figs'

            # Default figure name.
            if fig_name is None:
                title = title.translate(None, string.punctuation)
                fig_name = '_'.join(title.split()) + '.png'

            plt.savefig(os.path.join(fig_path, fig_name), bbox_inches='tight')

        sns.reset_orig()

        plt.show()


class ModelBasedRL(RLBase):
    def __init__(self, gamma=1, max_iter=5000, max_eval=1000):
        """
        :param gamma: Float discounting factor for rewards in (0,1].
        :param max_iter: Integer max number of iterations to run policy evaluation and improvement.
        :param max_eval: Integer max number of evaluations to run on each state or each state and action.
        """

        self.gamma = gamma
        self.max_iter = max_iter
        self.max_eval = max_eval


    def get_policy(self, mdp):
        """Given the value function find the policy for actions in states.
    
        :param mdp: Markov decision process object containing standard information.
        """
        
        self.action_vals = np.zeros((mdp.n, mdp.m))

        for s in mdp.states:
            self.action_vals[s] = [sum(mdp.P[s, a] * (mdp.R[s, a] + self.gamma*self.v)) 
                                       for a in mdp.actions]

        self.policy = random_argmax(self.action_vals)

    
    def iterative_policy_evaluation(self, mdp, pi=None):
        """Iterative policy evaluation finds the state value function for a policy.

        :param mdp: Markov decision process object containing standard information.
        :param pi: Probability distribution of actions given states.
        """
        
        # Random policy if a policy is not provided.
        if pi is None:
            mdp.pi = 1/float(mdp.m) * np.ones((mdp.n, mdp.m))
        else:
            mdp.pi = pi
        
        self.v = np.zeros(mdp.n)

        for iteration in xrange(self.max_iter):
            
            delta = 0

            for s in mdp.states:            
                v_temp = self.v[s].copy()       
                
                # Bellman equation to back up.
                self.v[s] = sum(mdp.pi[s, a] * sum(mdp.P[s, a] 
                                * (mdp.R[s, a] + self.gamma*self.v)) 
                                for a in mdp.actions)

                delta = max(delta, abs(v_temp - self.v[s]))

            # Convergence check.
            if delta < 1e-4:
                break
        
        self.get_policy(mdp)
 

    def policy_iteration(self, mdp):
        """Finds optimal policy and the value function for that policy.
        
        :param mdp: Markov decision process object containing standard information.
        """
        
        self.v = np.zeros(mdp.n)
        self.policy = np.zeros(mdp.n, dtype=int)
        self.action_vals = np.zeros((mdp.n, mdp.m))

        # Policy evaluation followed by policy improvement until convergence.
        for iteration in xrange(self.max_iter):

            # Policy evaluation.
            for evaluation in xrange(self.max_eval):
                
                delta = 0

                for s in mdp.states:    
                    v_temp = self.v[s].copy()       
                    a = self.policy[s]

                    # Bellman equation to back up.
                    self.v[s] = sum(mdp.P[s, a] * (mdp.R[s, a] + self.gamma*self.v)) 

                    delta = max(delta, abs(v_temp - self.v[s]))

                # Convergence check.
                if delta < 1e-4:
                    break

            # Policy improvement.
            stable = True

            for s in mdp.states:
                old_policy = self.policy[s].copy()

                self.action_vals[s] = [sum(mdp.P[s, a] * (mdp.R[s, a] + self.gamma*self.v)) 
                                           for a in mdp.actions]

                self.policy[s] = random_argmax(self.action_vals[s])

                if self.policy[s] != old_policy and stable:
                    stable = False

            # Policy convergence check.
            if stable:
                break
        
        # Policy probability distribution.
        self.pi = np.zeros((mdp.n, mdp.m))
        self.pi[np.arange(self.pi.shape[0]), self.policy] = 1. 


    def value_iteration(self, mdp):
        """Find the optimal value function and policy with value iteration.
        
        :param mdp: Markov decision process object containing standard information.
        """

        self.v = np.zeros(mdp.n)
        self.policy = np.zeros(mdp.n, dtype=int)

        # Value iteration step which effectively combines evaluation and improvement.
        for evaluation in xrange(self.max_eval):
            
            delta = 0

            for s in mdp.states:            
                v_temp = self.v[s].copy()       
                
                # Bellman equation to back up.
                self.v[s] = max([sum(mdp.P[s, a] * (mdp.R[s, a] + self.gamma*self.v)) 
                                     for a in mdp.actions])

                delta = max(delta, abs(v_temp - self.v[s]))

            # Convergence check.
            if delta < 1e-4:
                break

        self.get_policy(mdp)

        # Setting policy probability distribution to the greedy policy.
        self.pi = np.zeros((mdp.n, mdp.m))
        self.pi[np.arange(self.pi.shape[0]), self.policy] = 1. 


    def q_value_iteration(self, mdp):
        """Find the optimal q function using q value iteration.

        :param mdp: Markov decision process object containing standard information.
        """

        # Initializing q function.
        self.q = np.zeros((mdp.n, mdp.m))

        for evaluation in xrange(self.max_eval):

            delta = 0

            for state_action in itertools.product(mdp.states, mdp.actions):
                s = state_action[0]
                a = state_action[1]

                q_temp = self.q[s, a].copy()

                # Bellman equation to back up.
                self.q[s, a] = sum(mdp.P[s, a] * (mdp.R[s, a] + self.gamma*self.q.max(axis=1)))

                delta = max(delta, abs(self.q[s, a] - q_temp))

            # Convergence check.
            if delta < 1e-4:
                break

        self.v = self.q.max(axis=1)
        self.policy = random_argmax(self.q)

        # Setting policy probability distribution to the greedy policy.
        self.pi = np.zeros((mdp.n, mdp.m))
        self.pi[np.arange(self.pi.shape[0]), self.policy] = 1. 


    def test_optimal_q(self, mdp):
        """Testing if the q function is optimal.

        :param mdp: Markov decision process object containing standard information.
        """

        self.error = np.zeros((mdp.n, mdp.m))

        for s, a in itertools.product(mdp.states, mdp.actions):
            self.error[s, a] = sum(mdp.P[s, a] * (mdp.R[s, a] + self.gamma*self.q.max(axis=1) 
                                   - self.q[s,a]))

        self.error = abs(self.error)


    def test_optimal_v(self, mdp):
        """Testing if the value function is optimal.

        :param mdp: Markov decision process object containing standard information.
        """

        self.error = np.zeros(mdp.n)

        for s in mdp.states:                
            self.error[s] = max([sum(mdp.P[s, a] * (mdp.R[s, a] + self.gamma*self.v - self.v[s]))
                                 for a in mdp.actions])

        self.error = abs(self.error)


class ModelFreeRLBase(RLBase):
    def __init__(self, n, m, states, actions, gamma, alpha, alpha_decay, 
                 alpha_decay_param, epsilon, epsilon_decay, 
                 epsilon_decay_param, tau, tau_decay, tau_decay_param, 
                 policy_strategy, horizon, num_episodes):
        """Initialize model free reinforcement learning parameters.
        
        :param n: Integer number of discrete states.
        :param m: Integer number of discrete actions.
        :param states: List of all states.
        :param actions: List of all actions.
        :param gamma: Float discounting factor for rewards in (0,1].
        :param alpha: Float step size parameter for TD step. Typically in (0,1].
        :param alpha_decay: Bool indicating whether to use decay of step size.
        :param alpha_decay_param: Float param for decay given by alpha*e^(-alpha_decay_param * s_t)
        :param epsilon: Float value (0, 1) prob of taking random action vs. taking greedy action.
        :param epsilon_decay: Bool indicating whether to use decay of epsilon over episodes.
        :param epsilon_decay_param: Float param for decay given by epsilon*e^(-epsilon_decay_param * episode)
        :param tau: Float value for temp. param in the softmax, tau -> 0 = greedy, tau -> infinity = random.
        :param tau_decay: Bool indicating whether to use decay of tau over episodes.
        :param tau_decay_param: Float param for decay given by tau*e^(-tau_decay_param * episode)
        :param policy_strategy: String in {softmax, e-greedy, greedy}, exploration vs exploitation strategy.  
        :param horizon: Integer maximum number of steps to run an episode for.
        :param num_episodes: Integer number of episodes to run learning for.
        """

        self.n = n
        self.m = m
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_decay_param = alpha_decay_param
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_param = epsilon_decay_param
        self.tau = tau
        self.tau_decay = tau_decay
        self.tau_decay_param = tau_decay_param
        self.policy_strategy = policy_strategy
        self.horizon = horizon
        self.num_episodes = num_episodes


    def initialize(self):
        """Initialize parameters for learning algorithms."""

        # Q-value function - values of taking action a in state s and acting optimally after.
        self.q = np.zeros((self.n, self.m))

        # Value function - values of state acting optimally from the state.
        self.v = np.zeros(self.n)

        # Counts of state, action, state transition.
        self.visited_states = np.zeros((self.n, self.m, self.n))

        # Cumulative rewards for each state, action, state transition.
        self.experienced_rewards = np.zeros((self.n, self.m, self.n))

        # List tracking rewards of learning algorithm over episodes.
        self.episode_rewards = []

        # List tracking epsilon choices over episodes.
        self.epsilon_choices = []

        # List tracking tau choices over episodes.
        self.tau_choices = []

        # List of lists tracking step size over states (outer) and episodes (inner). 
        self.alpha_choices = [[] for s in self.states]

        """Default min reward which will be changed, so when learning the model
        from experience states that are not visited can be set to minimum reward."""
        self.reward_min = 1000000000


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
            a = random_argmax(self.q[s])
        else:
            a = np.random.choice(self.actions)

        return a


    def epsilon_greedy(self, s):
        """Epsilon greedy exploration-exploitation strategy.

        This policy strategy selects the current best action with probability
        of 1 - epsilon, and a random action with probability epsilon.
        
        :param s: Integer index of the current state index the agent is in.

        :return a: Integer index of the chosen index for the agent to take.
        """

        if self.epsilon_decay:
            epsilon = self.epsilon * np.exp(-self.epsilon_decay_param * self.episode)
        else:
            epsilon = self.epsilon

        if len(self.epsilon_choices) <= self.episode:
            self.epsilon_choices.append(epsilon)

        if not np.random.binomial(1, epsilon):
            a = random_argmax(self.q[s])
        else:
            a = np.random.choice(self.actions)

        return a


    def softmax(self, s):
        """Softmax exploration-exploitation strategy.

        This policy strategy uses a boltzman distribution with a temperature 
        parameter tau, to assign the probabilities of choosing an action based
        off of the current q value of the state and action.

        :param s: Integer index of the current state index the agent is in.

        :return a: Integer index of the chosen index for the agent to take.
        """

        if self.tau_decay:
            # Capping the minimum value of tau to prevent overflow issues.
            tau = max(self.tau * np.exp(-self.tau_decay_param * self.episode), .1)
        else:
            tau = self.tau

        if len(self.tau_choices) <= self.episode:
            self.tau_choices.append(tau)

        exp = lambda s, a: np.exp(self.q[s, a]/tau)
        values = []
        probs = []

        for a in self.actions:
            # Catching overflow and returning greedy action if it occurs.
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    value = exp(s, a)
                except RuntimeWarning:
                    return random_argmax(self.q[s])

            values.append(value) 
        
        total = sum(values)
        probs = [val/total for val in values]

        try:
            sample = np.random.multinomial(1, probs).tolist()
            a = sample.index(1)
        except:
            # Return greedy action if there is overflow issues.
            a = random_argmax(self.q[s])

        return a


    def set_alpha(self, s):
        """Selecting step size parameter for temporal difference methods.

        :param s: Integer index of the current state the agent is in.
        
        :return alpha: Step size parameter for temporal difference error.
        """

        if self.alpha_decay:
            alpha = self.alpha * np.exp(-self.alpha_decay_param * self.visited_states[s].sum())
        else:
            alpha = self.alpha

        self.alpha_choices[s].append(alpha)

        return alpha


    def get_learned_model(self):
        """Get the learned probability and reward distributions from sampled transitions."""

        self.P = np.zeros((self.n, self.m, self.n))
        self.R = np.zeros((self.n, self.m, self.n))

        for s in self.states:
            for a in self.actions:
                # If a state and action was never taken set to uniform probability.
                if self.visited_states[s, a].sum() == 0:
                    self.P[s, a] = 1./self.n
                    self.R[s, a] = 0.
                else:
                    # In case of 0/0, this flag will change average to nan.
                    with np.errstate(divide='ignore', invalid='ignore'):
                        self.R[s, a] = self.experienced_rewards[s, a]/self.visited_states[s, a]

                    self.P[s, a] = self.visited_states[s, a]/self.visited_states[s, a].sum()

        # Converting nan value to 0.
        self.R = np.nan_to_num(self.R)

        # Converting learned reward for all transitions not visited to minimum reward.
        self.R[np.where(self.visited_states == 0)[0], np.where(self.visited_states == 0)[1], 
                        np.where(self.visited_states == 0)[2]] = self.reward_min

        self.check_valid_dist()


    def check_valid_dist(self):
        """Checking the probability distribution sums to 1 for each state, action pair."""

        for s in self.states:
            for a in self.actions:
                assert abs(sum(self.P[s, a, :]) - 1) < 1e-3, 'Transitions do not sum to 1'


    def plot_alpha_parameters(self, s, title='State Step Choices', fig_path=None, 
                              fig_name=None, save_fig=True):
        """Plotting the step size choices for a state over episodes.
        
        :param s: Integer state index to plot the step size choices for.
        :param title: String title for figure.
        :param fig_path: File path to save figure to.
        :param fig_name: File name to save figure as.
        :param save_fig: Bool indicating whether to save the figure.
        """

        sns.set()
        sns.set_style("whitegrid")

        plt.figure()

        plt.plot(self.alpha_choices[s], color='red', lw=2)

        plt.title(title, fontsize=22)
        plt.xlabel('State Visit Number', fontsize=20)
        plt.ylabel(r'$\alpha_s$', fontsize=20)

        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tick_params(axis='both', which='minor', labelsize=18)
        plt.xlim([0, len(self.alpha_choices[s])])

        plt.tight_layout()

        if save_fig:
            # Default figure path.
            if fig_path is None:
                fig_path = os.getcwd() + '/../figs'

            # Default figure name.
            if fig_name is None:
                title = title.translate(None, string.punctuation)
                fig_name = '_'.join(title.split()) + '.png'

            plt.savefig(os.path.join(fig_path, fig_name), bbox_inches='tight')

        sns.reset_orig()

        plt.show()


    def plot_epsilon_parameters(self, title='Epsilon Parameters', fig_path=None, 
                                fig_name=None, save_fig=True):
        """Plotting the e-greedy parameter over episodes.
        
        :param title: String title for figure.
        :param fig_path: File path to save figure to.
        :param fig_name: File name to save figure as.
        :param save_fig: Bool indicating whether to save the figure.
        """

        sns.set()
        sns.set_style("whitegrid")

        plt.figure()

        plt.plot(self.epsilon_choices, color='red', lw=2)

        plt.title(title, fontsize=22)
        plt.xlabel('Episode Number', fontsize=20)
        plt.ylabel(r'$\epsilon$', fontsize=20)

        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tick_params(axis='both', which='minor', labelsize=18)
        plt.xlim([0, len(self.epsilon_choices)])

        plt.tight_layout()

        if save_fig:
            # Default figure path.
            if fig_path is None:
                fig_path = os.getcwd() + '/../figs'

            # Default figure name.
            if fig_name is None:
                title = title.translate(None, string.punctuation)
                fig_name = '_'.join(title.split()) + '.png'

            plt.savefig(os.path.join(fig_path, fig_name), bbox_inches='tight')

        sns.reset_orig()

        plt.show()


    def plot_tau_parameters(self, title='Tau Parameters', fig_path=None, 
                            fig_name=None, save_fig=True):
        """Plotting softmax parameter tau over episodes.
        
        :param title: String title for figure.
        :param fig_path: File path to save figure to.
        :param fig_name: File name to save figure as.
        :param save_fig: Bool indicating whether to save the figure.
        """

        sns.set()
        sns.set_style("whitegrid")

        plt.figure()

        plt.plot(self.tau_choices, color='red', lw=2)

        plt.title(title, fontsize=22)
        plt.xlabel('Episode Number', fontsize=20)
        plt.ylabel(r'$\tau$', fontsize=20)

        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tick_params(axis='both', which='minor', labelsize=18)
        plt.xlim([0, len(self.tau_choices)])

        plt.tight_layout()

        if save_fig:
            # Default figure path.
            if fig_path is None:
                fig_path = os.getcwd() + '/../figs'

            # Default figure name.
            if fig_name is None:
                title = title.translate(None, string.punctuation)
                fig_name = '_'.join(title.split()) + '.png'

            plt.savefig(os.path.join(fig_path, fig_name), bbox_inches='tight')

        sns.reset_orig()

        plt.show()


class ModelFreeRL(ModelFreeRLBase):
    def __init__(self, n, m, states, actions, gamma=1, alpha=.618, alpha_decay=True, 
                 alpha_decay_param=.001, epsilon=.2, epsilon_decay=True, epsilon_decay_param=.01, 
                 tau=100, tau_decay=True, tau_decay_param=.01, policy_strategy='e-greedy', 
                 horizon=1000, num_episodes=2000):

        super(ModelFreeRL, self).__init__(n, m, states, actions, gamma, alpha, 
                                          alpha_decay, alpha_decay_param,
                                          epsilon, epsilon_decay, epsilon_decay_param, 
                                          tau, tau_decay, tau_decay_param, policy_strategy, 
                                          horizon, num_episodes)

    
    def one_step_temporal_difference(self, env):
        """Finding the value function for a policy using one step temporal difference.

        A policy should already be defined for the class before running this function.
        
        :param env: environment class which the algorithm will attempt to learn.
        """

        self.initialize()

        for self.episode in xrange(self.num_episodes):

            done = False
            episode_reward = 0
            s = env.reset()
            iteration = 0

            while done != True and iteration < self.horizon:
                a = self.policy[s]
                s_new, reward, done, info = env.step(a)

                self.reward_min = min(self.reward_min, reward)
                episode_reward += reward

                self.visited_states[s, a, s_new] += 1.
                self.experienced_rewards[s, a, s_new] += reward

                alpha = self.set_alpha(s)

                # One step temporal difference equation.
                self.v[s] += self.alpha*(reward + self.gamma*self.v[s_new] - self.v[s])

                s = s_new
                iteration += 1

            self.episode_rewards.append(episode_reward)

        self.get_learned_model()


    def sarsa(self, env):
        """Finding the q function using the on policy TD method SARSA.

        :param env: environment class which the algorithm will attempt to learn.
        """

        self.initialize()

        for self.episode in xrange(self.num_episodes):

            done = False
            episode_reward = 0
            s = env.reset()
            a = self.choose_action(s)
            iteration = 0

            while done != True and iteration < self.horizon:
                s_new, reward, done, info = env.step(a)

                self.reward_min = min(self.reward_min, reward)
                episode_reward += reward

                a_new = self.choose_action(s_new)

                self.visited_states[s, a, s_new] += 1.
                self.experienced_rewards[s, a, s_new] += reward

                alpha = self.set_alpha(s)
                
                # On policy temporal difference update equation.
                self.q[s, a] += alpha*(reward + self.gamma*self.q[s_new, a_new] - self.q[s, a])

                s = s_new
                a = a_new
                iteration += 1

            self.episode_rewards.append(episode_reward)

        self.get_learned_model()
        self.v = self.q.max(axis=1)
        self.policy = random_argmax(self.q)


    def q_learning(self, env):
        """Finding the q function using the off policy TD method q-learning.

        :param env: environment class which the algorithm will attempt to learn.
        """

        self.initialize()

        for self.episode in xrange(self.num_episodes):

            done = False
            episode_reward = 0
            s = env.reset()
            iteration = 0

            while done != True and iteration < self.horizon:
                a = self.choose_action(s)
                s_new, reward, done, info = env.step(a)

                self.reward_min = min(self.reward_min, reward)
                episode_reward += reward

                self.visited_states[s, a, s_new] += 1.
                self.experienced_rewards[s, a, s_new] += reward

                alpha = self.set_alpha(s)

                # Off policy temporal difference update equation.
                self.q[s, a] += alpha*(reward + self.gamma*self.q[s_new].max() - self.q[s, a])

                s = s_new
                iteration += 1

            self.episode_rewards.append(episode_reward)

        self.get_learned_model()
        self.v = self.q.max(axis=1)
        self.policy = random_argmax(self.q)


class ModelFreeRiskRL(ModelFreeRLBase):
    def __init__(self, n, m, states, actions, gamma=1, alpha=.618, alpha_decay=True, 
                 alpha_decay_param=.001, epsilon=.2, epsilon_decay=True, 
                 epsilon_decay_param=.01, tau=100, tau_decay=True, tau_decay_param=.01, 
                 policy_strategy='e-greedy', horizon=1000, num_episodes=2000):

        super(ModelFreeRiskRL, self).__init__(n, m, states, actions, gamma, alpha, 
                                              alpha_decay, alpha_decay_param,
                                              epsilon, epsilon_decay, epsilon_decay_param, 
                                              tau, tau_decay, tau_decay_param, policy_strategy, 
                                              horizon, num_episodes)


    def risk_q_learning(self, env, agent):
        """Finding the q function using risk sensitive q learning.

        Risk sensitive q learning uses a mapping of temporal differences 
        through a utility function from an agent.

        :param env: environment class which the algorithm will attempt to learn.
        :param agent: Agent class containing utility function for mapping TD.
        """

        self.initialize()

        for self.episode in xrange(self.num_episodes):

            done = False
            episode_reward = 0
            s = env.reset()
            iteration = 0

            while done != True and iteration < self.horizon:
                a = self.choose_action(s)
                s_new, reward, done, info = env.step(a)

                self.reward_min = min(self.reward_min, reward)
                episode_reward += reward

                self.visited_states[s, a, s_new] += 1.
                self.experienced_rewards[s, a, s_new] += reward

                alpha = self.set_alpha(s)

                # Risk sensitive q learn step mapping TD through utility function.
                self.q[s, a] += alpha*(agent.u(reward + self.gamma*self.q[s_new].max() - self.q[s, a]) - agent.ref)

                s = s_new
                iteration += 1

            self.episode_rewards.append(episode_reward)

        self.get_learned_model()
        self.v = self.q.max(axis=1)
        self.policy = random_argmax(self.q)


    def eu_q_learning(self, env, agent):
        """Finding the q function using expected utility algorithm q learning.

        Expected utility q learning uses a mapping of rewards through a utility
        function from an agent.

        :param env: environment class which the algorithm will attempt to learn.
        :param agent: Agent class containing utility function for mapping TD.
        """

        self.initialize()

        for self.episode in xrange(self.num_episodes):

            done = False
            episode_reward = 0
            s = env.reset()
            iteration = 0

            while done != True and iteration < self.horizon:
                a = self.choose_action(s)
                s_new, reward, done, info = env.step(a)

                self.reward_min = min(self.reward_min, reward)
                episode_reward += reward

                self.visited_states[s, a, s_new] += 1.
                self.experienced_rewards[s, a, s_new] += reward

                alpha = self.set_alpha(s)

                # expected utility q learn step mapping reward through utility function.
                self.q[s, a] += alpha*(agent.u(reward) + self.gamma*self.q[s_new].max() - self.q[s, a])

                s = s_new
                iteration += 1

            self.episode_rewards.append(episode_reward)

        self.get_learned_model()
        self.v = self.q.max(axis=1)
        self.policy = random_argmax(self.q)


class RiskAgent(object):
    def __init__(self, ref=0, c_minus=1, c_plus=1, rho_minus=1, rho_plus=1, lamb=0):
        """Initialize agent parameters for its utility function.

        :param ref: Float reference point for risk value function.
        :param c_minus: Float scaling parameter for losses of prospect and
        logarithmic value functions.
        :param c_plus: Float scaling parameter for gains of prospect and
        logarithmic value functions.
        :param rho_minus: Float risk parameter for losses of prospect and
        logarithmic value functions. Value in (0, 1) leads to risk seeking 
        behavior in losses, value equal to 1 leads to risk neutral behavior 
        in losses, value greater than 1 leads to risk adverse behavior in losses.
        :param rho_plus: Float risk parameter for gains of prospect and
        logarithmic value functions. Value in (0, 1) leads to risk adverse 
        behavior in gains, value equal to 1 leads to risk neutral behavior 
        in gains, value greater than 1 leads to risk seeking behavior in gains.
        :param lamb: Float risk parameter for exponential value function.
        Value greater than 0 leads to risk adverse behavior and value less 
        than 0 leads to risk seeking behavior.
        """

        self.ref = ref
        self.c_minus = c_minus
        self.c_plus = c_plus
        self.rho_minus = rho_minus
        self.rho_plus = rho_plus
        self.lamb = lamb


class ProspectAgent(RiskAgent):
    def __init__(self, ref=0, c_minus=1, c_plus=1, rho_minus=1, rho_plus=1):

        super(ProspectAgent, self).__init__(ref=ref, c_minus=c_minus, c_plus=c_plus, 
                                            rho_minus=rho_minus, rho_plus=rho_plus)


    def u(self, x):
        """Mapping a value to a prospect theory utility value."""

        if isinstance(x, float) or isinstance(x, int):
            if x > self.ref:
                u = self.c_plus * (x - self.ref)**self.rho_plus
            elif x <= self.ref:
                u = -self.c_minus * (self.ref - x)**self.rho_minus
        elif isinstance(x, np.ndarray) or isinstance(x, list):

            u = np.zeros(len(x))

            for i in range(len(x)):
                if x[i] > self.ref:
                    u[i] = self.c_plus * (x[i] - self.ref)**self.rho_plus
                elif x[i] <= self.ref:
                    u[i] = -self.c_minus * (self.ref - x[i])**self.rho_minus
        else:
            print('Bad data txpe input to prospect value function')
            u = x

        return u


    def dudc(self, x):
        """Compute the derivative of the prospect value function with respect to c.""" 

        if x > self.ref:
            du_dc = (x - self.ref)**self.rho_plus
        elif x <= self.ref:
            du_dc = -1.*(self.ref - x)**self.rho_minus

        return du_dc


    def dudrho(self, x):
        """Compute the derivative of the prospect value function with respect to rho."""

        if x > self.ref:
            du_drho = self.c_plus * np.log(x - self.ref) * (x - self.ref)**self.rho_plus
        elif x <= self.ref:
            du_drho = -self.c_minus * np.log(self.ref - x) * (self.ref - x)**self.rho_minus

        return du_drho


    def dudx(self, x):
        """Compute the derivative of the prospect value function with respect to x."""

        if x > self.ref:
            du_dx = self.c_plus * self.rho_plus * (x - self.ref)**(self.rho_plus - 1.)
        elif x <= self.ref:
            du_dx = -self.c_minus * self.rho_minus * (self.ref - x)**(self.rho_minus - 1.)

        return du_dx


class EntropicAgent(RiskAgent):
    def __init__(self, ref=0, lamb=0):

        super(EntropicAgent, self).__init__(ref=ref, lamb=lamb)


    def u(self, x):
        """Mapping a value to a entropic utility value."""

        u = (np.exp(self.lamb*x) - 1)/self.lamb 
        return u


    def dudc(self, x):
        """Computing the derivative of the entropic value function with respect to lambda."""

        de_dl = (-1./self.lamb**2) * (np.exp(self.lamb*x) - 1) + (x * np.exp(self.lamb*x))/(self.lamb)

        return de_dl


    def dudx(self, x):
        """Computing the derivative of the entropic value function with respect to x."""

        de_dx = np.exp(self.lamb*x)

        return de_dx


class LogAgent(RiskAgent):
    def __init__(self, ref=0, c_minus=1, c_plus=1, rho_minus=1, rho_plus=1):

        super(LogAgent, self).__init__(ref=ref, c_minus=c_minus, c_plus=c_plus, 
                                       rho_minus=rho_minus, rho_plus=rho_plus)


    def u(self, x):
        """Mapping a value to logarithm utility value."""

        if isinstance(x, float) or isinstance(x, int):
            if x > self.ref:
                u = self.c_plus * np.log(1. + self.rho_plus*(x - self.ref))
            elif x <= self.ref:
                u = -self.c_minus * np.log(1. + self.rho_minus*(self.ref - x))
        elif isinstance(x, np.ndarray) or isinstance(x, list):
            
            u = np.zeros(len(x))

            for i in range(len(x)):
                if x[i] > self.ref:
                    u[i] = self.c_plus * np.log(1. + self.rho_plus*(x[i] - self.ref))
                elif x[i] <= self.ref:
                    u[i] = -self.c_minus * np.log(1. + self.rho_minus*(self.ref - x[i]))
        else:
            print('Bad data txpe input to prospect value function')
            u = x

        return u


    def dudc(self, x):
        """Compute the derivative of the log value function with respect to c."""

        if x > self.ref:
            dl_dc = np.log(1. + self.rho_plus*(x - self.ref))
        elif x <= self.ref:
            dl_dc = -1.*np.log(1. + self.rho_minus*(self.ref - x))

        return dl_dc


    def dudrho(self, x):
        """Compute the derivative of the log value function with respect to rho."""

        if x > self.ref:
            dl_drho = ((self.c_plus * (x - self.ref))/(1. + self.rho_plus*(x - self.ref))) 
        elif x <= self.ref:
            dl_drho = ((-self.c_minus * (self.ref - x))/(1. + self.rho_minus*(self.ref - x))) 

        return dl_drho


    def dudx(self, x):
        """Compute the derivative of the log value function with respect to x."""

        if x > self.ref:
            dl_dx = ((self.c_plus * self.rho_plus)/(1. + self.rho_plus*(x - self.ref))) 
        elif x <= self.ref:
            dl_dx = ((self.c_minus * self.rho_minus)/(1. + self.rho_minus*(self.ref - x))) 

        return dl_dx


class SimulatedMDP(object):
    def __init__(self, env, num_episodes=2000):
        """Creating a MDP from simulated experience in an environment.
        
        :param env: environment class which the agent will interact with.
        :param num_episodes: Integer number of episodes to interact.
        """
        
        # OpenAI Gym case.
        try:
            self.n = env.observation_space.n
        # Grid World case.
        except:
            self.n = env.n
        self.states = range(self.n)
        
        # OpenAI Gym case.
        try:
            self.m = env.action_space.n
        # Grid World case.
        except:
            self.m = env.m
        self.actions = range(self.m)

        self.P = np.zeros((self.n, self.m, self.n))
        self.R = np.zeros((self.n, self.m, self.n))

        counts, reward_min = self.simulate_environment(env, num_episodes)
        self.get_learned_model(counts, reward_min)


    def simulate_environment(self, env, num_episodes):
        """Simulate the agent in the environment to get sample transitions.
        
        :param env: environment class which the agent will interact with.
        :param num_episodes: Integer number of episodes to interact.

        :return counts: Numpy array containing counts of transitions for a (s, a, s'). 
        :return reward_min: Float smallest reward encountered in a transition.
        """

        counts = np.zeros((self.n, self.m, self.n))
        reward_min = 100000000
        
        for episode in range(num_episodes):
            
            s = env.reset()
            done = False
            
            while not done:
                # OpenAI Gym case.
                try:
                    a = env.action_space.sample()
                # Grid World Case.
                except:
                    a = env.sample()
                s_new, reward, done, info = env.step(a)
                
                reward_min = min(reward_min, reward)
                
                counts[s, a, s_new] += 1.
                self.R[s, a, s_new] += reward
                
                s = s_new

        return counts, reward_min
        

    def get_learned_model(self, counts, reward_min):
        """Get the learned probability and reward distributions from sampled transitions.
        
        :return counts: Numpy array containing counts of transitions for a (s, a, s'). 
        :return reward_min: Float smallest reward encountered in a transition.
        """

        for s in self.states:
            for a in self.actions:
                # If a state and action was never taken set to uniform probability.
                if counts[s, a].sum() == 0:
                    self.P[s, a] = 1./self.n
                    self.R[s, a] = 0.
                else:
                    # In case of 0/0, this flag will change average to nan.
                    with np.errstate(divide='ignore', invalid='ignore'):
                        self.R[s, a] = self.R[s, a]/counts[s, a]

                    self.P[s, a] = counts[s, a]/counts[s, a].sum()

        # Converting nan value to 0.
        self.R = np.nan_to_num(self.R)

        # Converting learned reward for all transitions not visited to minimum reward.
        self.R[np.where(counts == 0)[0], np.where(counts == 0)[1], np.where(counts == 0)[2]] = reward_min

        self.check_valid_dist()
        
        
    def check_valid_dist(self):
        """Checking the probability distribution sums to 1 for each state, action pair."""

        for s in self.states:
            for a in self.actions:
                assert abs(sum(self.P[s, a, :]) - 1) < 1e-3, 'Transitions do not sum to 1'


class Bandits(object):
    def __init__(self, num_arms, mu, sigma):
        """Initializing bandit problem by arms and parameters for the number of arms.

        :param num_arms: Number of arms to use in the bandit problem.
        :param mu: Mean for normal distribution to create over arms.
        :param sigma: Standard deviation for normal distribution to create over arms.
        """

        self.num_arms = num_arms
        self.mu = mu
        self.sigma = sigma


    def create_arms(self):
        """Creating the arms using a normal distribution on the arms."""

        self.bandit_arms = np.random.normal(self.mu, self.sigma, self.num_arms)


    def ucb(self):
        """Upper confidence bound bandit algorithm."""

        a = random_argmax(self.arm_averages + self.c*np.sqrt(np.log(self.i+1)/self.arm_counts))
        return a


    def epsilon_greedy(self):
        """Epsilon greedy exploration-exploitation strategy.

        This policy strategy selects the current best action with probability
        of 1 - epsilon, and a random action with probability epsilon.
        
        :return a: Integer index of the action for the agent to take.
        """

        if self.epsilon_decay:
            epsilon = self.epsilon * np.exp(-self.epsilon_decay_param * self.i)
        else:
            epsilon = self.epsilon

        if not np.random.binomial(1, epsilon):
            a = random_argmax(self.arm_averages)
        else:
            a = np.random.choice(range(self.num_arms))

        return a


    def softmax(self):
        """Softmax exploration-exploitation strategy.

        This policy strategy uses a boltzman distribution with a temperature 
        parameter tau, to assign the probabilities of choosing an action based
        off of the current sample arm averages.

        :return a: Integer index of the action for the agent to take.
        """

        if self.tau_decay:
            # Capping the minimum value of tau to prevent overflow issues.
            tau = max(self.tau * np.exp(-self.tau_decay_param * self.i), .1)
        else:
            tau = self.tau

        values = np.exp(self.arm_averages/tau)        
        total = np.sum(values)
        probs = values/total

        sample = np.random.multinomial(1, probs).tolist()

        # If there is overflow take the greedy action.
        try:
            a = sample.index(1)
        except:
            a = random_argmax(self.arm_averages)

        return a


    def sample_arm(self, arm_choice):
        """Sample arm by returning random reward for the choice of arm to pull.

        :param arm_choice: Index of the arm to pull.

        :return value: Random reward for choosing the arm selected.
        """

        value = np.random.normal(self.bandit_arms[arm_choice], self.sigma)
        return value


    def choose_action(self):
        """Choose action for a bandit algorithm using selected strategy.

        :return a: Integer index of the action for the agent to take.
        """

        if self.policy_strategy == 'softmax':
            a = self.softmax()
        elif self.policy_strategy == 'e-greedy':
            a = self.epsilon_greedy()
        elif self.policy_strategy == 'greedy':
            a = random_argmax(self.arm_averages)
        elif self.policy_strategy == 'ucb':
            a = self.ucb()
        else:
            a = np.random.choice(range(self.num_arms))

        return a


    def initialize(self, epsilon, epsilon_decay, epsilon_decay_param, tau, 
                   tau_decay, tau_decay_param, c, policy_strategy):
        """Initializing bandit parameters for policies.
    
        :param epsilon: Float value (0, 1) prob of taking random action vs. taking greedy action.
        :param epsilon_decay: Bool indicating whether to use decay of epsilon over episodes.
        :param epsilon_decay_param: Float param for decay given by epsilon*e^(-epsilon_decay_param * episode)
        :param tau: Float value for temp. param in the softmax, tau -> 0 = greedy, tau -> infinity = random.
        :param tau_decay: Bool indicating whether to use decay of tau over episodes.
        :param tau_decay_param: Float param for decay given by tau*e^(-tau_decay_param * episode)
        :param c: Float param for the UCB algorithm.
        :param policy_strategy: String in {softmax, e-greedy, greedy, ucb}, exploration vs exploitation strategy.  
        """

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_param = epsilon_decay_param
        self.tau = tau
        self.tau_decay = tau_decay
        self.tau_decay_param = tau_decay_param
        self.c = c
        self.policy_strategy = policy_strategy


    def simulate(self, horizon=1000, num_episodes=2000, epsilon=.2, 
                 epsilon_decay=True, epsilon_decay_param=.01, tau=100, 
                 tau_decay=True, tau_decay_param=.01, c=2, policy_strategy='ucb'):
        """Running the bandit problems using chosen algorithms.

        :param horizon: Number of times to sample using the bandit algorithm.
        :param num_episodes: Number of different randomly created bandit problems to run that will be averaged over.
        :param epsilon: Float value (0, 1) prob of taking random action vs. taking greedy action.
        :param epsilon_decay: Bool indicating whether to use decay of epsilon over episodes.
        :param epsilon_decay_param: Float param for decay given by epsilon*e^(-epsilon_decay_param * episode)
        :param tau: Float value for temp. param in the softmax, tau -> 0 = greedy, tau -> infinity = random.
        :param tau_decay: Bool indicating whether to use decay of tau over episodes.
        :param tau_decay_param: Float param for decay given by tau*e^(-tau_decay_param * episode)
        :param c: Float param for the UCB algorithm.
        :param policy_strategy: String in {softmax, e-greedy, greedy, ucb}, exploration vs exploitation strategy.  
        """

        self.initialize(epsilon, epsilon_decay, epsilon_decay_param, tau, 
                        tau_decay, tau_decay_param, c, policy_strategy)

        all_run_returns = np.zeros((num_episodes, horizon))
        all_run_optimal = np.zeros((num_episodes, horizon))
        all_run_regret = np.zeros((num_episodes, horizon))
        
        # Each iteration runs a new randomly created bandit problem.        
        for run in xrange(num_episodes):
            self.create_arms()

            optimal_action = random_argmax(self.bandit_arms)
            optimal_return = self.bandit_arms.max()
            
            self.arm_returns = np.zeros(self.num_arms)
            self.arm_averages = np.zeros(self.num_arms)
            self.arm_counts = np.ones(self.num_arms)
            
            optimal_counts = np.zeros(horizon)
            samples = np.zeros(horizon)
            
            # Each iteration runs a single step for the current bandit problem.
            for self.i in xrange(horizon):
                
                a = self.choose_action()
                    
                if a == optimal_action:
                    optimal_counts[self.i] = 1
                else:
                    optimal_counts[self.i] = 0 

                reward = self.sample_arm(a)
                samples[self.i] = reward
                
                self.arm_returns[a] += reward
                self.arm_averages[a] = self.arm_returns[a]/self.arm_counts[a] 
                self.arm_counts[a] += 1.
            
            all_run_returns[run] = samples
            all_run_optimal[run] = optimal_counts
            all_run_regret[run] = optimal_return - samples

        # Averaging values of interest over all bandit problems.
        self.average_rewards = all_run_returns.mean(axis=0)
        self.average_optimal = all_run_optimal.mean(axis=0) * 100
        self.average_regret = all_run_regret.mean(axis=0).cumsum()


    def initialize_figs(self, reward=True, regret=True, optimal=True):
        """This function creates figures to plot selected values of interest.

        :param reward: Bool indicating whether to plot the reward at each round.
        :param regret: Bool indicating whether to plot the cumulative regret.
        :param optimal: Bool indicating whether to plot the percentage of optimal actions
        at each round.
        """

        self.reward = reward
        self.regret = regret
        self.optimal = optimal

        sns.set()
        sns.set_style("whitegrid")

        if self.reward:
            self.fig1 = plt.figure(1)
            self.ax1 = plt.axes()
        if self.regret:
            self.fig2 = plt.figure(2)
            self.ax2 = plt.axes()
        if self.optimal:
            self.fig3 = plt.figure(3)
            self.ax3 = plt.axes()


    def get_label(self):
        """Choosing a label for a plot based off of the simulation parameters."""

        if self.policy_strategy == 'softmax':
            if self.tau_decay:
                self.label = r'softmax,  $\tau = %g$, $\tau$-decay$ = %g$' % (self.tau, self.tau_decay_param)
            else:
                self.label = r'softmax $\tau$ = %g' % self.tau
        elif self.policy_strategy == 'e-greedy':
            if self.epsilon_decay:
                self.label = r'$\epsilon$-greedy, $\epsilon = %g$, $\epsilon$-decay$ = %g$' % (self.epsilon, self.epsilon_decay_param)
            else:
                self.label = r'$\epsilon$-greedy, $\epsilon = %g$' % self.epsilon
        elif self.policy_strategy == 'greedy':
            self.label = 'greedy'
        elif self.policy_strategy == 'ucb':
            self.label = r'UCB, $c = %g$' % self.c
        else:
            self.label = 'random' 


    def plot_average_reward(self):
        """Plotting the average reward at each round from current simulation."""

        self.get_label()
        self.ax1.plot(self.average_rewards, label=self.label, lw=1)


    def plot_cumulative_regret(self):
        """Plotting the cumulative regret from current simulation."""

        self.get_label()
        self.ax2.plot(self.average_regret, label=self.label, lw=1)


    def plot_average_optimal(self):
        """Plotting the percentage of optimal actions at each round from current simulation."""

        self.get_label()
        self.ax3.plot(self.average_optimal, label=self.label, lw=1)


    def save_and_show_figs(self, save_figs=False, show_figs=True, fig_path=None):
        """Saving and showing results of rewards, regret, and optimal actions.

        :param save_figs: Bool indicating whether to save the figures.
        :param show_figs: Bool indicating whether to show the figures.
        :param fig_path: Path to save the figures.
        """

        if self.reward:
            self.ax1.set_xlabel('Steps', fontsize=22)
            self.ax1.set_ylabel('Average Reward', fontsize=22)
            self.ax1.set_title('Bandit Testbed of Strategies', fontsize=22)
            self.ax1.legend(bbox_to_anchor=(1, 1), loc='upper left', fancybox=True, fontsize=20)
            self.ax1.set_xlim([0, len(self.average_rewards)])
            self.ax1.tick_params(axis='both', which='major', labelsize=20)
            self.ax1.tick_params(axis='both', which='minor', labelsize=20)

            if save_figs:
                if fig_path is None:
                    fig_path = os.getcwd() + '/../figs'
                self.fig1.savefig(os.path.join(fig_path, 'bandit_rewards.png'), bbox_inches='tight')

        if self.regret:
            self.ax2.set_xlabel('Steps', fontsize=22)
            self.ax2.set_ylabel('Cumulative Regret', fontsize=22)
            self.ax2.set_title('Bandit Testbed of Strategies', fontsize=22)
            self.ax2.legend(bbox_to_anchor=(1, 1), loc='upper left', fancybox=True, fontsize=20)
            self.ax2.set_xlim([0, len(self.average_regret)])
            self.ax2.tick_params(axis='both', which='major', labelsize=20)
            self.ax2.tick_params(axis='both', which='minor', labelsize=20)

            if save_figs:
                if fig_path is None:
                    fig_path = os.getcwd() + '/../figs'
                self.fig2.savefig(os.path.join(fig_path, 'bandit_regret.png'), bbox_inches='tight')

        if self.optimal:
            self.ax3.set_xlabel('Steps', fontsize=22)
            self.ax3.set_ylabel(r'$\%$ Optimal Action', fontsize=22)
            self.ax3.set_title('Bandit Testbed of Strategies', fontsize=22)
            self.ax3.legend(bbox_to_anchor=(1, 1), loc='upper left', fancybox=True, fontsize=20)
            self.ax3.set_xlim([0, len(self.average_optimal)])
            self.ax3.tick_params(axis='both', which='major', labelsize=20)
            self.ax3.tick_params(axis='both', which='minor', labelsize=20)

            if save_figs:
                if fig_path is None:
                    fig_path = os.getcwd() + '/../figs'
                self.fig3.savefig(os.path.join(fig_path, 'bandit_optimal_action.png'), bbox_inches='tight')

        if show_figs:
            plt.show()

        sns.reset_orig()


class IRL(object):
    def __init__(self):
        pass


class GridWorldBase(object):
    def __init__(self, grid_rows, grid_cols, num_actions, terminal_states):
        """Initializing base class for grid mdp or environment for state, action information.

        :param grid_rows: Integer number of rows for which to make the grid.
        :param grid_cols: Integer number of cols for which to make the grid.
        :param num_actions: Integer that must be 4 or 8. These will be compass directions.
        :param terminal_states: List of the terminal states as integers.
        """     

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        # Number of states.
        self.n = grid_cols * grid_rows

        # Number of actions.
        self.m = num_actions
        
        self.states = range(self.n)
        self.actions = range(self.m)

        self.terminal_states = terminal_states

        if self.m == 4:
            
            # Action of N, W, S, E.
            self.actions_to_idx = {(-1,0): 0, (0,-1): 1, (1,0): 2, (0,1): 3}            
            self.idx_to_actions = {v:k for k,v in self.actions_to_idx.items()}   
            
            self.action_names_to_idx = {'N':0, 'W':1, 'S':2, 'E':3}
            self.idx_to_action_names = {v:k for k,v in self.action_names_to_idx.items()}

        elif self.m == 8:
            
            # Action of N, W, S, E, NE, NW, SW, SE 
            self.actions_to_idx = {(-1,0): 0, (0,-1): 1, (1,0): 2, (0,1): 3, 
                                    (-1,1): 4, (-1,-1): 5, (1,-1): 6, (1,1): 7}
            
            self.idx_to_actions = {v:k for k,v in self.actions_to_idx.items()}
            
            self.action_names_to_idx = {'N':0, 'W':1, 'S':2 , 'E':3, 'NE':4, 
                                        'NW':5, 'SW':6, 'SE':7}
            self.idx_to_action_names = {v:k for k,v in self.action_names_to_idx.items()}

        else:
            print('The number of actions must be 4 or 8: Exiting')
            return  

        self.idx_to_states = {self.states[i]:(i/self.grid_cols, i%self.grid_cols) for i in range(self.n)}
        self.states_to_idx = {v:k for k,v in self.idx_to_states.items()}


    def create_prob_dist(self, noise=0.0):
        """Creating the probability distribution for the MDP or environment.

        Default function to create a probability distribution for the MDP or environment.
        This is a noiseless distribution. If the agent takes an action, it will
        go to the desired location with probability 1 if the action is legal, 
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
                for noisy_action in self.actions:
                    if noisy_action == action:
                        continue

                    new_pos = (curr_pos[0] + self.idx_to_actions[noisy_action][0], 
                               curr_pos[1] + self.idx_to_actions[noisy_action][1])

                    if new_pos in self.states_to_idx:
                        new_state = self.states_to_idx[new_pos]
                        self.P[state, action, new_state] = noise/float(self.m - 1)
                    else:
                        self.P[state, action, state] += noise/float(self.m - 1)
        
    
    def create_rewards(self, noise=0.0):
        """Creating the reward structure for the MDP or environment.

        The default reward structure. -1 for all actions, except 0 if 
        going to terminal state. The noise parameter can be used to make the 
        rewards not uniform by adding in Gaussian noise with mean 0 and standard
        deviation of the noise to each of the state, action, next state rewards.

        :param noise: Float indicating the standard deviation for 0 mean Gaussian
        noise to add into each reward for a state, action, next state tuple.
        """
        
        noise_array = np.random.normal(0, noise, (self.n, self.m, self.n))
        self.R = -1*np.ones((self.n, self.m, self.n))

        # Adding noise selected so not all rewards are equal.
        self.R = noise_array + self.R

        for state in self.terminal_states:
            self.R[state] = 0        
    

    def check_valid_dist(self):
        """Checking the probability distribution sums to 1 for each state, action pair."""

        for s in self.states:
            for a in self.actions:
                assert abs(self.P[s, a].sum() - 1) < 1e-3, 'Transitions do not sum to 1'


class GridWorldMDP(GridWorldBase):
    def __init__(self, grid_rows=4, grid_cols=4, num_actions=4, terminal_states=[0,15], 
                 prob_noise=0.0, reward_noise=0.0, create_probs=True, create_rewards=True):
        """Creating the grid world MDP. Note that everything is indexed by 
        row, column, not x, y coordinates.

        :param grid_rows: Integer number of rows for which to make the grid.
        :param grid_cols: Integer number of cols for which to make the grid.
        :param num_actions: Integer that must be 4 or 8. These will be compass directions.
        :param terminal_states: List or None, of the terminal states as integers.
        :param prob_noise: Float in [0,1] to use as noise in the transition function.
        :param reward_noise: Float as standard deviation for Gaussian noise in the reward function.
        :param create_probs: Bool indicating whether to create the default probability distribution.
        :param create_rewards: Bool indicating whether to create the default reward distribution.
        """  
    
        super(GridWorldMDP, self).__init__(grid_rows=grid_rows, grid_cols=grid_cols, 
                                           num_actions=num_actions, 
                                           terminal_states=terminal_states)

        if create_probs:        
            self.create_prob_dist(prob_noise)
            self.check_valid_dist()

        if create_rewards:
            self.create_rewards(reward_noise)
        

class GridWorldEnv(GridWorldBase):
    def __init__(self, grid_rows, grid_cols, num_actions=4, terminal_states=[0,15], 
                 prob_noise=0.0, reward_noise=0.0, create_probs=True, create_rewards=True):
        """Creating the grid world environment. Note that everything is indexed by 
        row, column, not x, y coordinates.

        :param grid_rows: Integer number of rows for which to make the grid.
        :param grid_cols: Integer number of cols for which to make the grid.
        :param num_actions: Integer that must be 4 or 8. These will be compass directions.
        :param terminal_states: List or None, of the terminal states as integers.
        :param prob_noise: Float in [0,1] to use as noise in the transition function.
        :param reward_noise: Float as standard deviation for Gaussian noise in the reward function.
        :param create_probs: Bool indicating whether to create the default probability distribution.
        :param create_rewards: Bool indicating whether to create the default reward distribution.
        """  

        super(GridWorldEnv, self).__init__(grid_rows=grid_rows, grid_cols=grid_cols, 
                                           num_actions=num_actions, 
                                           terminal_states=terminal_states)

        self.clear_states = [s for s in self.states if s not in self.terminal_states]

        if create_probs:
            self.create_prob_dist(prob_noise)
            self.check_valid_dist()
            self.reset()

        if create_rewards:
            self.create_rewards(reward_noise)

    def reset(self):
        """Set the state to a random state.
        
        :return s: Integer index of the new state of the environment.
        """

        self.s = np.random.choice(self.clear_states)
        s = self.s

        return s


    def sample(self):
        """Select a random action.

        :return a: Integer index of the action.
        """

        a = np.random.choice(self.actions)
        return a


    def step(self, a):
        """Make a transition using the environments probability distribution.
        
        :param a: Integer index of the current action being taken.
        
        :return s_new: Integer state index the environment transitioned to.
        :return reward: Reward for the transition that occurred in the environment.
        :return done: Bool indicating whether the episode has ended.
        :return info: Blank list to match OpenAI gym format.
        """

        sample = np.random.multinomial(1, self.P[self.s, a]).tolist()
        s_new = sample.index(1)

        if self.s in self.terminal_states:
            done = True
        else:
            done = False

        reward = self.R[self.s, a, s_new]
        info = []

        # Updating state internally.
        self.s = s_new

        return s_new, reward, done, info       


class GridDisplay(object):
    def __init__(self, rl, grid_object):
        """Create and plot the grid with the values and policy.

        :param rl: RL object with needed information.
        :param title: Title for the figure.
        """

        # Flipping rows for plotting reasons.
        self.values = np.flipud(rl.v.reshape((grid_object.grid_rows, grid_object.grid_cols)))

        # Converting terminal state locs to the flipped orientation.
        state_locs = np.flipud(np.arange(grid_object.n).reshape((grid_object.grid_rows, grid_object.grid_cols)))
        state_locs = state_locs.reshape((-1)).tolist()
        self.terminal_states = [state_locs.index(state) for state in grid_object.terminal_states]

        # Flipping rows for plotting reasons.
        self.policy = np.flipud(rl.policy.reshape((grid_object.grid_rows, grid_object.grid_cols)))
        self.policy = self.policy.reshape((-1)).tolist()

        self.rl = rl
        self.grid_object = grid_object

    
    def show_q_values(self, title='Grid World', fig_path=None, fig_name=None, save_fig=False):
        """Create and plot the grid with the q-values.

        :param title: Title for the figure.
        :param fig_path: Path to save the figure to.
        :param fig_name: File name to save the figure as.
        :param save_fig: Bool indicating whether or not to save the figure.
        """
        
        cmap = mcolors.LinearSegmentedColormap.from_list('cmap', ['red', 'black', 'limegreen'])
        rc('axes', linewidth=4)

        fig, ax = plt.subplots(facecolor='black', edgecolor='white', linewidth=4)    

        grid = ax.pcolor(self.values, edgecolors='white', linewidths=4, cmap=cmap, 
                         vmin=self.values.min(), vmax=self.values.max())

        # Rearranging q values for plotting reasons.
        self.q_values = tuple(map(tuple, self.rl.q))
        self.q_values = zip(*[iter(self.q_values)]*self.grid_object.m)[::-1]
        self.q_values = [item for row in self.q_values for item in row]

        warnings.simplefilter('ignore', MatplotlibDeprecationWarning)
        ax = grid.get_axes()

        count = 0
        for p, value, choice in izip(grid.get_paths(), grid.get_array(), self.policy):
            x, y = p.vertices[:-2, :].mean(0)

            j = 0       
            min_v = self.values.min()
            max_v = self.values.max()

            verts = [[] for a in range(self.grid_object.m)]
            verts[0] = [x, y]
            verts[-1] = [0, 0]
            codes = [mpl.path.Path.MOVETO, mpl.path.Path.LINETO, mpl.path.Path.LINETO, mpl.path.Path.CLOSEPOLY]

            """
            Mapping from vertex iteration of W, N, E, S to distance to move 
            text from the vertex. The reference point is the bottom left hand corner at 0,0.
            """
            dist_change = {0:(-.3,0.), 1:(.0,.3), 2:(.3,.0), 3:(.0,-.3)}

            # Mapping from vertex iteration of W, N, E, S to directions N, W, S, E.
            a_change = {0:1, 1:0, 2:3, 3:2}

            for v in range(self.grid_object.m):
                if v == len(p.vertices) - 1:
                    verts[1] = p.vertices[v]
                    verts[2] = p.vertices[0]
                else:
                    verts[1] = p.vertices[v]
                    verts[2] = p.vertices[v+1]

                path = mpl.path.Path(verts, codes)

                patch = patches.PathPatch(path, edgecolor='white', 
                                          facecolor=cmap((self.q_values[count][a_change[v]] - min_v)/(max_v-min_v)), lw=4)
                ax.add_patch(patch)
                ax.text(x+dist_change[v][0], y+dist_change[v][1], "%.2f" % self.q_values[count][a_change[v]], 
                        ha="center", va="center", color='white', fontweight='bold', fontsize='24')
            count += 1

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
        
        plt.title(title, color='white', fontsize='24', fontweight='bold')
            
        fig.set_size_inches((self.values.shape[1]*4, self.values.shape[0]*4))

        if save_fig:
            # Default figure path.
            if fig_path is None:
                fig_path = os.getcwd() + '/../figs'

            # Default figure name.
            if fig_name is None:
                title = title.translate(None, string.punctuation)
                fig_name = '_'.join(title.split()) + '.png'

            plt.savefig(os.path.join(fig_path, fig_name), facecolor='black')

        plt.show()


    def show_values(self, title='Grid World', fig_path=None, fig_name=None, save_fig=False):
        """Create and plot the grid with the values and policy.

        :param title: Title for the figure.
        :param fig_path: Path to save the figure to.
        :param fig_name: File name to save the figure as.
        :param save_fig: Bool indicating whether or not to save the figure.
        """
        
        cmap = mcolors.LinearSegmentedColormap.from_list('cmap', ['red', 'black', 'limegreen'])
        rc('axes', linewidth=4)

        fig, ax = plt.subplots(facecolor='black', edgecolor='white', linewidth=4)    

        grid = ax.pcolor(self.values, edgecolors='white', linewidths=4, cmap=cmap, 
                         vmin=self.values.min(), vmax=self.values.max())

        warnings.simplefilter('ignore', MatplotlibDeprecationWarning)
        ax = grid.get_axes()


        orient_dict = {0:0, 1:np.pi/2., 2:np.pi, 3:3*np.pi/2., 
                       4:7*np.pi/4., 5:np.pi/4., 6:3*np.pi/4., 7:5*np.pi/4.}

        dist = 0.42
        arrow_loc = {0:(0, dist), 1:(-dist, 0), 2:(0, -dist),
                     3:(dist, 0), 4:(dist, dist), 5:(-dist, dist),
                     6:(-dist, -dist), 7:(dist, -dist)}

        count = 0

        for p, value, choice in izip(grid.get_paths(), grid.get_array(), self.policy):
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
        
        plt.title(title, color='white', fontsize='24', fontweight='bold')
            
        fig.set_size_inches((self.values.shape[1]*4, self.values.shape[0]*4))

        if save_fig:
            if fig_path is None:
                fig_path = os.getcwd() + '/../figs'

            if fig_name is None:
                title = title.translate(None, string.punctuation)
                fig_name = '_'.join(title.split()) + '.png'

            plt.savefig(os.path.join(fig_path, fig_name), facecolor='black')

        plt.show()
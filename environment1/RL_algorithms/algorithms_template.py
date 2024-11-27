from collections import defaultdict

import numpy as np
from scipy.special import softmax


def softmax_(env, beta, Q):
    """
    Chooses an action using softmax distribution over the available actions
    :param env: environment
    :param beta: scaling parameter of the softmax policy
    :param Q: current Q-values
    assumed structure: dictionary of dictionaries: Q = {(x,y):{"u": , "d": , "l": , "r": } for (x,y) in T-maze}
    :return:
        - the chosen action
    """
    # Hint: start by filtering out the non-available actions in the current state
    state = env.get_state()
    available_actions =  env.available()
    Q_vals = np.array([Q[state][action] for action in available_actions])
    
    # Hint: remember to rescale all Q-values for the current state by beta
    Q_vals = Q_vals*beta
    # Hint: to do a softmax operation on a set of Q-values you can use the scipy.special.softmax() function
    action = np.random.choice(available_actions, p=softmax(Q_vals))
    action = str(action)
    return action


def epsilon_greedy(env, epsilon, Q):
    """
    Chooses an epsilon-greedy action starting from a given state (which you can access via env.get_state()) and given a set of
    Q-values
    :param env: environment
    :param epsilon: current exploration parameter
    :param Q: current Q-values.
    :return:
        - the chosen action
    """
    # Hint: start by filtering out the non-available actions in the current state
    state = env.get_state()
    available_actions =  env.available()        
    
    
    if np.random.uniform(0, 1) < epsilon:
        # with probability epsilon make a random move (exploration)
        action = available_actions[np.random.randint(len(available_actions))]
    else:
        Q_vals = np.array([Q[state][action] for action in available_actions])
        # with probability 1-epsilon choose the action with the highest immediate reward (exploitation):
        # Hint: remember to break ties randomly
        permuts = np.random.permutation(len(available_actions))
        action = available_actions[permuts[np.argmax(Q_vals[permuts])]]
    return action



def sarsa(env, alpha=0.05, gamma=0.99, num_episodes=1000, action_policy="epsilon_greedy", epsilon_exploration=0.1,
          epsilon_exploration_rule=None, trace_decay=0, initial_q=0):
    """
    Trains an agent using the Sarsa algorithm by playing num_episodes games until the reward states are reached
    :param env: the environment
    :param alpha: learning rate
    :param gamma: discount rate for future rewards
    :param num_episodes: number of training episodes
    :param action_policy: string for the action policy to be followed during training
        It is usually "epsilon_greedy" or "softmax_"
    :param trace_decay: trace decay factor for eligibility traces
        If 0, sarsa(0) is implemented without any eligibility trace. If a non-zero float is given in input
        the latter represents the trace decay factor and sarsa(lambda) is implemented
    :param epsilon_exploration: parameter of the exploration policy: exploration rate or softmax_ scaling factor
        If action_policy is "epsilon_greedy":
            If epsilon_exploration_rule is None, at each iteration the action with the highest Q-value
            is taken with probability (1-epsilon_exploration)
        If action_policy is "softmax_":
            epsilon_exploration is actually beta, the scaling factor for the softmax.
    :param epsilon_exploration_rule: function mapping each positive integer to the exploitation epsilon
        of the corresponding episode.
        If epsilon_exploration_rule is not None, at episode number n during training the action
        with the highest Q-value is taken with probability (1-epsilon_exploration_rule(n))
    :param initial_q: initialization value of all Q-values
    :return:
        - Q: empirical estimates of the Q-values
        - stats: dictionary of statistics collected during training, with keys 'episode_rewards' and 'episode_lengths',
            and the corresponding values being a list (of length num_episodes) containing, for each episode, the reward 
            collected and the length of the episode respectively.
    """
    # Q-values map
    # Dictionary that maps the tuple representation of the state to a dictionary of action values
    # Q = defaultdict(lambda: initial_q * np.ones(env.get_num_actions()))  # All Q-values are initialized to initial_q
    a,b = env._horizontal_depth, env._vertical_depth
    Q = {}
    for y in range(b):
        Q[(0,y)] = {"u":initial_q, "d":initial_q, "l":initial_q, "r":initial_q}
    for x in range(-a,a+1):
        Q[(x,b)] = {"u":initial_q, "d":initial_q, "l":initial_q, "r":initial_q}


    # Stats of training
    episode_rewards = np.empty(num_episodes)
    episode_lengths = np.empty(num_episodes)

    for itr in range(num_episodes):

        env.reset()

        # re-initialize the eligibility traces
        egilibility_traces = {}
        for y in range(b):
            egilibility_traces[(0,y)] = {"u":0., "d":0., "l":0., "r":0.}
        for x in range(-a,a+1):
            egilibility_traces[(x,b)] = {"u":0., "d":0., "l":0., "r":0.}
        
        episode_length = 0
        while not env.end:
            # rescale all traces
            for S in Q.keys():
                for action in ["u","d","l","r"]:
                    egilibility_traces[S][action] *= trace_decay
                
            # current state:
            S = env.get_state()
            # choose action according to the desired policy
            if action_policy=="epsilon_greedy":
                A = epsilon_greedy(env, epsilon_exploration, Q)
            else:
                beta = epsilon_exploration
                A = softmax_(env, beta, Q)

            # move according to the policy
            S_, R = env.do_action(A)
            # update trace of current state action pair
            egilibility_traces[S][A] += 1

            # compute the target
            # Hint: all Q-values for fictitious state-action pairs are set to zero by convention
            """
            new_possible_actions = env.available() 
            Q_vals = np.array([Q[S_][action] for action in new_possible_actions])
            Q_ = np.max(Q_vals)
            A_ = new_possible_actions[np.argmax(Q_vals)]
            """
            if action_policy=="epsilon_greedy":
                A_ = epsilon_greedy(env, epsilon_exploration, Q)
            else:
                A_ = softmax_(env, beta, Q)
            
            Q_ = Q[S_][A_]

            # update all Q-values
            for state in Q.keys():
                for action in ["u","d","l","r"]:
                    Q[state][action] += alpha*(R + gamma*Q_ - Q[S][A])*egilibility_traces[state][action]
            #print(R)
            #print(Q)
            # prepare for the next move
            episode_length += 1 

        # save reward of the current episode
        episode_rewards[itr] = env.reward()
        # save length of the current episode
        episode_lengths[itr] = episode_length
    # save stats
    stats = {'episode_rewards': episode_rewards, 'episode_lengths':episode_lengths}
    return Q, stats



def q_learning(env, alpha=0.05, gamma=0.99, num_episodes=1000, action_policy="epsilon_greedy", epsilon_exploration=0.1,
               epsilon_exploration_rule=None, trace_decay=0, initial_q=0):
    """
    Trains an agent using the Q-Learning algorithm by playing num_episodes games until the reward states are reached
    :param env: environment
    :param alpha: learning rate
    :param gamma: discount rate for future rewards
    :param num_episodes: number of training episodes
    :param action_policy: string for the action policy to be followed during training
        It is usually "epsilon_greedy" or "softmax_"
    :param trace_decay: trace decay factor for eligibility traces
        If 0, q_learning(0) is implemented without any eligibility trace. If a non-zero float is given in input
        the latter represents the trace decay factor and q_learning(lambda) is implemented
    :param epsilon_exploration: parameter of the exploration policy: exploration rate or softmax_ scaling factor
        If action_policy is "epsilon_greedy":
            If epsilon_exploration_rule is None, at each iteration the action with the highest Q-value
            is taken with probability (1-epsilon_exploration)
        If action_policy is "softmax_":
            epsilon_exploration is actually beta, the scaling factor for the softmax.
    :param epsilon_exploration_rule: function mapping each positive integer to the exploitation epsilon
        of the corresponding episode.
        If epsilon_exploration_rule is not None, at episode number n during training the parameter for the
        exploration policy is epsilon_exploration_rule(n).
    :param initial_q: initialization value of all Q-values
    :return:
        - Q: empirical estimates of the Q-values
        - stats: dictionary of statistics collected during training, with keys 'episode_rewards' and 'episode_lengths',
            and the corresponding values being a list (of length num_episodes) containing, for each episode, the reward
            collected and the length of the episode respectively.
    """

    # Q-values map
    # Dictionary that maps the tuple representation of the state to a dictionary of action values
    a,b = env._horizontal_depth, env._vertical_depth
    Q = {}
    for y in range(b):
        Q[(0,y)] = {"u":initial_q, "d":initial_q, "l":initial_q, "r":initial_q}
    for x in range(-a,a+1):
        Q[(x,b)] = {"u":initial_q, "d":initial_q, "l":initial_q, "r":initial_q}

    # Stats of training
    episode_rewards = np.empty(num_episodes)  # reward obtained for each episode
    episode_lengths = np.empty(num_episodes)  # length for each training episode

    for itr in range(num_episodes):

        env.reset()

        # re-initialize the eligibility traces
        egilibility_traces = {}
        for y in range(b):
            egilibility_traces[(0,y)] = {"u":0., "d":0., "l":0., "r":0.}
        for x in range(-a,a+1):
            egilibility_traces[(x,b)] = {"u":0., "d":0., "l":0., "r":0.}
        
        episode_length = 0
        while not env.end:
            # rescale all traces
            for S in Q.keys():
                for action in ["u","d","l","r"]:
                    egilibility_traces[S][action] *= trace_decay
            
            # current state:
            S = env.get_state()
            
            # choose action according to the desired policy
            if action_policy=="epsilon_greedy":
                A = epsilon_greedy(env, epsilon_exploration, Q)
            else:
                beta = epsilon_exploration
                A = softmax_(env, beta, Q)


            # move according to the policy
            S_, R = env.do_action(A)

            # update trace of current state action pair
            egilibility_traces[S][A] += 1

            # compute the target
            # Hint: all Q-values for fictitious state-action pairs are set to zero by convention
            A_ = epsilon_greedy(env, 0, Q)
            Q_ = Q[S_][A_]
            
            # update all Q-values
            for state in Q.keys():
                for action in ["u","d","l","r"]:
                    Q[state][action] += alpha*(R + gamma*Q_ - Q[S][A])*egilibility_traces[state][action]

            # prepare for the next move
            episode_length += 1 

        # save reward of the current episode
        episode_rewards[itr] = env.reward()
        # save length of the current episode
        episode_lengths[itr] = episode_length

    # save stats
    stats = {'episode_rewards': episode_rewards, 'episode_lengths':episode_lengths}
    return Q, stats


def n_step_sarsa(env, alpha=0.05, gamma=0.99, num_episodes=1000, action_policy="epsilon_greedy", n=1,
                 epsilon_exploration=0.5, epsilon_exploration_rule=None, initial_q=0):
    """
    Trains an agent using the Sarsa algorithm by playing num_episodes games until the reward states are reached
    :param env: the environment
    :param alpha: learning rate
    :param gamma: discount rate for future rewards
    :param num_episodes: number of training episodes
    :param action_policy: string for the action policy to be followed during training
        It is usually "epsilon_greedy" or "softmax_"
    :param n: for n = 1 standard Sarsa(0) is recovered, otherwise n-step Sarsa is implemented
    :param epsilon_exploration: parameter of the exploration policy: exploration rate or softmax_ scaling factor
        If action_policy is "epsilon_greedy":
            If epsilon_exploration_rule is None, at each iteration the action with the highest Q-value
            is taken with probability (1-epsilon_exploration)
        If action_policy is "softmax_":
            epsilon_exploration is actually beta, the scaling factor for the softmax.
    :param initial_q: initialization value of all Q-values
    :return:
        - Q: empirical estimates of the Q-values
        - stats: dictionary of statistics collected during training, with keys 'episode_rewards' and 'episode_lengths',
            and the corresponding values being a list (of length num_episodes) containing, for each episode, the reward
            collected and the length of the episode respectively.
    """
    # Q-values map
    # Dictionary that maps the tuple representation of the state to a dictionary of action values
    a,b = env._horizontal_depth, env._vertical_depth
    Q = {}
    for y in range(b):
        Q[(0,y)] = {"u":initial_q, "d":initial_q, "l":initial_q, "r":initial_q}
    for x in range(-a,a+1):
        Q[(x,b)] = {"u":initial_q, "d":initial_q, "l":initial_q, "r":initial_q}

    # Stats of training
    episode_rewards = np.empty(num_episodes)
    episode_lengths = np.empty(num_episodes)

    # Hint: it may be useful to compute the weight of the rewards, something like
    reward_weights = np.array([gamma ** i for i in range(n)])

    for itr in range(num_episodes):
        episode_length = 0

        env.reset()
        # initialize a queue for state action pairs and rewards of the current episode
        States = [env.get_state()]
        Actions = []
        Rewards = []

        while not env.end:
            # Move according to the policy
            if action_policy=="epsilon_greedy":
                A = epsilon_greedy(env, epsilon_exploration, Q)
            else:
                beta = epsilon_exploration
                A = softmax_(env, beta, Q)
                
            S_, R = env.do_action(A)
            Actions.append(A)
            States.append(S_)
            # Save obtained reward
            Rewards.append(R)
            

            # compute the target
            # Hint: all Q-values for fictitious state-action pairs are set to zero by convention
            if action_policy=="epsilon_greedy":
                A_ = epsilon_greedy(env, epsilon_exploration, Q)
            else:
                A_ = softmax_(env, beta, Q)
            
            Q_ = Q[S_][A_]
            
            # update Q-value of state-action pair which is n steps away from the reward
            t = len(States)
            tau = t - n -1
            if tau >= 0:
                G = np.sum([Rewards[i]*gamma**(i-tau) for i in range(tau, tau+n)])
                G += gamma**n * Q_
                Q[States[tau]][Actions[tau]] += alpha*(G - Q[States[tau]][Actions[tau]])
                
            # prepare next move
            episode_length += 1 

        # update remaining Q-values within n steps from the reward
        T = len(States)
        for i in range(T-n,T-1):
            G = 0
            G += np.sum([Rewards[j]*gamma**(j-i) for j in range(i, T-1)])
            Q[States[i]][Actions[i]] += alpha*(G - Q[States[i]][Actions[i]])

        # save reward of the current episode
        episode_rewards[itr] = env.reward()
        # save length of the current episode
        episode_lengths[itr] = episode_length

    # save stats
    stats = {'episode_rewards': episode_rewards, 'episode_lengths':episode_lengths}
    return Q, stats

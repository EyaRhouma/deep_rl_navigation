## `1- Learning Algorithm`
Agents use a policy to decide which actions to take within an environment. The primary objective of the learning algorithm is to find an optimal policy—i.e., a policy that maximizes the reward for the agent. Since the effects of possible actions aren't known in advance, the optimal policy must be discovered by interacting with the environment and recording observations. Therefore, the agent "learns" the policy through a process of trial-and-error that iteratively maps various environment states to the actions that yield the highest reward. This type of algorithm is called Q-Learning.

As for constructing the Q-Learning algorithm, the general approach is to implement a handful of different components, then run a series of tests to determine which combination of components and which hyperparameters yield the best results.

In the following sections, we'll describe each component of the algorithm in detail.

### Q-Function
To discover an optimal policy, I setup a Q-function. The Q-function calculates the expected reward R for all possible actions A in all possible states S.

We can then define our optimal policy π* as the action that maximizes the Q-function for a given state across all possible states. The optimal Q-function Q*(s,a) maximizes the total expected reward for an agent starting in state s and choosing action a, then following the optimal policy for each subsequent state.

In order to discount returns at future time steps, the Q-function can be expanded to include the hyperparameter gamma γ.

### Epsilon Greedy Algorithm
One challenge with the Q-function above is choosing which action to take while the agent is still learning the optimal policy. Should the agent choose an action based on the Q-values observed thus far? Or, should the agent try a new action in hopes of earning a higher reward? This is known as the exploration vs. exploitation dilemma.

To address this, I implemented an 𝛆-greedy algorithm. This algorithm allows the agent to systematically manage the exploration vs. exploitation trade-off. The agent "explores" by picking a random action with some probability epsilon 𝛜. However, the agent continues to "exploit" its knowledge of the environment by choosing actions based on the policy with probability (1-𝛜).

Furthermore, the value of epsilon is purposely decayed over time, so that the agent favors exploration during its initial interactions with the environment, but increasingly favors exploitation as it gains more experience. The starting and ending values for epsilon, and the rate at which it decays are three hyperparameters that are later tuned during experimentation.

You can find the 𝛆-greedy logic implemented as part of the agent.act() method here in agent.py of the source code.

### Deep Q-Network (DQN)
With Deep Q-Learning, a deep neural network is used to approximate the Q-function. Given a network F, finding an optimal policy is a matter of finding the best weights w such that F(s,a,w) ≈ Q(s,a).

The neural network architecture used for this project can be found here in the model.py file of the source code. The network contains three fully connected layers with 64, 64, and 4 nodes respectively. Testing of bigger networks (more nodes) and deeper networks (more layers) did not produce better results.

As for the network inputs, rather than feeding-in sequential batches of experience tuples, I randomly sample from a history of experiences using an approach called Experience Replay.

### Experience Replay
Experience replay allows the RL agent to learn from past experience.

Each experience is stored in a replay buffer as the agent interacts with the environment. The replay buffer contains a collection of experience tuples with the state, action, reward, and next state (s, a, r, s'). The agent then samples from this buffer as part of the learning step. Experiences are sampled randomly, so that the data is uncorrelated. This prevents action values from oscillating or diverging catastrophically, since a naive Q-learning algorithm could otherwise become biased by correlations between sequential experience tuples.

Also, experience replay improves learning through repetition. By doing multiple passes over the data, our agent has multiple opportunities to learn from a single experience tuple. This is particularly useful for state-action pairs that occur infrequently within the environment.

The implementation of the replay buffer can be found here in the agent.py file of the source code.

## `2- Code implementation`
The code used here is derived from the "Lunar Lander" tutorial from the Deep Reinforcement Learning Nanodegree, and has been slightly adjusted for being used with the banana environment.

The code consist of :

- **model.py** : In this python file, a PyTorch QNetwork class is implemented. This is a regular fully connected Deep Neural Network using the PyTorch Framework. This network will be trained to predict the action to perform depending on the environment observed states. This Neural Network is used by the DQN agent and is composed of :
    - the input layer which size depends of the state_size parameter passed in the constructor
    - 2 hidden fully connected layers of 1024 cells each
    - the output layer which size depends of the action_size parameter passed in the constructor

- **dqn_agent.py** : In this python file, a DQN agent and a Replay Buffer memory used by the DQN agent) are defined.
    - The DQN agent class is implemented, as described in the Deep Q-Learning algorithm. It provides several methods :
        - constructor :
        Initialize the memory buffer (Replay Buffer)
        Initialize 2 instance of the Neural Network : the target network and the local network
        - step() :
        Allows to store a step taken by the agent (state, action, reward, next_state, done) in the Replay Buffer/Memory
        Every 4 steps (and if their are enough samples available in the Replay Buffer), update the target network weights with the current weight values from the local network (That's part of the Fixed Q Targets technique)
        - act() which returns actions for the given state as per current policy (Note : The action selection use an Epsilon-greedy selection so that to balance between exploration and exploitation for the Q Learning)
        - learn() which update the Neural Network value parameters using given batch of experiences from the Replay Buffer.
        - soft_update() is called by learn() to softly updates the value from the target Neural Network from the local network weights (That's part of the Fixed Q Targets technique)
    - The ReplayBuffer class implements a fixed-size buffer to store experience tuples (state, action, reward, next_state, done)
        - add() allows to add an experience step to the memory
        - sample() allows to randomly sample a batch of experience steps for the learning

- **Navigation.ipynb** : This Jupyter notebooks allows to train the agent. More in details it allows to :
    - Import the Necessary Packages
    - Examine the State and Action Spaces
    - Take Random Actions in the Environment (No display)
    - Train an agent using DQN
    - Plot the scores
    
    
## `3- DQN parameters`
The DQN agent uses the following parameters values (defined in dqn_agent.py)

- BUFFER_SIZE = int(1e5)    # replay buffer size
- BATCH_SIZE = 64           # minibatch size 
- GAMMA = 0.995             # discount factor 
- TAU = 1e-3                # for soft update of target parameters
- LR = 5e-4                 # learning rate 
- UPDATE_EVERY = 4          # how often to update the network
- eps_decay = 0.98
- eps_end = 0.02
## `4- DQN Results`
Given the chosen architecture and parameters, our results are :


## `5- Ideas for future work`
A further evolution to this project would be to train the agent directly from the environment's observed raw pixels instead of using the environment's internal states (37 dimensions)

To do so a Convolutional Neural Network would be added at the input of the network in order to process the raw pixels values (after some little preprocessing like rescaling the image size, converting RGB to gray scale, ...)

Other enhancements might also be implemented to increase the performance of the agent:

### Double DQN
The popular Q-learning algorithm is known to overestimate action values under certain conditions. It was not previously known whether, in practice, such overestimations are common, whether they harm performance, and whether they can generally be prevented. In this paper, we answer all these questions affirmatively. In particular, we first show that the recent DQN algorithm, which combines Q-learning with a deep neural network, suffers from substantial overestimations in some games in the Atari 2600 domain. We then show that the idea behind the Double Q-learning algorithm, which was introduced in a tabular setting, can be generalized to work with large-scale function approximation. We propose a specific adaptation to the DQN algorithm and show that the resulting algorithm not only reduces the observed overestimations, as hypothesized, but that this also leads to much better performance on several games.

### Dueling DQN
In recent years there have been many successes of using deep representations in reinforcement learning. Still, many of these applications use conventional architectures, such as convolutional networks, LSTMs, or auto-encoders. In this paper, we present a new neural network architecture for model-free reinforcement learning. Our dueling network represents two separate estimators: one for the state value function and one for the state-dependent action advantage function. The main benefit of this factoring is to generalize learning across actions without imposing any change to the underlying reinforcement learning algorithm. Our results show that this architecture leads to better policy evaluation in the presence of many similar-valued actions. Moreover, the dueling architecture enables our RL agent to outperform the state-of-the-art on the Atari 2600 domain.

### Prioritized experience replay
Experience replay lets online reinforcement learning agents remember and reuse experiences from the past. In prior work, experience transitions were uniformly sampled from a replay memory. However, this approach simply replays transitions at the same frequency that they were originally experienced, regardless of their significance. In this paper we develop a framework for prioritizing experience, so as to replay important transitions more frequently, and therefore learn more efficiently. We use prioritized experience replay in Deep Q-Networks (DQN), a reinforcement learning algorithm that achieved human-level performance across many Atari games. DQN with prioritized experience replay achieves a new state-of-the-art, outperforming DQN with uniform replay on 41 out of 49 games.

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Navigation

## Project description

For this project, the task is to train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

- **State space** has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  

- **Action space** is 4 dimentional. Four discrete actions are available, corresponding to:
    - **`0`** - move forward.
    - **`1`** - move backward.
    - **`2`** - turn left.
    - **`3`** - turn right.

- **Reward** A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

- **Solution criteria**  The task is episodic, and in order to solve the environment, the agent must get an average score of +13 in fewer than 1800 episodes.

## Getting Started
 
### Installation requirements

1. Configure a Python 3.6 / PyTorch 0.4.0 environment with the needed requirements as described in the Udacity repository

2. Install "unityagents" [click here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)

3. Download the Banana environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Finally, unzip the environment archive in the 'project's environment' directory and eventually adjust thr path to the UnityEnvironment in the code.


## Train The agent
Execute the provided notebook Navigation.ipynb (The headless / no visualization version of the Unity environment was used)
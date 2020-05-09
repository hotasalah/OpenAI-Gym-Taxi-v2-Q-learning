import numpy as np
import gym
import random

# step1: Create the environmemt
env = gym.make("taxi-v2")
env.render()

# step 2: Create the Q-table and initialize it
action_size = env.action_space.n
print(action_size)

state_size = env.opservation_space.n
print(state_size)

q_table = np.zeros((action_size, state_size))
print(q_table)

# step 3: Create the hyberparameters
total_episodes = 5000       # total number of episodes
total_test_episodes = 100   # total number of test episodes
max_steps = 99              # max steps per episode

learning_rate = 0.7
gamma = 0.618

# Exploration parameteers
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

# step 4: The Q-Learning algorithm

# 1 initialize Q-values arbitary (Q(s,a)) for all state action pairs
# (done is step 2)

# 2 For life or ynyil searningisstopped (all episods)
for episode in range(total_episodes):
    #reset the environment
    state = env.reset()
    step = 0
    done = False
    
    for step in range(max_steps):
        # 3 choose an action a in the curren world state (s)
        # first we randomize a number to emplement Exponrntial-Exploitation trade-off
        exp_exp_tradeoff = random.uniform(0, 1)
        
        # if this number is > than epsilon  --> Exploitation (taking the action with the biggest Q value for this state)
        if exp_exp_tradeoff < epsilon :
            action = np.argmax(q_table[state, :])
            
        # else doing a nandom choice (action) --> Exploration
        if exp_exp_tradeoff > epsilon :
            action = env.action_space.sample()
            
        # 4 Take the action (a) and observe the outcome state (s') and reward (r)
        new_state, reward, done, info = env.step(action)
        
        # 5 update Q(s,a) := Q(s,a) + Lr( R(s,a) + gamma*maxQ(s',a') - Q(s.a) )
        q_table[state, action] = q_table[state, action] + learning_rate*( reward +
                                   gamma*np.max(q_table[new_state, :]) - q_table[state, action] )

        # new state is new_state
        state = new_state
        
        # if done : finish episode
        if done == True :
            break
        
    episode += 1
        
    # reduce epsilon (cause we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)

# step 5: Train the taxi on the Q-table
env.reset()
rewards = []

for episode in range(total_test_episodes):

    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    
    print("********************************")
    print("EPISODE: ", episode)
    
    for step in range(max_steps):
        env.render()
        # Take the action (index) that havethe maximum expected future rewardgiven that state
        action = np.argmax(q_table[state, :])
        
        new_state, reward, done, info = env.step(action)
    
        total_rewards += reward
        
        if done:
            rewards.append(total_rewards)
            print("Score: ", total_rewards)
            break
        
        state = new_state
        
env.close()
print("Score over time: " + str( sum(rewards) / total_test_episodes))
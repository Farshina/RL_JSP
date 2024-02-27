import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from random import randint
import pickle
import copy
#from geneticalgorithm import geneticalgorithm as ga



class Field_class:
    def __init__(self, height, width):
        self.width      = int(np.ceil(width))
        self.height     = int(np.ceil(height))
        self.body = np.zeros(shape=(self.height, self.width))
    
    #update field with all its players
    def update_field(self, players):
        try:
            # Clear the field:
            self.body = np.zeros(shape=(self.height, self.width))            
            
            # Put the players on the field
            for i in range(len(players)):
                player = players[i]
                self.body[player.y:player.y+player.height,
                              player.x:player.x+player.width] += player.body ### y is constant throughout the game
        except :
            pass
        
class Player_class:
   
    def __init__(self, height, width, x, y, speed, left_bound, right_bound):
        self.height        = int(np.ceil(height))
        self.width         = int(np.ceil(width))
        self.x             = x
        self.y             = y
        self.speed         = speed
        self.body_unit     = 1 # 1 means player is there
        self.body          = np.ones(shape = (self.height, self.width))*self.body_unit
        self.left_bound = left_bound
        self.right_bound = right_bound

    def move(self, direction):
        '''
        Moves the player :
         - left, if direction  = 0
         - right, if direction = 1
        '''
        val2dir   = {0:-1 , 1:1}
        direction = val2dir[direction]
        next_x = (self.x + self.speed*direction)
        out_flag = False
        # check if out-of-bound
        if (next_x + self.width > self.right_bound or next_x < self.left_bound):  
            out_flag = True
        else:
            self.x += self.speed*direction    
        return out_flag


class Environment:
    def __init__(self, job_instance, max_height, max_time, min_time):
        
        # load job instance data
        self.job_instance = job_instance
        self.max_height = max_height
        self.max_time = max_time
        self.min_time = min_time
        
        #generate field
        self.F_HEIGHT      = int(np.ceil(self.max_height)) # Height of the field
        self.F_WIDTH       = int(np.ceil(self.max_time)) # Width of the field
        
        # generate players
        self.n_players = len(self.job_instance) # total number of players
        
        self.ACTION_SPACE      = [0,1]
        self.ACTION_SPACE_SIZE = len(self.ACTION_SPACE)
        self.episode_global_best_score = np.float64('Inf')
        self.current_state = self.reset()
         
        
    def reset(self):
        
        self.game_over      = False
        self.field = Field_class(height=self.F_HEIGHT, width=self.F_WIDTH )
        self.prev_max_total_power = 0
        
        self.players = [] 
        self.MAX_VAL = self.n_players # worst case, all jobs overlap with each other
        self.episode_global_best_score = np.float64('Inf')
        
        for p in range(self.n_players):
            P_HEIGHT      = self.job_instance[p]['job_height']  # Height of the player
            P_WIDTH       = self.job_instance[p]['duration']  # Width of the player
            P_left_bound = self.job_instance[p]['release'] # Left-bound for the player, will be changed for successors, predecessors
            P_right_bound = self.job_instance[p]['deadline'] # Right-bound for the player, will be changed for successors, predecessors
            
            # if y = 0 the whole thing is inverted, x is a random position in between the bound
            self.players.append(Player_class( height = P_HEIGHT,   width =  P_WIDTH,
                                    x = randint(P_left_bound,P_right_bound-P_WIDTH),
                                    y = 0, speed = 1, left_bound = P_left_bound, right_bound = P_right_bound))

        # Update the field :
        self.field.update_field(self.players)
        
        #observation = self.field.body
        observation = [i for i in self.field.body[0]]
        return observation
    
    
    def find_max_total_power(self):
        '''
        for each column along x axis, calculate the total height (power consumed)
        select max(total overlap)
        '''
        max_total_height = 0
        for i in range(self.field.width):
            temp_height = 0
            for j in range(self.field.height):
                temp_height = temp_height + self.field.body[j][i] # body[j][i] : all row single column (check!!!!!!!!!!!)
            if temp_height > max_total_height:
                max_total_height = temp_height 
        return max_total_height
    
    def step(self, player_id, action):

        reward = 0

        #get player
        player = self.players[player_id]
        
        # move player
        out_flag = None
        out_flag = player.move(action)    

        # Update the field :
        self.field.update_field(self.players)           

        # Return New Observation , reward, game_over(bool)
        current_max_total_power =  self.find_max_total_power()
        #'''
        if self.prev_max_total_power == 0 : #just initialized, otherwise max_power > 0
            reward = 0
        else:
            if out_flag == True:
                #reward = -10
                reward = 0
            else:
                if current_max_total_power < self.prev_max_total_power:
                    reward = +1
                
    
        #'''
        if current_max_total_power < self.episode_global_best_score:
            if self.episode_global_best_score != np.float64('Inf'): #############################check
                reward = 100
            self.episode_global_best_score = current_max_total_power
            template = "episode_global_best_score {}"
            print(template.format(self.episode_global_best_score))

        self.prev_max_total_power = current_max_total_power
        self.MAX_VAL = max(max(item) for item in self.field.body) #max overlap
        
        observation = [i for i in self.field.body[0]]
        return observation, reward, current_max_total_power, self.episode_global_best_score

# pre-process data
saveddata = pickle.load(open("./test_job_instance_data_selected_bin.pkl", "rb"))
additional_info = saveddata[0]
job_instance = saveddata[1]
max_height = saveddata[2]
max_time = saveddata[3]
min_time = saveddata[4]

# create environment
env = Environment(job_instance, max_height, max_time, min_time)

# Rendering variables
score_increased = False
game_over = False 

num_inputs = env.F_WIDTH
num_actions = env.ACTION_SPACE_SIZE * env.n_players  # (player_id,action) pair
# define model parameters
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 50
max_episode = 5
episode_interval = 1
# episode_interval = int(max_episode/20)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
num_hidden = [10]


# create model
inputs = layers.Input(shape=(num_inputs,))

''' # 2 hidden layers
common_actor1 = layers.Dense(num_hidden[0], activation="relu")(inputs)
common_actor2 = layers.Dense(num_hidden[1], activation="relu")(common_actor1)
action_out = layers.Dense(num_actions, activation="softmax")(common_actor2)
model_actor = keras.Model(inputs=inputs, outputs=action_out)

common_critic1 = layers.Dense(num_hidden[0], activation="relu")(inputs)
common_critic2 = layers.Dense(num_hidden[1], activation="relu")(common_critic1)
critic_out = layers.Dense(1)(common_critic2)
model_critic = keras.Model(inputs=inputs, outputs=critic_out)
'''
#''' # 1 hidden layer
common_actor1 = layers.Dense(num_hidden[0], activation="relu")(inputs)
action_out = layers.Dense(num_actions, activation="softmax")(common_actor1)
model_actor = keras.Model(inputs=inputs, outputs=action_out)

common_critic1 = layers.Dense(num_hidden[0], activation="relu")(inputs)
critic_out = layers.Dense(1)(common_critic1)
model_critic = keras.Model(inputs=inputs, outputs=critic_out)
#'''
#optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1 = 0.999, beta_2 = 0.999, epsilon = 0.001)
optimizer = keras.optimizers.legacy.SGD(learning_rate=0.0001)
#optimizer = keras.optimizers.Adam()
huber_loss = keras.losses.Huber()

# create episode storage
action_probs_history = []
critic_value_history = []
reward_per_episode = []
running_reward = 0
episode_count = 0

#what to save in a file
average_maxpower = []
minimum_maxpower = []
count_maxpower_all_episodes = []
global_best_all_episodes = []
episode_reward_list = []
global_best_solution_list = []
global_best_score = np.float64("Inf")
check_instance = [] #this variable should 

def CountFrequency(my_list):
    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[int(item)] += 1
        else:
            freq[int(item)] = 1

    return freq

def Explore_environment(state,model_actor,model_critic,num_actions,env, episode_reward):
    state = tf.convert_to_tensor(state)
    state = tf.expand_dims(state, 0)

    # Predict action probabilities and estimated future rewards
    # from environment state
    action_probs = model_actor(state)
    critic_value = model_critic(state)
    # Sample action from action probability distribution
    selected_action = np.random.choice(num_actions, p=np.squeeze(
        action_probs))  # consider output nodes 0 theke (#num_actions-1)
    # extract player_id and action from the selected action
    player_id = int(selected_action / env.ACTION_SPACE_SIZE)
    action = selected_action % env.ACTION_SPACE_SIZE
    # Apply the sampled action in our environment
    state, reward, current_max_total_power, episode_global_best_score = env.step(player_id, action)
    episode_reward += reward
    
    return critic_value, action_probs, selected_action, reward, current_max_total_power,episode_reward, episode_global_best_score


for episode in range(max_episode):
    
    if episode % 1 == 0: 
        state = env.reset()
    
    episode_reward = 0
    max_power_per_episode = []
    global_best_per_episode = []
    with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape_actor, tf.GradientTape(
            persistent=False, watch_accessed_variables=False) as tape_critic:
        tape_actor.reset()
        tape_critic.reset()
        tape_actor.watch(model_actor.trainable_variables)
        tape_critic.watch(model_critic.trainable_variables)
        for timestep in range(1, max_steps_per_episode):
            critic_value, action_probs, selected_action, reward, current_max_total_power,episode_reward, episode_global_best_score = Explore_environment(state,model_actor,model_critic,num_actions,env,episode_reward)
            
            
            ########save the best solution
            if global_best_score > episode_global_best_score:
                global_best_score = episode_global_best_score
                temp_global_best = ["global_best_score", global_best_score]
                for p in range(env.n_players):
                    temp_global_best.append(env.players[p].x)
                if len(global_best_solution_list) > 0:
                    for index in range(len(global_best_solution_list)):
                        item = global_best_solution_list[index]
                        if global_best_score < item[1]: ###item[1] is the global best score stored
                            global_best_solution_list[index] = temp_global_best ######replace the worse solution by the best
                        elif global_best_score == item[1]:
                            global_best_solution_list.append(temp_global_best)
                else:
                    global_best_solution_list.append(temp_global_best) #if list is empty, append directly
                
            critic_value_history.append(critic_value[0, 0])
            action_probs_history.append(tf.math.log(action_probs[0, selected_action]))
            reward_per_episode.append(reward)
            global_best_per_episode.append(episode_global_best_score)
            max_power_per_episode.append(current_max_total_power)
            

        if episode % episode_interval == 0: #count at every #max_episode/20 th episode
            count_maxpower_per_episode = CountFrequency(max_power_per_episode)
            count_maxpower_all_episodes.append(count_maxpower_per_episode)
        
        min_max_power = min(max_power_per_episode)
        avg_max_power = np.mean(max_power_per_episode)
        max_power_per_episode.clear()

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in reward_per_episode[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)


        actor_losses = []
        critic_losses = []

        for log_prob, value, ret in history:
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up recieving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # actor loss
        
                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_losses.append(
                    huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )

        

        # Backpropagation
        loss_actor = sum(actor_losses)
        loss_critic = sum(critic_losses)

        # Clear the loss and reward history for each episode and store the values
        action_probs_history.clear()
        critic_value_history.clear()
        actor_losses.clear()
        critic_losses.clear()
        reward_per_episode.clear()
        returns.clear()
        del history

       
        grads_actor = tape_actor.gradient(loss_actor, model_actor.trainable_variables)
        optimizer.apply_gradients(zip(grads_actor, model_actor.trainable_variables))
        grads_critic = tape_critic.gradient(loss_critic, model_critic.trainable_variables)
        optimizer.apply_gradients(zip(grads_critic, model_critic.trainable_variables))

        # Log details
        template = "episode {}, episode reward: {}, Min Max power: {}, Avg Max power: {:.2f} "
        print(template.format(episode_count, episode_reward, min_max_power,
                              avg_max_power))
        episode_count += 1

        del tape_actor
        del tape_critic
        
        average_maxpower.append(avg_max_power)
        minimum_maxpower.append(min_max_power)
        episode_reward_list.append(episode_reward)
        global_best_all_episodes.append(global_best_per_episode)
        info = ["episode_reward, minimum_maxpower, average_maxpower, max_steps_per_episode, num_hidden, count_maxpower_all_episodes, global_best_all_episodes, global_best_solution_list, info"]
        savedresults = [episode_reward_list, minimum_maxpower, average_maxpower, max_steps_per_episode, num_hidden, count_maxpower_all_episodes, global_best_all_episodes, global_best_solution_list, info]
        pickle.dump(savedresults, open("actor_critic_selected_bin_test_results.pkl", "wb"))  # will be overwrite in every episode
        
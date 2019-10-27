import numpy as np 
import random
import gym
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam
import os


class Agent:

    def __init__(self, state_size, action_size, lr, eps, gamma, eps_min, eps_decay):
        self.learning_rate = lr
        self.action_size = action_size
        self.state_size = state_size
        self.eps = eps
        self.gamma = gamma
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.memory = deque(maxlen=2000)
        self.model = self.create_model(self.state_size, self.action_size, self.learning_rate)

    def create_model(self, ss, acts, lr):
        model = Sequential()
        model.add(Dense(24,input_dim=ss, activation='relu'))
        model.add(Dense(24,activation='relu'))
        model.add(Dense(acts, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=lr))
        return model 

    def action(self, st):
        if np.random.rand() <= self.eps:
            return random.randrange(self.action_size)
        act_values = self.model.predict(st)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        mb = random.sample(self.memory, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in mb:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            states.append(state[0])
            targets.append(target_f[0])
        history = self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)            

        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
        return history.history['loss'][0]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        
        
def main():
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size, action_size, 0.001, 1.0, 0.95, 0.01, 0.9995)
    if os.path.exists("./save/cartpole-master.h5"):
        agent.load("./save/cartpole-master.h5")
    elif not os.path.exists("./save/"):
        os.mkdir("./save/")
    done = False
    episodes = 1000
    batch_size = 32

    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for t in range(500):
            #env.render()
            action = agent.action(state)
            next_state, reward, done, _ = env.step(action)
            if done:
                print("Episode: {}/{}, Score: {}, Epsilon: {:.2}".format(episode+1, episodes, t, agent.eps))
                reward = -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)       
            if (episode+1) % 100 == 0 and t == 0:
                print("Episode: {}/{}, loss: {:.4f}"
                    .format(episode+1, episodes, loss))                 

        if not done:
            print("Episode Final: {}/{}, Score: {}, Epsilon: {:.2}".format(episode+1, episodes, t, agent.eps))
        if (episode+1) % 100 == 0:
            agent.save("./save/cartpole-dqn.h5")


if __name__ == "__main__":
    main()

import numpy as np
import math
import random
from gym import Env
from gym.spaces import Discrete, MultiDiscrete
import tensorflow as tf
import tf_agents as tfa
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent

# pip install tensorflow
# pip install tf_agents (this should install gym)
# pip install gym

def get_primes(n):
    primes = np.array([], dtype=int)
    for i in range(2, n+1):
        if np.all(i % primes != 0):
            primes = np.append(primes, i)
    return primes


def main():
    s = get_primes(1000)
    print(s)

    # side=right means strictly less than value on right
    p = s.searchsorted(300, side='right')
    print(f'index={p} value={s[p]}')
    print(s[p-2:p+3])

class Game(Env):

    def __init__(self):
        self.board = None
        self.player = 1
        self.score = 0
        self.lines = self.line_masks()
        self.reset()
        # init Env

        # a single value with 9 discrete states
        # use action_space.sample() and observation_space.sample() to check
        self.action_space = Discrete(9)
        # 9 values with 3 discrete states
        a = np.zeros(9).astype(int)
        a.fill(3)
        self.observation_space = MultiDiscrete(a, dtype=int)
        self.state = self.board

    def __str__(self) -> str:
        a = np.reshape(self.board, (3,3))
        return str(a)
    
    def reset(self):
        self.player = 1
        self.score = 0
        self.board = np.zeros(9).astype(int)
    
    def step(self, action):
        """Gym.step make the player move/action then a random response. return the reward and whether the game is complete."""
        moves = self.valid_moves()
        if action in moves:
            self.make_move(action)
        
        if self.is_not_complete():
            self.next_move()

        reward = 1 if self.score == 3 else 0
        if self.score == -3:
            reward = -1
        done = not self.is_not_complete()
        return self.board, reward, done, {}
        
    def render(self):
        pass

    def line_masks(self):
        xs = [
            [0,1,2],
            [3,4,5],
            [6,7,8],
            [0,3,6],
            [1,4,7],
            [2,5,8],
            [0,4,8],
            [2,4,6]
            ]
        return np.array([[y in x for y in range(0,9)] for x in xs])

    def calculate_score(self):
        mx = 0
        ys = [sum(self.board[x]) for x in self.lines]
        mx = np.max(ys)
        mn = np.min(ys)
        return mx if mx > abs(mn) else mn

    def valid_moves(self):
        return np.where(self.board == 0)[0]

    def choose_random_move(self):
        moves = self.valid_moves()
        return np.random.choice(moves)

    def make_move(self, mv):
        self.board[mv] = self.player
        self.player *= -1
        self.score = self.calculate_score()
        return self.score
       
    def is_not_complete(self):
        return not np.all(self.board) and abs(self.score) < 3
    
    def next_move(self):
        return self.make_move(self.choose_random_move())
        #print(self.board, self.score)
    
# --- end class Game ---

def build_model(states, actions):
    model = Sequential()
    model.add(Dense(81, activation=keras.activations.relu, input_shape = states))
    model.add(Dense(36, activation=keras.activations.relu))
    model.add(Dense(actions, activation=keras.activations.softmax))
    print(model.summary())
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model = model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

def play_random():
    game = Game()
    while game.is_not_complete():
        game.next_move()

    if game.score == 3:
        print("player 1 wins")
    elif game.score == -3:
        print("player 2 wins")
    else:
        print("game drawn")
    print(game)

if __name__ == "__main__":
    print("TensorFlow version ", tf.__version__)
    print("TF agents version ", tfa.__version__)
    game = Game()
    pyEnv = tfa.environments.suite_gym.wrap_env(game)
    states = game.observation_space.shape
    actions = game.action_space.n
    model = build_model(states, actions)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2),
                  loss=keras.losses.Huber(),
                  metrics=['mae'])
    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.fit(game, nb_steps=50000, visualize=False, verbose=1)
    # todo - is the vidoe right or the example in the docs https://keras.io/examples/rl/deep_q_network_breakout/

    stats = np.zeros(3).astype(int)
    for i in range(100):
        done = False
        reward = 0
        game.reset()
        while not done:
            action = game.action_space.sample()
            state, reward, done, x = game.step(action)
        # count the rewards
        stats[reward + 1] = stats[reward + 1] + 1
    
    print(stats)

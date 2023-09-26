from typing import Tuple
from jax import random
from jax import numpy as jnp
from tqdm import tqdm
from env import Easy21

N0 = 100
EPISODES = 1000

S = jnp.zeros((22, 11, 2))
Q = jnp.zeros((22, 11, 2))

key = random.PRNGKey(42)


def NSA(s, a):
    """Number of times action a has been selected from state s"""
    dealer, player, _ = s
    return S[dealer, player, a]


def NS(s):
    """Number of times state s has been visited"""
    dealer, player, _ = s
    return jnp.sum(S[dealer, player])


def epsilon(s):
    """Epsilon(t) = N0/(N0 + N (st))"""
    return N0 / (N0 + NS(s))


def alpha(s, a):
    """Step-size alpha(t) = 1/N(st, at)"""
    return 1 / NSA(s, a)


def epsilon_greedy(state: Tuple[int, int, bool], actions: Tuple[int, int]) -> int:
    """Epsilon-greedy action function"""
    if random.uniform(key) < epsilon(state):
        return random.choice(key=key, a=jnp.array(actions))
    else:
        return jnp.argmax(
            jnp.array([Q[state[0], state[1], action] for action in actions])
        )


easy21 = Easy21()

eps = tqdm(range(EPISODES))
for ep in eps:
    eps.set_description(f"Running episode {ep}: ")
    state = easy21.start()

    history = []

    while state[2] is False:
        action = epsilon_greedy(state, Easy21.actions())
        dealer, player, _ = state
        S = S.at[dealer, player, action].add(1)
        next_state, reward = easy21.step(state, action)
        history.append((state, action, reward))
        state = next_state

    G = jnp.sum(jnp.array([sar[-1] for sar in history]))
    for state, action, reward in history:
        (dealer, player, _) = state
        Q = Q.at[dealer, player, action].add(
            alpha(state, action) * (G - Q[dealer, player, action])
        )

print(Q)

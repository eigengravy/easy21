from typing import Tuple
from jax import random


class Easy21:
    def __init__(self):
        self.key = random.PRNGKey(24)

    def start(self) -> Tuple[int, int, bool]:
        key, a = random.split(self.key)
        self.key, b = random.split(key)

        return (
            random.randint(a, (), 1, 11),
            random.randint(b, (), 1, 11),
            False,
        )

    @staticmethod
    def actions() -> Tuple[int, int]:
        return (0, 1)

    def draw(self) -> int:
        key, a = random.split(self.key)
        self.key, b = random.split(key)

        return random.randint(a, (), 1, 11) * (-1 if random.uniform(b) <= 1 / 3 else 1)

    def step(
        self, state: Tuple[int, int, bool], action: int
    ) -> (Tuple[int, int, bool], int):
        dealer, player, terminal = state
        assert 1 <= dealer <= 10
        assert 1 <= player <= 21
        assert terminal is False
        assert action in self.actions()

        if action == 0:
            player += self.draw()
            bust = not 1 <= player <= 21
            return (dealer, player, bust), -1 if bust else 0
        else:
            while dealer < 17:
                dealer += self.draw()
            bust = not 1 <= dealer <= 21
            if bust or player > dealer:
                return (dealer, player, True), 1
            elif player < dealer:
                return (dealer, player, True), -1
            else:
                return (dealer, player, True), 0

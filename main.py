from env import Easy21


game = Easy21()
state = [game.start()]
print(state)
while not state[0][2]:
    state = game.step(state[0], 0)
    print(state)

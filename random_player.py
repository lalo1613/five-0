import random

class RandomPlayer():
    def __init__(self):
        None

    def get_choice(self, level, turn, player, deck, pot_card):
        options = [x for x in range(5) if player[turn][level][x] == -1]
        decision = random.sample(options, 1)[0]
        return decision


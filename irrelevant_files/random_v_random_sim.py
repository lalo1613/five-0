import numpy as np
import pandas as pd
import random
from random_player import RandomPlayer
from tqdm import tqdm

def calculate_winner(game_board):
    hands = ["High Card", "Pair", "Two Pair", "Three of a Kind", "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"]
    cardNames = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    handValueCounts = [[[0 for x in range(13)] for y in range(5)] for z in range(2)]
    handSuited =[[0 for x in range(5)] for y in range(2)]
    handValues = [[[0 for x in range(5)] for y in range(5)] for z in range(2)]
    handLabels = [[0 for x in range(5)] for y in range(2)]
    handCode = [[0 for x in range(5)] for y in range(2)]  # first number - hand type, next 2 - hand parameters, last 5 - sorted hand values
    for p in range(2):
        for c in range(5):
            handSuited[p][c] = (game_board[p][0][c] // 13 == game_board[p][1][c] // 13 == game_board[p][2][c] // 13 == game_board[p][3][c] // 13 == game_board[p][4][c] // 13)
            for r in range(5):
                handValueCounts[p][c][game_board[p][r][c] % 13] += 1
                handValues[p][c][r] = game_board[p][r][c] % 13

    for p in range(2):
        for c in range(5):
            handValues[p][c].sort(reverse=True)
            m = max(handValueCounts[p][c])
            if (m == 4):
                handLabels[p][c] = hands[7] + " " + cardNames[handValueCounts[p][c].index(m)] + "s"
                handCode[p][c] = [7, handValueCounts[p][c].index(m), handValueCounts[p][c].index(m)]+handValues[p][c]
            if (m == 3):
                ap = -1 # looking for additional pair, which would give a full house
                for i in range(13):
                    if handValueCounts[p][c][i] == 2:
                        ap = i
                if(ap>-1):
                    handLabels[p][c] = hands[6] + " " + cardNames[handValueCounts[p][c].index(m)] + "s and " + cardNames[ap] + "s"
                    handCode[p][c] = [6, handValueCounts[p][c].index(m), ap] + handValues[p][c]
                else:
                    handLabels[p][c] = hands[3] + " " + cardNames[handValueCounts[p][c].index(m)] + "s"
                    handCode[p][c] = [3, handValueCounts[p][c].index(m), handValueCounts[p][c].index(m)] + handValues[p][c]
            if (m == 2):
                tp = [] # looking for two pair
                for i in range(13):
                    if handValueCounts[p][c][i] == 2:
                        tp.append(i)
                if(len(tp)==2):
                    handLabels[p][c] = hands[2] + " " + cardNames[max(tp)] + "s and " + cardNames[min(tp)] + "s"
                    handCode[p][c] = [2, max(tp), min(tp)] + handValues[p][c]
                else:
                    handLabels[p][c] = hands[1] + " of " + cardNames[tp[0]] + "s"
                    handCode[p][c] = [1, tp[0], tp[0]] + handValues[p][c]
            if (m == 1):
                count = 0
                last = False
                l = list(range(13))
                l.insert(0, 12)
                for i in l:
                    if handValueCounts[p][c][i] == 1:
                        if last is False:
                            last = True
                            count = 1
                        else:
                            count += 1
                    else:
                        last = False
                if(count==5):
                    if(handSuited[p][c]):
                        handLabels[p][c] = hands[8] + " " + cardNames[max(handValues[p][c])] + " High"
                        handCode[p][c] = [8] + [handValues[p][c][0]] + [handValues[p][c][1]] + handValues[p][c]
                    else:
                        handLabels[p][c] = hands[4] + " " + cardNames[max(handValues[p][c])] + " High"
                        handCode[p][c] = [4] + [handValues[p][c][0]] + [handValues[p][c][1]] + handValues[p][c]
                else:
                    if (handSuited[p][c]):
                        handLabels[p][c] = hands[5] + " " + cardNames[max(handValues[p][c])] + " High"
                        handCode[p][c] = [5] + [handValues[p][c][0]] + [handValues[p][c][1]] + handValues[p][c]
                    else:
                        handLabels[p][c] = hands[0] + " " + cardNames[max(handValues[p][c])] + " High"
                        handCode[p][c] = [0] + [handValues[p][c][0]] + [handValues[p][c][1]] + handValues[p][c]

    winner, finalLabels = [-1] * 5, [-1] * 5
    for c in range(5):
        found = False
        for i in range(8):
            if found is False:
                if handCode[0][c][i] > handCode[1][c][i]:
                    winner[c] = 0
                    found = True
                if handCode[0][c][i] < handCode[1][c][i]:
                    winner[c] = 1
                    found = True
        finalLabels[c] = "Won column " + str(c + 1) + " with " + str(handLabels[winner[c]][c])

    return winner, finalLabels, handCode


nmlp0 = RandomPlayer()
nmlp1 = RandomPlayer()
score = [0, 0]
n_games_same_board = 10_000
n_sims = 1000
res_l = []

for board_ind in tqdm(range(n_sims)):
    # creating board
    d = list(range(52))
    b = [[[-1 for x in range(5)] for y in range(5)] for z in range(2)]
    for i in range(10):
        r = random.sample(d, 1)[0]
        d.remove(r)
        b[(i % 2)][0][i % 5] = r

    b = np.array(b)
    d = np.array(d)
    score = [0, 0]

    for game_ind in (range(n_games_same_board)):
        board = b.copy().tolist()
        deck = d.copy().tolist()
        level = 0
        turn = 1 - (sum(score) % 2)

        for i in range(40):
            curCard = random.sample(deck, 1)[0]
            deck.remove(curCard)
            level = 4 - (len(deck) - 2) // 10
            if (len(deck) - 2) % 10 <= 1:
                choice = -1
                for i in range(5):
                    if board[turn][level][i] == -1:
                        choice = i
            else:
                choice = (nmlp0 if turn == 0 else nmlp1).get_choice(level, turn, board, deck, curCard)
            board[turn][level][choice] = curCard
            turn = 1 - turn

        hand_winner, _, _ = calculate_winner(board)

        score[int(sum(hand_winner) >= 3)] += 1
        if (game_ind % 10) + 1 == 0:
            print(score)

    res_l.append((score[0]/sum(score) + np.array([-1,1]) * 2 *((score[0]/sum(score))*(score[1]/sum(score))/n_games_same_board)**0.5).tolist())

df = pd.DataFrame(res_l, columns=["lb", "ub"])
((df["lb"] > 0.5) | (df["ub"] < 0.5)).mean()
(abs((df["lb"] + df["ub"]) / 2 - 0.5)).describe()

"""
conclusions:

    regarding initial board:
     - initial 10 cards' draw has an (estimated) expected effect of 5% on score
        in other words average game start with probas being 55%-45%.
     - some significant effect is expected on 88% of the games.
     - Those figures are calculated from random players
        I'd expect that a calculated player could exploit this and the advantage might be larger.
     - hard to estimate in a small simulation, but max effect should be somewhere near 20%.
     
    regarding playing 2nd:
    - did a shorter player v player sim in the other script, didn't find any significant effect.
    - theoretically, 2nd player has more information each turn...
"""
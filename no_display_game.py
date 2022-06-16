import random
import pandas as pd
from non_ml_player import NonMlPlayer
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


nmlp0 = NonMlPlayer(proba_method="extended")
nmlp1 = NonMlPlayer(proba_method="none")  # RandomPlayer()
player_0_hand_count = [0] * 9
player_1_hand_count = [0] * 9
score = [0, 0]
n_games = 500
player_game_hands_data, random_game_hands_data = {}, {}

for game_ind in tqdm(range(n_games)):
    deck = list(range(52))
    board = [[[-1 for x in range(5)] for y in range(5)] for z in range(2)]
    level = 0
    turn = 1 - (sum(score) % 2)  # shows whose turn it is

    for i in range(10):
        r = random.sample(deck, 1)[0]
        deck.remove(r)
        board[(i % 2)][level][i % 5] = r

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

    hand_winner, _, player_hands = calculate_winner(board)

    score[int(sum(hand_winner) >= 3)] += 1
    player_game_hands_data[game_ind], random_game_hands_data[game_ind] = [], []
    for i in range(5):
        player_0_hand_count[player_hands[0][i][0]] += 1
        player_1_hand_count[player_hands[1][i][0]] += 1
        player_game_hands_data[game_ind].append(player_hands[0][i][0])
        random_game_hands_data[game_ind].append(player_hands[1][i][0])

# checking proba correctness
if False:
    from catboost import CatBoostClassifier
    import numpy as np
    tqdm.pandas()

    probas_df, decs_df = nmlp0.get_data_as_dfs()
    probas_df["hand_ind"] = [i//3 for i in range(15)] * (n_games * 12)
    probas_df["proba_type"] = ["reg_cpu", "dec_cpu", "opp"] * (n_games * 12 * 5)
    probas_df["level"] = probas_df["decision_id"] // 4 + 2

    player_game_hands_df = pd.DataFrame(list(player_game_hands_data.values()))
    random_game_hands_df = pd.DataFrame(list(random_game_hands_data.values()))
    probas_df["cpu_res"] = probas_df.progress_apply(lambda row: player_game_hands_df.loc[row["game_id"]][row["hand_ind"]], axis=1)

    temp = probas_df[probas_df["proba_type"] == "reg_cpu"]
    temp = pd.melt(temp, id_vars=temp.columns[9:].to_list(), value_vars=temp.columns[:9].to_list(),
                   var_name='hand_type', value_name='proba')
    temp['y'] = temp['cpu_res'] == temp['hand_type']
    x_mat = temp[['level', 'proba', 'hand_type']]

    cat_features = ['level', 'hand_type']
    cb = CatBoostClassifier(n_estimators=500, learning_rate=0.05, max_depth=2, subsample=0.6,
                            one_hot_max_size=10, cat_features=cat_features)
    cb.fit(x_mat, temp['y'])
    temp['preds'] = cb.predict_proba(x_mat)[:, 1]

    a = list(np.concatenate([np.linspace(0.0001, 0.001, 10), np.linspace(0.002, 0.01, 9), np.linspace(0.02, 0.99, 98)]))
    a = a * (9 * 3)
    b = [item for sublist in [[j for i in range(int(len(a)/9))] for j in range(9)] for item in sublist]
    c = [item for sublist in [[j for i in range(int(len(a)/27))] for j in range(2, 5)] for item in sublist] * 9

    proba_conv_df = pd.DataFrame(zip(c, a, b), columns=x_mat.columns)
    proba_conv_df["convd"] = cb.predict_proba(proba_conv_df)[:, 1]
    proba_conv_df = proba_conv_df.sort_values(["hand_type", "level", "proba"], ascending=True)
    proba_conv_df.to_parquet("proba_conv_df.parq")

# checking hand dist
if False:
    df = pd.DataFrame(zip(player_0_hand_count, player_1_hand_count), columns=["p0", "p1"],
                      index=["High Card", "Pair", "Two Pair", "Three of a Kind", "Straight", "Flush",
                             "Full House", "Four of a Kind", "Straight Flush"])

    df/(n_games * 5)
    score[0]/sum(score)


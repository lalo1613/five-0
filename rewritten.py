import pygame, random, os
import numpy as np
import pandas as pd
from poisson_binomial import PoissonBinomial
from scipy.stats import mode

pygame.init()
pygame.font.init()

def loadImages():
    global images, backImage
    suits = ("C", "D", "H", "S")
    images = [[[0 for x in range(13)] for y in range(4)] for z in range(2)]
    backImage = pygame.image.load(r"Cards\CardBack.png")
    for s in range(4):
        for i in range(13):
            images[0][s][i] = pygame.image.load(r"Cards\\" + str(i + 2) + suits[s] + ".png")
            images[1][s][i] = pygame.image.load(r"Cards\D" + str(i + 2) + suits[s] + ".png")

def newGame():
    # creating game arrays
    global player, curCard, turn, level, deck, gameOver
    deck = list(range(52))
    player = [[[-1 for x in range(5)] for y in range(5)] for z in range(2)]

    for i in range(10):
        r = random.sample(deck, 1)[0]
        deck.remove(r)
        player[(i % 2)][0][i % 5] = r

    gameOver = False
    turn = 1 - (sum(score) % 2)  # shows whose turn it is
    curCard = -1  # current cards are drawn during the loop
    level = 0
    redrawBoard()

def calcRandomHandProbas(current_hand, deck):
    current_hand = current_hand[current_hand >= 0]
    ch_suits = current_hand // 13
    ch_values = current_hand % 13

    deck_suits = deck // 13
    deck_values = deck % 13
    deck_n = len(deck)

    ch_mode_val, ch_mode_count = mode(ch_values).mode[0], mode(ch_values).count[0]

    if len(current_hand) == 5:
        if ch_mode_count >= 2:
            if ch_mode_count == 4:
                return [0, 0, 0, 0, 0, 0, 0, 1, 0]
            if ch_mode_count == 3 and mode(ch_values[ch_values != ch_mode_val]).count[0] == 2:
                return [0, 0, 0, 0, 0, 0, 1, 0, 0]
            if ch_mode_count == 3:
                return [0, 0, 0, 1, 0, 0, 0, 0, 0]
            if ch_mode_count == 2 and mode(ch_values[ch_values != ch_mode_val]).count[0] == 2:
                return [0, 0, 1, 0, 0, 0, 0, 0, 0]
            return [0, 1, 0, 0, 0, 0, 0, 0, 0]

        # straight
        if max(ch_values) - min(ch_values) == 5:
            return [0, 0, 0, 0, 1, 0, 0, 0, 0]
        if 12 in ch_values:
            temp_ch_values = ch_values.copy()
            temp_ch_values[temp_ch_values == 12] = -1
            if max(ch_values) - min(ch_values) == 5:
                return [0, 0, 0, 0, 1, 0, 0, 0, 0]

        # flush
        if len(np.unique(ch_suits)) == 1:
            return [0, 0, 0, 0, 0, 1, 0, 0, 0]

        return [1, 0, 0, 0, 0, 0, 0, 0, 0]

    if len(current_hand) == 4:
        if ch_mode_count >= 2:
            if ch_mode_count == 4:
                return [0, 0, 0, 0, 0, 0, 0, 1, 0]
            if ch_mode_count == 3 and mode(ch_values[ch_values != ch_mode_val]).count[0] == 2:
                return [0, 0, 0, 0, 0, 0, 1, 0, 0]
            if ch_mode_count == 2 and mode(ch_values[ch_values != ch_mode_val]).count[0] == 2:
                fh_odds = sum([1 for x in deck_values if x in [ch_mode_val, ch_values[ch_values != ch_mode_val][0]]]) / deck_n
                return [0, 0, 1 - fh_odds, 0, 0, 0, fh_odds, 0, 0]
            if ch_mode_count == 3:
                fh_odds = sum([1 for x in deck_values if x == ch_values[ch_values != ch_mode_val][0]]) / deck_n
                foak_odds = sum([1 for x in deck_values if x == ch_mode_val]) / deck_n
                return [0, 0, 0, 1 - fh_odds - foak_odds, 0, 0, fh_odds, foak_odds, 0]
            # only remaining case is 1p
            toak_odds = sum([1 for x in deck_values if x == ch_mode_val]) / deck_n
            tp_odds = sum([1 for x in deck_values if x in ch_values[ch_values != ch_mode_val]]) / deck_n
            return [0, 1 - tp_odds - toak_odds, tp_odds, toak_odds, 0, 0, 0, 0, 0]

        # straight odds
        if max(ch_values) - min(ch_values) == 4:
            missing_val = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values][0]
            straight_odds = sum([1 for x in deck_values if x == missing_val]) / deck_n
        elif max(ch_values) - min(ch_values) == 3:
            straight_odds = sum([1 for x in deck_values if x in [min(ch_values) - 1, max(ch_values) + 1]]) / deck_n
        else:
            straight_odds = 0
        if 12 in ch_values:
            ch_values[ch_values == 12] = -1
            if max(ch_values) - min(ch_values) == 4:
                missing_val = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values][0]
                straight_odds = sum([1 for x in deck_values if x == missing_val]) / deck_n
            elif max(ch_values) - min(ch_values) == 3:
                straight_odds = sum([1 for x in deck_values if x in [min(ch_values) - 1, max(ch_values) + 1]]) / deck_n
            ch_values[ch_values == -1] = 12

        # flush odds
        if len(np.unique(ch_suits)) == 1:
            flush_odds = sum([1 for x in deck_suits if x == ch_suits[0]]) / deck_n
        else:
            flush_odds = 0

        # pair odds
        pair_odds = sum([1 for x in deck_values if x in ch_values]) / deck_n

        return [1 - straight_odds - flush_odds - pair_odds, pair_odds, 0, 0, straight_odds, flush_odds, 0, 0, 0]

    if len(current_hand) == 3:
        unseen_cards, unseen_cards_counts = np.unique(deck_values[~np.isin(deck_values, ch_values)], return_counts=True)
        extra_pair_odds = ((sum(unseen_cards_counts == 2) * 2) +
                           (sum(unseen_cards_counts == 3) * 3 * 2) +
                           (sum(unseen_cards_counts == 4) * 4 * 3)) / (deck_n * (deck_n - 1))
        if ch_mode_count >= 2:
            if ch_mode_count == 3:
                foak_odds = sum([1 for x in deck_values if x == ch_mode_val]) / deck_n + \
                            (1 - (sum([1 for x in deck_values if x == ch_mode_val]) / deck_n)) * \
                            sum([1 for x in deck_values if x == ch_mode_val]) / (deck_n-1)
                return [0, 0, 0, 1 - extra_pair_odds - foak_odds, 0, 0, extra_pair_odds, foak_odds, 0]
            # only remaining case is 1p
            tp_odds = 2 * (sum([1 for x in deck_values if x in ch_values[ch_values != ch_mode_val]]) / deck_n) * \
                      (sum([1 for x in deck_values if x not in ch_values]) / (deck_n - 1)) + extra_pair_odds
            toak_odds = 2 * (sum([1 for x in deck_values if x == ch_mode_val]) / deck_n) * \
                        (sum([1 for x in deck_values if x not in ch_values]) / (deck_n - 1))
            foak_odds = (sum([1 for x in deck_values if x == ch_mode_val]) / deck_n) * \
                        (max(sum([1 for x in deck_values if x == ch_mode_val]) - 1, 0) / (deck_n-1))
            fh_odds = 2 * (sum([1 for x in deck_values if x in ch_values[ch_values != ch_mode_val]]) / deck_n) * \
                      sum([1 for x in deck_values if x == ch_mode_val]) / (deck_n-1) + \
                      (sum([1 for x in deck_values if x in ch_values[ch_values != ch_mode_val]]) / deck_n) * \
                      (max(sum([1 for x in deck_values if x in ch_values[ch_values != ch_mode_val]]) - 1, 0) / (deck_n - 1))

            return [0, 1 - tp_odds - toak_odds - fh_odds - foak_odds, tp_odds, toak_odds, 0, 0, fh_odds, foak_odds, 0]

        # straight odds
        if max(ch_values) - min(ch_values) == 4:
            missing_vals = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values]
            straight_odds = 2 * (sum([1 for x in deck_values if x == missing_vals[0]]) / deck_n) * \
                            sum([1 for x in deck_values if x == missing_vals[1]]) / (deck_n - 1)
        elif max(ch_values) - min(ch_values) == 3:
            missing_val = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values][0]
            straight_odds = 2 * (sum([1 for x in deck_values if x == missing_val]) / deck_n) * \
                            sum([1 for x in deck_values if x in [min(ch_values) - 1, max(ch_values) + 1]]) / (deck_n - 1)
        elif max(ch_values) - min(ch_values) == 2:
            straight_odds = 2 * (sum([1 for x in deck_values if x == (min(ch_values) - 1)]) / deck_n) * \
                            (sum([1 for x in deck_values if x == (min(ch_values) - 2)]) / (deck_n - 1)) + \
                            2 * (sum([1 for x in deck_values if x == (max(ch_values) + 1)]) / deck_n) * \
                            (sum([1 for x in deck_values if x == (max(ch_values) + 2)]) / (deck_n - 1)) + \
                            2 * (sum([1 for x in deck_values if x == (max(ch_values) + 1)]) / deck_n) * \
                            (sum([1 for x in deck_values if x == (min(ch_values) - 1)]) / (deck_n - 1))
        else:
            straight_odds = 0
        if 12 in ch_values:
            ch_values[ch_values == 12] = -1
            if max(ch_values) - min(ch_values) == 4:
                missing_vals = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values]
                straight_odds = 2 * (sum([1 for x in deck_values if x == missing_vals[0]]) / deck_n) * \
                                sum([1 for x in deck_values if x == missing_vals[1]]) / (deck_n - 1)
            elif max(ch_values) - min(ch_values) == 3:
                missing_val = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values][0]
                straight_odds = 2 * (sum([1 for x in deck_values if x == missing_val]) / deck_n) * \
                                sum([1 for x in deck_values if x in [min(ch_values) - 1, max(ch_values) + 1]]) / (
                                            deck_n - 1)
            elif max(ch_values) - min(ch_values) == 2:
                straight_odds = 2 * (sum([1 for x in deck_values if x == (min(ch_values) - 1)]) / deck_n) * \
                                (sum([1 for x in deck_values if x == (min(ch_values) - 2)]) / (deck_n - 1)) + \
                                2 * (sum([1 for x in deck_values if x == (max(ch_values) + 1)]) / deck_n) * \
                                (sum([1 for x in deck_values if x == (max(ch_values) + 2)]) / (deck_n - 1)) + \
                                2 * (sum([1 for x in deck_values if x == (max(ch_values) + 1)]) / deck_n) * \
                                (sum([1 for x in deck_values if x == (min(ch_values) - 1)]) / (deck_n - 1))
            ch_values[ch_values == -1] = 12


        # flush odds
        if len(np.unique(ch_suits)) == 1:
            flush_odds = (sum([1 for x in deck_suits if x == ch_suits[0]]) / deck_n) * \
                         (max(sum([1 for x in deck_suits if x == ch_suits[0]]) - 1, 0) / (deck_n - 1))
        else:
            flush_odds = 0

        # pair odds
        pair_odds = 2 * (sum([1 for x in deck_values if x in ch_values]) / deck_n) * \
                    (sum([1 for x in deck_values if x not in ch_values]) / (deck_n - 1)) + extra_pair_odds
        toak_odds = sum([(sum([1 for x in deck_values if x == ch_values[i]]) / deck_n) * \
                    (max(sum([1 for x in deck_values if x == ch_values[i]]) - 1, 0) / (deck_n - 1)) for i in range(3)])
        tp_odds = sum([(sum([1 for x in deck_values if x == ch_values[i]]) / deck_n) * \
                        (sum([1 for x in deck_values if x in np.delete(ch_values, i)]) / (deck_n - 1)) for i in range(3)])/2
        high_card_odds = 1 - straight_odds - flush_odds - pair_odds - tp_odds - toak_odds

        return [high_card_odds, pair_odds, tp_odds, toak_odds, straight_odds, flush_odds, 0, 0, 0]

    if len(current_hand) == 2:
        unseen_cards, unseen_cards_counts = np.unique(deck_values[~np.isin(deck_values, ch_values)], return_counts=True)
        extra_toak_odds = (sum(unseen_cards_counts == 3) * 3 * 2) / (deck_n * (deck_n - 1) * (deck_n - 2)) + \
                          (sum(unseen_cards_counts == 4) * 4 * 3 * 2) / (deck_n * (deck_n - 1) * (deck_n - 2))

        extra_pair_odds = (3 * sum([1 for x in deck_values if x not in ch_values]) *
                            sum(unseen_cards_counts == 2) * 2) / (deck_n * (deck_n - 1) * (deck_n - 2)) + \
                          (3 * sum([1 for x in deck_values if x not in ch_values]) *
                            sum(unseen_cards_counts == 3) * 3 * 2) / (deck_n * (deck_n - 1) * (deck_n - 2)) + \
                          (3 * sum([1 for x in deck_values if x not in ch_values]) *
                            sum(unseen_cards_counts == 4) * 4 * 3) / (deck_n * (deck_n - 1) * (deck_n - 2)) - extra_toak_odds

        extra_two_unrelated_odds = sum([i * sum(unseen_cards_counts == i) *
                                        (sum([1 for x in deck_values if x not in ch_values]) - i) for i in range(1,5)]) / \
                                    ((deck_n - 1) * (deck_n - 2))

        if ch_mode_count == 2:
            tp_odds = extra_pair_odds

            toak_odds = 3 * (sum([1 for x in deck_values if x == ch_mode_val]) / deck_n) * extra_two_unrelated_odds

            foak_odds = 3 * (sum([1 for x in deck_values if x == ch_mode_val]) / deck_n) * \
                        (max(sum([1 for x in deck_values if x == ch_mode_val]) - 1, 0) / (deck_n - 1)) * \
                        (sum([1 for x in deck_values if x != ch_mode_val]) / (deck_n - 2))

            fh_odds = 3 * (sum([1 for x in deck_values if x == ch_mode_val]) / deck_n) * extra_pair_odds + extra_toak_odds

            return [0, 1 - tp_odds - toak_odds - fh_odds - foak_odds, tp_odds, toak_odds, 0, 0, fh_odds, foak_odds, 0]

        # straight odds
        if max(ch_values) - min(ch_values) == 4:
            missing_vals = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values]
            straight_odds = 3 * 2 * (sum([1 for x in deck_values if x == missing_vals[0]]) / deck_n) * \
                            sum([1 for x in deck_values if x == missing_vals[1]]) / (deck_n - 1) * \
                            sum([1 for x in deck_values if x == missing_vals[2]]) / (deck_n - 2)
        elif max(ch_values) - min(ch_values) == 3:
            missing_vals = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values]
            straight_odds = 3 * 2 * (sum([1 for x in deck_values if x == missing_vals[0]]) / deck_n) * \
                            sum([1 for x in deck_values if x == missing_vals[1]]) / (deck_n - 1) * \
                            sum([1 for x in deck_values if x in [min(ch_values) - 1, max(ch_values) + 1]]) / (deck_n - 2)
        elif max(ch_values) - min(ch_values) == 2:
            missing_val = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values][0]
            straight_odds = 3 * 2 * ((sum([1 for x in deck_values if x == missing_val]) / (deck_n - 2))) * \
                            ((sum([1 for x in deck_values if x == (min(ch_values) - 1)]) / deck_n) * \
                            (sum([1 for x in deck_values if x == (min(ch_values) - 2)]) / (deck_n - 1)) + \
                            (sum([1 for x in deck_values if x == (max(ch_values) + 1)]) / deck_n) * \
                            (sum([1 for x in deck_values if x == (max(ch_values) + 2)]) / (deck_n - 1)) + \
                            (sum([1 for x in deck_values if x == (max(ch_values) + 1)]) / deck_n) * \
                            (sum([1 for x in deck_values if x == (min(ch_values) - 1)]) / (deck_n - 1)))
        elif max(ch_values) - min(ch_values) == 1:
            poss_vals = [-3, -2, -1, 1, 2, 3]
            needed_func = [min, min, min, max, max, max]
            straight_odds = 0
            for s in range(4):
                straight_odds += np.prod([sum([1 for x in deck_values if x == needed_func[i](ch_values) + i]) for i in poss_vals[s:(s+3)]]) / \
                                (deck_n * (deck_n - 1) * (deck_n - 2))
            straight_odds = 3 * 2 * straight_odds
        else:
            straight_odds = 0
        if 12 in ch_values:
            ch_values[ch_values == 12] = -1
            if max(ch_values) - min(ch_values) == 4:
                missing_vals = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values]
                straight_odds = 3 * 2 * (sum([1 for x in deck_values if x == missing_vals[0]]) / deck_n) * \
                                sum([1 for x in deck_values if x == missing_vals[1]]) / (deck_n - 1) * \
                                sum([1 for x in deck_values if x == missing_vals[2]]) / (deck_n - 2)
            elif max(ch_values) - min(ch_values) == 3:
                missing_vals = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values]
                straight_odds = 3 * 2 * (sum([1 for x in deck_values if x == missing_vals[0]]) / deck_n) * \
                                sum([1 for x in deck_values if x == missing_vals[1]]) / (deck_n - 1) * \
                                sum([1 for x in deck_values if x in [min(ch_values) - 1, max(ch_values) + 1]]) / (
                                            deck_n - 2)
            elif max(ch_values) - min(ch_values) == 2:
                missing_val = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values][0]
                straight_odds = 3 * 2 * ((sum([1 for x in deck_values if x == missing_val]) / (deck_n - 2))) * \
                                ((sum([1 for x in deck_values if x == (min(ch_values) - 1)]) / deck_n) * \
                                 (sum([1 for x in deck_values if x == (min(ch_values) - 2)]) / (deck_n - 1)) + \
                                 (sum([1 for x in deck_values if x == (max(ch_values) + 1)]) / deck_n) * \
                                 (sum([1 for x in deck_values if x == (max(ch_values) + 2)]) / (deck_n - 1)) + \
                                 (sum([1 for x in deck_values if x == (max(ch_values) + 1)]) / deck_n) * \
                                 (sum([1 for x in deck_values if x == (min(ch_values) - 1)]) / (deck_n - 1)))
            elif max(ch_values) - min(ch_values) == 1:
                poss_vals = [-3, -2, -1, 1, 2, 3]
                needed_func = [min, min, min, max, max, max]
                straight_odds = 0
                for s in range(4):
                    straight_odds += np.prod(
                        [sum([1 for x in deck_values if x == needed_func[i](ch_values) + i]) for i in
                         poss_vals[s:(s + 3)]]) / \
                                     (deck_n * (deck_n - 1) * (deck_n - 2))
                straight_odds = 3 * 2 * straight_odds
                ch_values[ch_values == -1] = 12

        # flush odds
        if len(np.unique(ch_suits)) == 1:
            flush_odds = (sum([1 for x in deck_suits if x == ch_suits[0]]) / deck_n) * \
                         (max(sum([1 for x in deck_suits if x == ch_suits[0]]) - 1, 0) / (deck_n - 1)) * \
                         (max(sum([1 for x in deck_suits if x == ch_suits[0]]) - 2, 0) / (deck_n - 2))
        else:
            flush_odds = 0

        # pair odds
        pair_odds = 3 * (sum([1 for x in deck_values if x in ch_values]) / deck_n) * extra_two_unrelated_odds + extra_pair_odds

        tp_odds = 3 * 0.5 * sum([(sum([1 for x in deck_values if x == ch_values[i]]) / deck_n) * \
                  (sum([1 for x in deck_values if x in np.delete(ch_values, i)]) / (deck_n - 1)) for i in range(2)]) * \
                  (sum([1 for x in deck_values if x not in ch_values]) / (deck_n - 2)) + \
                  3 * (sum([1 for x in deck_values if x in ch_values]) / deck_n) * extra_pair_odds

        toak_odds = 3 * sum([(sum([1 for x in deck_values if x == ch_values[i]]) / deck_n) * \
                    (max(sum([1 for x in deck_values if x == ch_values[i]]) - 1, 0) / (deck_n - 1)) for i in range(2)]) * \
                    (sum([1 for x in deck_values if x not in ch_values]) / (deck_n - 2)) + extra_toak_odds

        fh_odds = 3 * sum([(sum([1 for x in deck_values if x == ch_values[i]]) / deck_n) * \
                           (max(sum([1 for x in deck_values if x == ch_values[i]]) - 1, 0) / (deck_n - 1)) * \
                           (sum([1 for x in deck_values if x == ch_values[1 - i]]) / (deck_n - 2)) for i in range(2)])

        foak_odds = sum([(sum([1 for x in deck_values if x == ch_values[i]]) / deck_n) * \
                         (max(sum([1 for x in deck_values if x == ch_values[i]]) - 1, 0) / (deck_n - 1)) * \
                         (max(sum([1 for x in deck_values if x == ch_values[i]]) - 2, 0) / (deck_n - 2)) for i in range(2)])

        high_card_odds = 1 - straight_odds - flush_odds - pair_odds - tp_odds - toak_odds - fh_odds - foak_odds

        return [high_card_odds, pair_odds, tp_odds, toak_odds, straight_odds, flush_odds, fh_odds, foak_odds, 0]

def optimisticProbas(p_vec, n_remaining):
    n = n_remaining
    if n_remaining == 0:
        return p_vec
    new_vec = np.array(p_vec).copy()
    for i in np.flip(range(2, 9)):
        new_vec[i] = min((1 - (1 - (new_vec[i]**(1/n)))**(2.5))**n, 1 - sum(new_vec[i:]))
        new_vec[:i] *= (1 - sum(new_vec[i:]))/sum(new_vec[:i])
    return new_vec

def calcHandProbas(current_hand, deck):
   return calcRandomHandProbas(current_hand, deck)

def rearrangeHand(hand):
    hand_vals = np.array([val % 13 if val > 0 else -1 for val in hand])
    hand_vals = list(hand_vals[hand_vals >= 0])
    hand_order = np.flip(np.argsort([i + (hand_vals.count(i) - 1) * 13 for i in hand_vals]))
    rearrranged_hand = np.array(hand_vals)[hand_order]
    return np.concatenate([rearrranged_hand, [-1] * (5 - len(hand_vals))])

def get_curr_leader(cpu_hand, opp_hand):
    cpu_hand_vals = rearrangeHand(cpu_hand)
    opp_hand_vals = rearrangeHand(opp_hand)
    for i in range(5):
        if cpu_hand_vals[i] > opp_hand_vals[i]:
            return "cpu", i
        elif cpu_hand_vals[i] < opp_hand_vals[i]:
            return "opp", i
    return "tie", -1

def tieBreakProbas(cpu_probas, opp_probas, cpu_hand, opp_hand, deck):
    res_vec = [0.5] * 9
    deck_suits = np.array(deck) // 13
    deck_values = np.array(deck) % 13
    deck_n = len(deck)
    n_remaining = sum(cpu_hand == -1)

    cpu_hand_vals = np.array([val % 13 if val > 0 else -1 for val in cpu_hand])
    opp_hand_vals = np.array([val % 13 if val > 0 else -1 for val in opp_hand])

    lead, lead_ind = get_curr_leader(cpu_hand_vals, opp_hand_vals)
    
    ### both don't have pairs yet
    # high card calc:
    if cpu_probas[0] > 0 and opp_probas[0] > 0:
        if lead == "cpu":
            over_cards = sum(deck_values > cpu_hand_vals[lead_ind])
            res_vec[0] = 1 - ((over_cards/deck_n)**(1/n_remaining)) * 0.5
        elif lead == "opp":
            over_cards = sum(deck_values > opp_hand_vals[lead_ind])
            res_vec[0] = ((over_cards/deck_n)**(1/n_remaining)) * 0.5

        # pair, tp, toak, fh, foak calc:
        counter = 0
        for i in range(5):
            counter += (sum(deck_values == cpu_hand_vals[i]) / sum(np.isin(deck_values, cpu_hand_vals))) * \
                       ((sum(np.isin(deck_values, opp_hand_vals[opp_hand_vals < cpu_hand_vals[i]])) +
                         0.5 * sum(np.isin(deck_values, opp_hand_vals[opp_hand_vals == cpu_hand_vals[i]])))
                        / sum(np.isin(deck_values, opp_hand_vals)))
        res_vec[1] = counter
        res_vec[2] = res_vec[1]
        res_vec[3] = res_vec[1]
        res_vec[6] = res_vec[1]
        res_vec[7] = res_vec[1]

    # opp has pair, cpu doesn't
    elif cpu_probas[0] > 0:
        res_vec[1] = sum(np.isin(deck_values, cpu_hand_vals[cpu_hand_vals > opp_hand_vals[0]])) \
                     / sum(np.isin(deck_values, cpu_hand_vals))
        res_vec[1] += 0.5 * sum(np.isin(deck_values, cpu_hand_vals[cpu_hand_vals == opp_hand_vals[0]])) \
                      / sum(np.isin(deck_values, cpu_hand_vals))
        if any(np.isnan(res_vec)):
            print("stop")


    # cpu has pair, opp doesn't
    elif opp_probas[0] > 0:
        res_vec[1] = 1 - sum(np.isin(deck_values, opp_hand_vals[opp_hand_vals > cpu_hand_vals[0]])) \
                     / sum(np.isin(deck_values, opp_hand_vals))
        res_vec[1] -= 0.5 * sum(np.isin(deck_values, opp_hand_vals[opp_hand_vals == cpu_hand_vals[0]])) \
                      / sum(np.isin(deck_values, opp_hand_vals))

    # both have at least a pair
    else:
        if lead == "cpu":
            res_vec[1] = 1
        elif lead == "opp":
            res_vec[1] = 0

        # both have only pair

    # straight and flush
        # if no chance of overcoming - 0
        # otherwise calc that chance
    return res_vec

def calcHandWinProba(cpu_probas, opp_probas, cpu_hand, opp_hand, deck):
    if sum(cpu_hand == -1) == 0:
        win_count = 0
        for i in range(len(deck)):
            poss_opp_hand = opp_hand.copy()
            poss_opp_hand[4] = deck[i]
            poss_opp_probas = calcHandProbas(poss_opp_hand, deck)
            if np.argmax(cpu_probas) > np.argmax(poss_opp_probas):
                win_count += 1
            elif np.argmax(cpu_probas) == np.argmax(poss_opp_probas):
                if get_curr_leader(cpu_hand, poss_opp_hand) == "cpu":
                    win_count += 1
        return win_count/len(deck)

    tie_break_win_probas = tieBreakProbas(cpu_probas, opp_probas, cpu_hand, opp_hand, deck)
    tie_break_win_probas = np.array(cpu_probas) * opp_probas * tie_break_win_probas
    opp_cum_probas = np.cumsum(opp_probas)
    cpu_win_proba = sum([cpu_probas[i] * opp_cum_probas[i-1] for i in range(1, 9)]) + sum(tie_break_win_probas)
    return cpu_win_proba

def getCPUchoice():
    options = [x for x in range(5) if player[turn][level][x] == -1]
    temp_player = np.array(player)
    cpu_hands = [temp_player[turn, :, i] for i in range(5)]
    opp_hands = [temp_player[1 - turn, :, i] for i in range(5)]
    for i in range(5):
        opp_hands[i][4] = -1
    deck_left = np.concatenate([deck, temp_player[1 - turn][4][temp_player[1 - turn][4] != -1]])

    if level < 2:
        cpu_decision_probas_list = [[-1] * 9] * 5
        for i in range(5):
            vec = cpu_hands[i].copy()
            if i in options:
                vec[level] = curCard
                cpu_decision_probas_list[i] = calcHandProbas(vec, deck_left)

        expected_val = [sum([l[x] * x for x in range(9)]) for l in cpu_decision_probas_list]
        decision = int(np.argmax(expected_val))
        return decision

    cpu_probas_list = [calcHandProbas(cpu_hands[i], deck_left) for i in range(5)]
    opp_probas_list = [calcHandProbas(opp_hands[i], deck_left) for i in range(5)]

    cpu_decision_probas_list = [[-1] * 9] * 5
    as_is_win_probas, decision_win_probas = [-1] * 5, [-1] * 5

    for i in range(5):
        vec = cpu_hands[i].copy()
        opp_vec = opp_hands[i].copy()
        as_is_win_probas[i] = calcHandWinProba(cpu_probas_list[i], opp_probas_list[i],
                                               vec, opp_vec, deck_left)
        if i in options:
            vec[level] = curCard
            cpu_decision_probas_list[i] = calcHandProbas(vec, deck_left)
            decision_win_probas[i] = calcHandWinProba(cpu_decision_probas_list[i], opp_probas_list[i],
                                                      vec, opp_vec, deck_left)
    # interwined_probas_list = [item for trio in zip(cpu_probas_list, cpu_decision_probas_list, opp_probas_list) for item in trio]
    # temp_df = pd.DataFrame(interwined_probas_list, index=[["cpu", "cpu+card", "opp"][i % 3] + str(i//3) for i in range(15)])

    dec_ovr_win_probas = [-1] * 5
    for i in options:
        post_dec_vec = as_is_win_probas.copy()
        post_dec_vec[i] = decision_win_probas[i]
        pb = PoissonBinomial(post_dec_vec)
        dec_ovr_win_probas[i] = pb.x_or_more(3)

    decision = int(np.argmax(dec_ovr_win_probas))
    return decision

def redrawBoard():
    win.fill(backcolor)

    for i in range(5):
        for j in range(level+1):
            if player[0][j][i] > -1:
                if (cpuPlayer is True) and (j == 4):
                    win.blit(backImage,(260 + i * 100, 170 - 30 * j))
                else:
                    win.blit(images[1 - (turn == 0)][player[0][j][i] // 13][player[0][j][i] % 13], (260 + i * 100, 170 - 30 * j))
            if player[1][j][i] > -1:
                win.blit(images[1 - (turn == 1)][player[1][j][i] // 13][player[1][j][i] % 13], (260 + i * 100, 320 + 30 * j))

    textFont = pygame.font.SysFont('comicsansms', 22)
    pygame.draw.rect(win, (150, 200, 150), (50, 250, 160, 100))
    pygame.draw.rect(win, (0, 0, 0), (50, 250, 160, 100),3)
    pygame.draw.line(win,(0, 0, 0), (50, 300),(210,300),3)
    p1 = textFont.render("Player 1 ", False, (0, 0, 250))
    p2 = textFont.render("Player 2 ", False, (250, 0, 0))
    win.blit(p1, (60, 258))
    win.blit(p2, (60, 308))
    s1 = textFont.render(str(score[0]), False, (0, 0, 250))
    s2 = textFont.render(str(score[1]), False, (250, 0, 0))
    if(score[0]//10 > 0):
        win.blit(s1, (180, 258))
    else:
        win.blit(s1, (190, 258))
    if (score[1] // 10 > 0):
        win.blit(s2, (180, 308))
    else:
        win.blit(s2, (190, 308))
    # pygame.draw.rect(win, (150, 200, 150), (870, 510, 100, 60))
    # pygame.draw.rect(win, (0, 0, 0), (870, 510, 100, 60), 3)
    # button = textFont.render("Restart", False, (0, 0, 0))
    # win.blit(button, (882, 522))

def drawEndBoard(winner,labels):
    win.fill(backcolor)
    for i in range(5):
        for j in range(5):
                win.blit(images[1-(winner[i] == 0)][player[0][j][i] // 13][player[0][j][i] % 13],(260 + i * 100, 170 - 30 * j))
                win.blit(images[1-(winner[i] == 1)][player[1][j][i] // 13][player[1][j][i] % 13],(260 + i * 100, 320 + 30 * j))
    textFont = pygame.font.SysFont('comicsansms', 22)
    pygame.draw.rect(win, (150, 200, 150), (50, 250, 160, 100))
    pygame.draw.rect(win, (0, 0, 0), (50, 250, 160, 100), 3)
    pygame.draw.line(win, (0, 0, 0), (50, 300), (210, 300), 3)
    p1 = textFont.render("Player 1 ", False, (0, 0, 250))
    p2 = textFont.render("Player 2 ", False, (250, 0, 0))
    win.blit(p1, (60, 258))
    win.blit(p2, (60, 308))
    s1 = textFont.render(str(score[0]), False, (0, 0, 250))
    s2 = textFont.render(str(score[1]), False, (250, 0, 0))
    if (score[0] // 10 > 0):
        win.blit(s1, (180, 258))
    else:
        win.blit(s1, (190, 258))
    if (score[1] // 10 > 0):
        win.blit(s2, (180, 308))
    else:
        win.blit(s2, (190, 308))
    pygame.draw.rect(win, (150, 200, 150), (870, 510, 100, 60))
    pygame.draw.rect(win, (0, 0, 0), (870, 510, 100, 60), 3)
    button = textFont.render("Restart", False, (0, 0, 0))
    win.blit(button, (882, 522))

    labelFont = pygame.font.SysFont('arial', 12)
    for i in [4,3,2,1,0]: #230 and up or 355 and down
        l = labelFont.render(labels[i], False, (255, 255, 0))
        if winner[i] ==0:
            win.blit(l,(15, 230 - 15*(i-sum(winner[0:i]))))
        else:
            win.blit(l, (15, 355 + 15 * sum(winner[0:i])))

def moveCardTo(turn,row,column):
    n = round(speedModifier*12*(10-column) - 1)
    for i in range(1,n+1):
        redrawBoard()
        if (turn == 0) and (cpuPlayer is True) and (level == 4):
            win.blit(backImage, (780 * ((n - i) / n) + (260 + column * 100) * ((i) / n), 245 * ((n - i) / n) + (170 + 150 * turn + 30 * row * (2 * turn - 1)) * ((i) / n)))
        else:
            win.blit(images[0][curCard // 13][curCard % 13],(780*((n-i)/n) + (260+column*100)*((i)/n), 245*((n-i)/n) + (170+150*turn + 30*row*(2*turn-1))*((i)/n)))
        pygame.display.update()

def calculateWinner():
    hands = ["High Card", "Pair", "Two Pair", "Three of a Kind", "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"]
    cardNames = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    handValueCounts = [[[0 for x in range(13)] for y in range(5)] for z in range(2)]
    handSuited =[[0 for x in range(5)] for y in range(2)]
    handValues = [[[0 for x in range(5)] for y in range(5)] for z in range(2)]
    handLabels = [[0 for x in range(5)] for y in range(2)]
    handCode = [[0 for x in range(5)] for y in range(2)]  # first number - hand type, next 2 - hand parameters, last 5 - sorted hand values
    for p in range(2):
        for c in range(5):
            handSuited[p][c] = (player[p][0][c] // 13 == player[p][1][c] // 13 == player[p][2][c] // 13 == player[p][3][c] // 13 == player[p][4][c] // 13)
            for r in range(5):
                handValueCounts[p][c][player[p][r][c] % 13] += 1
                handValues[p][c][r] = player[p][r][c] % 13

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

    winner = [-1 for i in range(5)]
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

    finalLabels = [-1 for i in range(5)]
    for c in range(5):
        finalLabels[c] = "Won column "+ str(c+1)+ " with "+ str(handLabels[winner[c]][c])

    if(sum(winner) < 3):
        score[0] += 1
    else:
        score[1] += 1

    drawEndBoard(winner, finalLabels)



    #return (handCode + handLabels)


# creating window, setting background color to casino green and naming the window
os.environ['SDL_VIDEO_WINDOW_POS'] = str(100) + "," + str(50)
win = pygame.display.set_mode((1000, 600))
backcolor = (0, 120, 20)
win.fill(backcolor)
pygame.display.set_caption("Five-O Poker")

run = True
score = [0, 0]
speedModifier = 1/5  # smaller means faster
cpuPlayer = True
opp_is_cpu_as_well = True
loadImages()
newGame()

while run:
    pygame.time.delay(10)
    pygame.display.update()

    # should a new card be drawn?
    if (curCard == -1) and (len(deck) > 2):
        curCard = random.sample(deck, 1)[0]
        deck.remove(curCard)
        redrawBoard()
        if (turn == 0) and (cpuPlayer is True) and (level == 4):
            win.blit(backImage, (780, 245))
        else:
            win.blit(images[0][curCard // 13][curCard % 13], (780, 245))

    # has the board been filled?
    if len(deck) == 2:
        curCard = random.sample(deck, 1)[0]
        deck.remove(curCard)
        gameOver = True
        calculateWinner()

    # is this the last card of the row? (if so, it'll be auto-placed)
    if (len(deck) - 2) % 10 <= 1:
        pygame.display.update()
        pygame.time.delay(round(200 * speedModifier))
        choice = -1
        for i in range(5):
            if player[turn][level][i] == -1:
                choice = i
        moveCardTo(turn, level, choice)
        player[turn][level][choice] = curCard
        curCard = -1
        turn = 1 - turn

    # is the cpu on?
    if (cpuPlayer is True) and ((len(deck)-2) % 10 > 1) and (turn == 0) and (len(deck) > 2):
        if len(deck) % 10 == 1:
            level += 1
        pygame.display.update()
        pygame.time.delay(round(200 * speedModifier))
        choice = getCPUchoice()
        moveCardTo(turn, level, choice)
        player[turn][level][choice] = curCard
        curCard = -1
        turn = 1 - turn

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if not gameOver:
            # placing cards based on mouse click
            if event.type == pygame.MOUSEBUTTONUP and ((len(deck)-2) % 10 > 1) and (cpuPlayer*(1-turn) != 1):
                pos = pygame.mouse.get_pos()
                if ((pos[0]) in range(260,740)) and ((pos[0]-260) % 100 in range(0,80)) and (pos[1]+270*(turn == 0) in range(320,550)):
                    choice = (pos[0] - 260)//100
                    level = 4 - ((len(deck)-2) // 10)
                    if player[turn][level][choice] == -1:
                        moveCardTo(turn, level, choice)
                        player[turn][level][choice] = curCard  # make sure to do the addition of the card after the animation
                        curCard = -1
                        turn = 1-turn
            # placing cards based on use of 1-5 keys
            if event.type == pygame.KEYDOWN and ((len(deck) - 2) % 10 > 1) and (cpuPlayer*(1-turn) != 1):
                if event.key == pygame.K_1:
                    choice = 0
                    level = 4 - ((len(deck) - 2) // 10)
                    if player[turn][level][choice] == -1:
                        moveCardTo(turn, level, choice)
                        player[turn][level][choice] = curCard  # make sure to do the addition of the card after the animation
                        curCard = -1
                        turn = 1 - turn
                if event.key == pygame.K_2:
                    choice = 1
                    level = 4 - ((len(deck) - 2) // 10)
                    if player[turn][level][choice] == -1:
                        moveCardTo(turn, level, choice)
                        player[turn][level][choice] = curCard  # make sure to do the addition of the card after the animation
                        curCard = -1
                        turn = 1 - turn
                if event.key == pygame.K_3:
                    choice = 2
                    level = 4 - ((len(deck) - 2) // 10)
                    if player[turn][level][choice] == -1:
                        moveCardTo(turn, level, choice)
                        player[turn][level][choice] = curCard  # make sure to do the addition of the card after the animation
                        curCard = -1
                        turn = 1 - turn
                if event.key == pygame.K_4:
                    choice = 3
                    level = 4 - ((len(deck) - 2) // 10)
                    if player[turn][level][choice] == -1:
                        moveCardTo(turn, level, choice)
                        player[turn][level][choice] = curCard  # make sure to do the addition of the card after the animation
                        curCard = -1
                        turn = 1 - turn
                if event.key == pygame.K_5:
                    choice = 4
                    level = 4 - ((len(deck) - 2) // 10)
                    if player[turn][level][choice] == -1:
                        moveCardTo(turn, level, choice)
                        player[turn][level][choice] = curCard  # make sure to do the addition of the card after the animation
                        curCard = -1
                        turn = 1 - turn
        else:
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                if (pos[0] in range(870, 970)) and (pos[1] in range(510, 570)):
                    newGame()

pygame.quit()


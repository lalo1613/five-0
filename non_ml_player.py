import numpy as np
import pandas as pd
from poisson_binomial import PoissonBinomial
from scipy.stats import mode
from math import log10, ceil
from time import time

class NonMlPlayer():
    """
    Things left to work on:
    Overall:
        1. accumulating data

    get_choice:
        1. Improve approach for first round cards
        2. "As Is probas" vs. "Decision probas" in last round differ due to different calcs, but they shouldn't.

    calc_random_hand_probas:
        1. whole thing is a mess, but somehow it's a functioning mess. Not the most pressing issue atm.
        2. straight flush is incorrect

    win_tie_probas:
        1. current code is a band-aid, supplying a bad solution for only 1/3 of the problem. Top priority!
        2. There are divisions by 0 here. That's embarassing.

    calc_hand_probas and optimistic_probas:
        1. this are meant to be solutions for using random assignment probas on a choice assignment game.
        2. one possible approach is gathering data and using it to optimize this.

    """
    def __init__(self, save_data=True, proba_method="convd_probas"):
        self.save_data = save_data
        self.proba_method = proba_method
        if proba_method == "convd_probas":
            self.proba_conv_df = pd.read_parquet("proba_conv_df.parq")
            self.proba_conv_df["proba"] = self.proba_conv_df["proba"].apply(lambda x: round(x, 4))
            self.proba_conv_df.index = 10*self.proba_conv_df["level"] + self.proba_conv_df["hand_type"] + self.proba_conv_df["proba"]
            self.proba_conv_df = self.proba_conv_df["convd"]
        self.game_id = 0
        self.decision_id = 0
        self.probas_data = {}
        self.decisions_data = {}

    def get_choice(self, level, turn, player, deck, pot_card):
            options = [x for x in range(5) if player[turn][level][x] == -1]
            temp_player = np.array(player)
            cpu_hands = [temp_player[turn, :, i].copy() for i in range(5)]
            opp_hands = [temp_player[1 - turn, :, i].copy() for i in range(5)]
            for i in range(5):
                opp_hands[i][4] = -1
            deck_left = np.concatenate([deck, temp_player[1 - turn][4][temp_player[1 - turn][4] != -1]])

            if level < 2:
                cpu_decision_probas_list = [[-1] * 9] * 5
                for i in range(5):
                    vec = cpu_hands[i].copy()
                    if i in options:
                        vec[level] = pot_card
                        cpu_decision_probas_list[i] = self.calc_hand_probas(vec, deck_left)

                expected_val = [sum([l[x] * x for x in range(9)]) for l in cpu_decision_probas_list]
                decision = int(np.argmax(expected_val))
                return decision

            cpu_probas_list = [self.calc_hand_probas(cpu_hands[i], deck_left) for i in range(5)]
            opp_probas_list = [self.calc_hand_probas(opp_hands[i], deck_left) for i in range(5)]

            cpu_decision_probas_list = [[-1] * 9] * 5
            as_is_win_probas, decision_win_probas = [-1] * 5, [-1] * 5

            for i in range(5):
                vec = cpu_hands[i].copy()
                opp_vec = opp_hands[i].copy()
                as_is_win_probas[i] = self.calc_hand_win_proba(cpu_probas_list[i], opp_probas_list[i],
                                                               vec, opp_vec, deck_left)
                if i in options:
                    vec[level] = pot_card
                    cpu_decision_probas_list[i] = self.calc_hand_probas(vec, deck_left)
                    decision_win_probas[i] = self.calc_hand_win_proba(cpu_decision_probas_list[i], opp_probas_list[i],
                                                                      vec, opp_vec, deck_left)
            dec_ovr_win_probas = [-1] * 5
            for i in options:
                post_dec_vec = as_is_win_probas.copy()
                post_dec_vec[i] = decision_win_probas[i]
                pb = PoissonBinomial(post_dec_vec)
                dec_ovr_win_probas[i] = pb.x_or_more(3)

            decision = int(np.argmax(dec_ovr_win_probas))
            # a = pd.DataFrame([opp_probas_list[2][(13*i):(13*(i+1))] for i in range(9)])
            # b = pd.DataFrame([cpu_decision_probas_list[2][(13*i):(13*(i+1))] for i in range(9)])
            if self.save_data:
                interwined_probas_list = [item for trio in zip(cpu_probas_list, cpu_decision_probas_list, opp_probas_list) for item in trio]
                probas_data = pd.DataFrame(interwined_probas_list, index=[["cpu", "cpu+card", "opp"][i % 3] + str(i//3) for i in range(15)])
                self.log_data(probas_data, dec_ovr_win_probas)

            return decision

    def calc_hand_probas(self, current_hand, deck):
        if self.proba_method == "convd_probas":
            return self.convd_probas(self.calc_random_hand_probas(current_hand, deck), 4 - (len(deck) - 2) // 10)
        if self.proba_method == "optimistic":
            return self.optimistic_probas(self.calc_random_hand_probas(current_hand, deck), 4 - (len(deck) - 2) // 10)
        if self.proba_method == "extended":
            return [item for sub_l in self.calc_extended_random_hand_probas(current_hand, deck) for item in sub_l]

        return self.calc_random_hand_probas(current_hand, deck)

    def calc_random_hand_probas(self, current_hand, deck):
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
            if max(ch_values) - min(ch_values) == 4:
                if len(np.unique(ch_suits)) == 1:
                    return [0, 0, 0, 0, 0, 0, 0, 0, 1]
                else:
                    return [0, 0, 0, 0, 1, 0, 0, 0, 0]
            if 12 in ch_values:
                temp_ch_values = ch_values.copy()
                temp_ch_values[temp_ch_values == 12] = -1
                if max(ch_values) - min(ch_values) == 4:
                    if len(np.unique(ch_suits)) == 1:
                        return [0, 0, 0, 0, 0, 0, 0, 0, 1]
                    else:
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
                    fh_odds = sum(
                        [1 for x in deck_values if x in [ch_mode_val, ch_values[ch_values != ch_mode_val][0]]]) / deck_n
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

            return [1 - straight_odds - flush_odds - pair_odds, pair_odds, 0, 0, straight_odds, flush_odds, 0, 0, straight_odds * flush_odds]

        if len(current_hand) == 3:
            unseen_cards, unseen_cards_counts = np.unique(deck_values[~np.isin(deck_values, ch_values)], return_counts=True)
            extra_pair_odds = ((sum(unseen_cards_counts == 2) * 2) +
                               (sum(unseen_cards_counts == 3) * 3 * 2) +
                               (sum(unseen_cards_counts == 4) * 4 * 3)) / (deck_n * (deck_n - 1))
            if ch_mode_count >= 2:
                if ch_mode_count == 3:
                    foak_odds = sum([1 for x in deck_values if x == ch_mode_val]) / deck_n + \
                                (1 - (sum([1 for x in deck_values if x == ch_mode_val]) / deck_n)) * \
                                sum([1 for x in deck_values if x == ch_mode_val]) / (deck_n - 1)
                    return [0, 0, 0, 1 - extra_pair_odds - foak_odds, 0, 0, extra_pair_odds, foak_odds, 0]
                # only remaining case is 1p
                tp_odds = 2 * (sum([1 for x in deck_values if x in ch_values[ch_values != ch_mode_val]]) / deck_n) * \
                          (sum([1 for x in deck_values if x not in ch_values]) / (deck_n - 1)) + extra_pair_odds
                toak_odds = 2 * (sum([1 for x in deck_values if x == ch_mode_val]) / deck_n) * \
                            (sum([1 for x in deck_values if x not in ch_values]) / (deck_n - 1))
                foak_odds = (sum([1 for x in deck_values if x == ch_mode_val]) / deck_n) * \
                            (max(sum([1 for x in deck_values if x == ch_mode_val]) - 1, 0) / (deck_n - 1))
                fh_odds = 2 * (sum([1 for x in deck_values if x in ch_values[ch_values != ch_mode_val]]) / deck_n) * \
                          sum([1 for x in deck_values if x == ch_mode_val]) / (deck_n - 1) + \
                          (sum([1 for x in deck_values if x in ch_values[ch_values != ch_mode_val]]) / deck_n) * \
                          (max(sum([1 for x in deck_values if x in ch_values[ch_values != ch_mode_val]]) - 1, 0) / (
                                      deck_n - 1))

                return [0, 1 - tp_odds - toak_odds - fh_odds - foak_odds, tp_odds, toak_odds, 0, 0, fh_odds, foak_odds, 0]

            # straight odds
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
                             (max(sum([1 for x in deck_values if x == ch_values[i]]) - 1, 0) / (deck_n - 1)) for i in
                             range(3)])
            tp_odds = sum([(sum([1 for x in deck_values if x == ch_values[i]]) / deck_n) * \
                           (sum([1 for x in deck_values if x in np.delete(ch_values, i)]) / (deck_n - 1)) for i in
                           range(3)])
            high_card_odds = 1 - straight_odds - flush_odds - pair_odds - tp_odds - toak_odds

            return [high_card_odds, pair_odds, tp_odds, toak_odds, straight_odds, flush_odds, 0, 0, straight_odds * flush_odds]

        if len(current_hand) == 2:
            unseen_cards, unseen_cards_counts = np.unique(deck_values[~np.isin(deck_values, ch_values)], return_counts=True)
            extra_toak_odds = (sum(unseen_cards_counts == 3) * 3 * 2) / (deck_n * (deck_n - 1) * (deck_n - 2)) + \
                              (sum(unseen_cards_counts == 4) * 4 * 3 * 2) / (deck_n * (deck_n - 1) * (deck_n - 2))

            extra_pair_odds = (3 * sum([1 for x in deck_values if x not in ch_values]) *
                               sum(unseen_cards_counts == 2) * 2) / (deck_n * (deck_n - 1) * (deck_n - 2)) + \
                              (3 * sum([1 for x in deck_values if x not in ch_values]) *
                               sum(unseen_cards_counts == 3) * 3 * 2) / (deck_n * (deck_n - 1) * (deck_n - 2)) + \
                              (3 * sum([1 for x in deck_values if x not in ch_values]) *
                               sum(unseen_cards_counts == 4) * 4 * 3) / \
                              (deck_n * (deck_n - 1) * (deck_n - 2)) - extra_toak_odds

            extra_two_unrelated_odds = sum([i * sum(unseen_cards_counts == i) *
                                            (sum([1 for x in deck_values if x not in ch_values]) - i) for i in
                                            range(1, 5)]) / ((deck_n - 1) * (deck_n - 2))

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
                    a = np.prod([sum([1 for x in deck_values if x == needed_func[s + i](ch_values) + poss_vals[s + i]])
                                              for i in range(3)]) * 3 * 2 / (deck_n * (deck_n - 1) * (deck_n - 2))
                    straight_odds += a
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
                             poss_vals[s:(s + 3)]]) / (deck_n * (deck_n - 1) * (deck_n - 2))
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
            pair_odds = 3 * (sum([1 for x in deck_values if x in ch_values]) / deck_n) * \
                        extra_two_unrelated_odds + extra_pair_odds

            tp_odds = 3 * sum([(sum([1 for x in deck_values if x == ch_values[i]]) / deck_n) * \
                                     (sum([1 for x in deck_values if x in np.delete(ch_values, i)]) / (deck_n - 1)) for i in
                                     range(2)]) * \
                      (sum([1 for x in deck_values if x not in ch_values]) / (deck_n - 2)) + \
                      3 * (sum([1 for x in deck_values if x in ch_values]) / sum([1 for x in deck_values if x not in ch_values])) * extra_pair_odds

            toak_odds = 3 * sum([(sum([1 for x in deck_values if x == ch_values[i]]) / deck_n) * \
                                 (max(sum([1 for x in deck_values if x == ch_values[i]]) - 1, 0) / (deck_n - 1))
                                 for i in range(2)]) * (sum([1 for x in deck_values if x not in ch_values]) / (deck_n - 2)) + extra_toak_odds

            fh_odds = 3 * sum([(sum([1 for x in deck_values if x == ch_values[i]]) / deck_n) * \
                               (max(sum([1 for x in deck_values if x == ch_values[i]]) - 1, 0) / (deck_n - 1)) * \
                               (sum([1 for x in deck_values if x == ch_values[1 - i]]) / (deck_n - 2)) for i in range(2)])

            foak_odds = sum([(sum([1 for x in deck_values if x == ch_values[i]]) / deck_n) * \
                             (max(sum([1 for x in deck_values if x == ch_values[i]]) - 1, 0) / (deck_n - 1)) * \
                             (max(sum([1 for x in deck_values if x == ch_values[i]]) - 2, 0) / (deck_n - 2)) for i in
                             range(2)])

            high_card_odds = 1 - straight_odds - flush_odds - pair_odds - tp_odds - toak_odds - fh_odds - foak_odds

            return [high_card_odds, pair_odds, tp_odds, toak_odds, straight_odds, flush_odds, fh_odds, foak_odds, straight_odds * flush_odds]

    def depreciated_calc_extended_random_hand_probas(self, current_hand, deck):
        """
        current_hand, deck = cpu_hands[1].copy(), deck_left.copy()
        """
        probas_df = pd.DataFrame([[0.0] * 9] * 13, index=range(13), columns=range(9))
        current_hand = current_hand[current_hand >= 0]
        ch_suits = current_hand // 13
        ch_values = current_hand % 13
        n_remaining = 5 - len(current_hand)

        deck_suit_counts = pd.Series(deck // 13).value_counts()
        deck_value_counts = pd.Series(deck % 13).value_counts()
        for v in ([-3,-2,-1,13,14,15] + [i for i in range(13) if i not in deck_value_counts.index]):
            deck_value_counts[v] = 0
        deck_n = len(deck)

        ch_mode = mode(ch_values)
        ch_mode_val, ch_mode_count = ch_mode.mode[0], ch_mode.count[0]
        sec_ch_mode = mode(ch_values[ch_values != ch_mode_val])
        if len(sec_ch_mode.mode) > 0:
            sec_ch_mode_val, sec_ch_mode_count = sec_ch_mode.mode[0], sec_ch_mode.count[0]

        if n_remaining == 0:
            if ch_mode_count >= 2:
                if ch_mode_count == 4:
                    probas_df.loc[ch_mode_val, 7] = 1
                    return probas_df
                elif ch_mode_count == 3:
                    if sec_ch_mode_count == 2:
                        probas_df.loc[ch_mode_val, 6] = 1
                        return probas_df
                    else:
                        probas_df.loc[ch_mode_val, 3] = 1
                        return probas_df
                if sec_ch_mode_count == 2:
                    probas_df.loc[max(ch_mode_val, sec_ch_mode_val), 2] = 1
                    return probas_df
                probas_df.loc[ch_mode_val, 1] = 1
                return probas_df

            # straight
            if max(ch_values) - min(ch_values) == 4:
                if len(np.unique(ch_suits)) == 1:
                    probas_df.loc[max(ch_values), 8] = 1
                    return probas_df
                else:
                    probas_df.loc[max(ch_values), 4] = 1
                    return probas_df
            if 12 in ch_values:
                temp_ch_values = ch_values.copy()
                temp_ch_values[temp_ch_values == 12] = -1
                if max(ch_values) - min(ch_values) == 5:
                    if len(np.unique(ch_suits)) == 1:
                        probas_df.loc[max(ch_values), 8] = 1
                        return probas_df
                    else:
                        probas_df.loc[max(ch_values), 4] = 1
                        return probas_df

            # flush
            if len(np.unique(ch_suits)) == 1:
                probas_df.loc[max(ch_values), 5] = 1
                return probas_df

            probas_df.loc[max(ch_values), 0] = 1
            return probas_df

        if n_remaining == 1:
            if ch_mode_count >= 2:
                if ch_mode_count == 4:
                    probas_df.loc[ch_mode_val, 7] = 1
                    return probas_df
                if ch_mode_count == 3 and mode(ch_values[ch_values != ch_mode_val]).count[0] == 2:
                    probas_df.loc[ch_mode_val, 6] = 1
                    return probas_df
                if ch_mode_count == 2 and sec_ch_mode_count == 2:
                    probas_df.loc[ch_mode_val, 6] = deck_value_counts[ch_mode_val]/deck_n
                    probas_df.loc[sec_ch_mode_val, 6] = deck_value_counts[sec_ch_mode_val]/deck_n
                    probas_df.loc[max(ch_mode_val, sec_ch_mode_val), 2] = 1 - probas_df.sum().sum()
                    return probas_df
                if ch_mode_count == 3:
                    probas_df.loc[ch_mode_val, 6] = deck_value_counts[sec_ch_mode_val]/deck_n
                    probas_df.loc[ch_mode_val, 7] = deck_value_counts[ch_mode_val]/deck_n
                    probas_df.loc[ch_mode_val, 3] = 1 - probas_df.sum().sum()
                    return probas_df
                # only remaining case is 1p
                probas_df.loc[ch_mode_val, 3] = deck_value_counts[ch_mode_val]/deck_n
                for val in ch_values[ch_values != ch_mode_val]:
                    probas_df.loc[val, 2] = deck_value_counts[val]/deck_n
                probas_df.loc[ch_mode_val, 1] = 1 - probas_df.sum().sum()
                return probas_df

            # new HC odds
            for val in deck_value_counts[(deck_value_counts.index > max(ch_values)) & (deck_value_counts.index <= 12)].index:
                probas_df.loc[val, 0] = deck_value_counts[val] / deck_n

            # straight odds
            if max(ch_values) - min(ch_values) == 4:
                missing_val = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values][0]
                probas_df.loc[max(ch_values), 4] = deck_value_counts[missing_val]/deck_n
            elif max(ch_values) - min(ch_values) == 3:
                probas_df.loc[max(ch_values) + 1, 4] = deck_value_counts[max(ch_values) + 1] / deck_n
                probas_df.loc[max(ch_values), 4] = deck_value_counts[min(ch_values) - 1] / deck_n
                probas_df.loc[max(ch_values) + 1, 0] = 0
            elif 12 in ch_values:
                ch_values[ch_values == 12] = -1
                if max(ch_values) - min(ch_values) == 4:
                    missing_val = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values][0]
                    probas_df.loc[max(ch_values), 4] = deck_value_counts[missing_val] / deck_n
                    probas_df.loc[missing_val, 0] = 0
                elif max(ch_values) - min(ch_values) == 3:
                    probas_df.loc[max(ch_values) + 1, 4] = deck_value_counts[max(ch_values) + 1] / deck_n
                    probas_df.loc[max(ch_values), 4] = deck_value_counts[min(ch_values) - 1] / deck_n
                    probas_df.loc[max(ch_values) + 1, 0] = 0
                ch_values[ch_values == -1] = 12

            # flush odds
            if len(np.unique(ch_suits)) == 1:
                n_tot_cards = deck_suit_counts[ch_suits[0]]
                over_cards = deck[(deck > max(current_hand)) & (deck < 13 * (ch_suits[0] + 1))]
                for val in over_cards:
                    probas_df.loc[val % 13, 5] = 1 / deck_n
                    probas_df.loc[val % 13, 0] *= (deck_value_counts[val % 13] - 1)/(deck_value_counts[val % 13])
                probas_df.loc[max(ch_values), 5] = (n_tot_cards - len(over_cards)) / deck_n

            # pair odds
            for val in ch_values:
                probas_df.loc[val, 1] = deck_value_counts[val] / deck_n

            # existing HC odds
            probas_df.loc[max(ch_values), 0] = 1 - probas_df.sum().sum()
            return probas_df

        if n_remaining == 2:
            extra_pair_vec = deck_value_counts * (deck_value_counts - 1) / (deck_n * (deck_n - 1))
            extra_pair_vec[extra_pair_vec.index.isin(ch_values)] = 0

            if ch_mode_count >= 2:
                if ch_mode_count == 3:
                    probas_df.loc[ch_mode_val, 7] = deck_value_counts[ch_mode_val] / deck_n + \
                                                    (1 - deck_value_counts[ch_mode_val] / deck_n) * \
                                                    (deck_value_counts[ch_mode_val] / (deck_n - 1))

                    probas_df.loc[ch_mode_val, 6] = sum(extra_pair_vec)
                    probas_df.loc[ch_mode_val, 3] = 1 - probas_df.sum().sum()
                    return probas_df

                # only remaining case is 1p

                # tp
                probas_df.loc[max(sec_ch_mode_val, ch_mode_val), 2] = 2 * (deck_value_counts[sec_ch_mode_val] / deck_n) * \
                          (deck_value_counts[~deck_value_counts.index.isin(ch_values)].sum() / (deck_n - 1))

                for val in extra_pair_vec[extra_pair_vec > 0].index:
                    if val > ch_mode_val:
                        probas_df.loc[val, 2] = extra_pair_vec[val]
                    else:
                        probas_df.loc[ch_mode_val, 2] += extra_pair_vec[val]

                # toak
                probas_df.loc[ch_mode_val, 3] = 2 * (deck_value_counts[ch_mode_val] / deck_n) * \
                          (deck_value_counts[~deck_value_counts.index.isin(ch_values)].sum() / (deck_n - 1))
                # fh
                probas_df.loc[sec_ch_mode_val, 6] = (deck_value_counts[sec_ch_mode_val] / deck_n) * \
                          ((deck_value_counts[sec_ch_mode_val] - 1) / (deck_n - 1))

                probas_df.loc[ch_mode_val, 6] = 2 * (deck_value_counts[sec_ch_mode_val] / deck_n) * \
                          (deck_value_counts[ch_mode_val] / (deck_n - 1))
                # foak
                probas_df.loc[ch_mode_val, 7] = (deck_value_counts[ch_mode_val] / deck_n) * \
                          ((deck_value_counts[ch_mode_val] - 1) / (deck_n - 1))

                probas_df.loc[ch_mode_val, 1] = 1 - probas_df.sum().sum()
                return probas_df

            # new HC odds
            tot_nr_cards = deck_value_counts[~deck_value_counts.index.isin(ch_values)].sum()
            over_cards = np.flip(np.sort(deck_value_counts[(deck_value_counts.index > max(ch_values)) & (deck_value_counts.index <= 12)].index))
            cumulative_overs = deck_value_counts[over_cards].cumsum()
            for i in range(len(over_cards)):
                val = over_cards[i]
                probas_df.loc[val, 0] = 2 * (deck_value_counts[val] / deck_n) * ((tot_nr_cards - cumulative_overs.iloc[i]) / (deck_n - 1))

            # straight odds
            if max(ch_values) - min(ch_values) == 4:
                missing_vals = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values]
                probas_df.loc[max(ch_values), 4] = 2 * (deck_value_counts[missing_vals[0]] / deck_n) * \
                                                   (deck_value_counts[missing_vals[1]] / (deck_n - 1))
            elif max(ch_values) - min(ch_values) == 3:
                missing_val = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values][0]
                probas_df.loc[max(ch_values), 4] = 2 * (deck_value_counts[missing_val] / deck_n) * \
                                                   (deck_value_counts[min(ch_values) - 1] / (deck_n - 1))
                probas_df.loc[max(ch_values) + 1, 4] = 2 * (deck_value_counts[missing_val] / deck_n) * \
                                                   (deck_value_counts[max(ch_values) + 1] / (deck_n - 1))
                probas_df.loc[max(ch_values) + 1, 0] = max(0, probas_df.loc[max(ch_values) + 1, 0] -
                                                           probas_df.loc[max(ch_values) + 1, 4])

            elif max(ch_values) - min(ch_values) == 2:
                poss_vals = [-2, -1, 1, 2]
                needed_func = [min, min, max, max]
                for s in range(3):
                    probas_df.loc[max(ch_values) + (poss_vals[s+1] if poss_vals[s+1] > 0 else 0), 4] = 2 * \
                        deck_value_counts[needed_func[s](ch_values) + poss_vals[s]] * \
                        deck_value_counts[needed_func[s+1](ch_values) + poss_vals[s+1]] / (deck_n * (deck_n - 1))
                probas_df.loc[max(ch_values) + 1, 0] = max(0, probas_df.loc[max(ch_values) + 1, 0] -
                                                           probas_df.loc[max(ch_values) + 1, 4])
                probas_df.loc[max(ch_values) + 2, 0] = max(0, probas_df.loc[max(ch_values) + 2, 0] -
                                                           probas_df.loc[max(ch_values) + 2, 4])

            if 12 in ch_values:
                ch_values[ch_values == 12] = -1
                if max(ch_values) - min(ch_values) == 4:
                    missing_vals = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values]
                    probas_df.loc[max(ch_values), 4] = 2 * (deck_value_counts[missing_vals[0]] / deck_n) * \
                                                       (deck_value_counts[missing_vals[1]] / (deck_n - 1))
                elif max(ch_values) - min(ch_values) == 3:
                    missing_val = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values][0]
                    probas_df.loc[max(ch_values), 4] = 2 * (deck_value_counts[missing_val] / deck_n) * \
                                                       (deck_value_counts[min(ch_values) - 1] / (deck_n - 1))
                    probas_df.loc[max(ch_values) + 1, 4] = 2 * (deck_value_counts[missing_val] / deck_n) * \
                                                           (deck_value_counts[max(ch_values) + 1] / (deck_n - 1))
                    probas_df.loc[max(ch_values) + 1, 0] = max(0, probas_df.loc[max(ch_values) + 1, 0] -
                                                               probas_df.loc[max(ch_values) + 1, 4])

                elif max(ch_values) - min(ch_values) == 2:
                    poss_vals = [-2, -1, 1, 2]
                    needed_func = [min, min, max, max]
                    for s in range(3):
                        probas_df.loc[max(ch_values) + (poss_vals[s+1] if poss_vals[s+1] > 0 else 0), 4] = 2 * \
                            deck_value_counts[needed_func[s](ch_values) + poss_vals[s]] * \
                            deck_value_counts[needed_func[s+1](ch_values) + poss_vals[s+1]] / (deck_n * (deck_n - 1))
                    probas_df.loc[max(ch_values) + 1, 0] = max(0, probas_df.loc[max(ch_values) + 1, 0] -
                                                               probas_df.loc[max(ch_values) + 1, 4])
                    probas_df.loc[max(ch_values) + 2, 0] = max(0, probas_df.loc[max(ch_values) + 2, 0] -
                                                               probas_df.loc[max(ch_values) + 2, 4])
                ch_values[ch_values == -1] = 12

            # flush odds
            if len(np.unique(ch_suits)) == 1:
                n_tot_cards = deck_suit_counts[ch_suits[0]]
                over_cards = np.flip(np.sort(deck[(deck > max(current_hand)) & (deck < 13 * (ch_suits[0] + 1))]))
                for i in range(len(over_cards)):
                    val = over_cards[i] % 13
                    probas_df.loc[val, 5] = 2 * (1 / deck_n) * (n_tot_cards - i - 1) / (deck_n - 1)
                    probas_df.loc[val, 0] = max(0, probas_df.loc[val, 0] - probas_df.loc[val, 5])

                probas_df.loc[max(ch_values), 5] = ((n_tot_cards - len(over_cards)) / deck_n) * \
                                                   (n_tot_cards - len(over_cards) - 1) / (deck_n - 1)

            # pair odds
            for val in ch_values:
                probas_df.loc[val, 1] = 2 * (deck_value_counts[val] / deck_n) * \
                            (deck_value_counts[~deck_value_counts.index.isin(ch_values)].sum() / (deck_n - 1))
            for val in extra_pair_vec[extra_pair_vec > 0].index:
                probas_df.loc[val, 1] = extra_pair_vec[val]

            # toak odds
            for val in ch_values:
                probas_df.loc[val, 3] = (deck_value_counts[val] / deck_n) * ((deck_value_counts[val] - 1) / (deck_n - 1))

            # tp odds
            sorted_ch_vals = np.flip(np.sort(ch_values.copy()))
            for i in range(len(sorted_ch_vals) - 1):
                probas_df.loc[sorted_ch_vals[i], 2] = 2 * (deck_value_counts[sorted_ch_vals[i]] / deck_n) * \
                          (sum(deck_value_counts[deck_value_counts.index.isin(sorted_ch_vals[(i+1):])]) / (deck_n - 1))

            # existing HC odds
            probas_df.loc[max(ch_values), 0] = 1 - probas_df.sum().sum()
            return probas_df

        if n_remaining == 3:
            extra_pair_vec = 3 * deck_value_counts * (deck_value_counts - 1) * \
                             (deck_value_counts[~deck_value_counts.index.isin(ch_values)].sum()) / \
                             (deck_n * (deck_n - 1) * (deck_n - 2))
            extra_pair_vec[extra_pair_vec.index.isin(ch_values)] = 0

            extra_toak_vec = deck_value_counts * (deck_value_counts - 1) * (deck_value_counts - 2) / \
                             (deck_n * (deck_n - 1) * (deck_n - 2))
            extra_toak_vec[extra_toak_vec.index.isin(ch_values)] = 0

            extra_pair_vec = extra_pair_vec - extra_toak_vec

            extra_two_unrelated_odds = sum([i * sum(deck_value_counts[~deck_value_counts.index.isin(ch_values)] == i) *
                                            (sum(deck_value_counts[~deck_value_counts.index.isin(ch_values)]) - i)
                                            for i in range(1, 5)]) / ((deck_n - 1) * (deck_n - 2))

            if ch_mode_count == 2:
                # tp odds
                for val in extra_pair_vec[extra_pair_vec > 0].index:
                    if val > ch_mode_val:
                        probas_df.loc[val, 2] = extra_pair_vec[val]
                    else:
                        probas_df.loc[ch_mode_val, 2] += extra_pair_vec[val]

                # toak odds
                probas_df.loc[ch_mode_val, 3] = 3 * (deck_value_counts[ch_mode_val] / deck_n) * extra_two_unrelated_odds

                # fh odds
                for val in extra_toak_vec[extra_toak_vec > 0].index:
                    probas_df.loc[val, 6] = extra_toak_vec[val]

                probas_df.loc[ch_mode_val, 6] = 3 * (deck_value_counts[ch_mode_val] / deck_n) * sum(extra_pair_vec)

                # foak odds
                probas_df.loc[ch_mode_val, 7] = 3 * (deck_value_counts[ch_mode_val] * (deck_value_counts[ch_mode_val] - 1) *
                     deck_value_counts[~deck_value_counts.index.isin(ch_values)].sum()) / (deck_n * (deck_n - 1) * (deck_n - 2))

                # current pair only odds
                probas_df.loc[ch_mode_val, 1] = 1 - probas_df.sum().sum()
                return probas_df

            # new HC odds
            tot_nr_cards = deck_value_counts[~deck_value_counts.index.isin(ch_values)].sum()
            over_cards = np.flip(np.sort(deck_value_counts[(deck_value_counts.index > max(ch_values)) & (deck_value_counts.index <= 12)].index))
            cumulative_overs = deck_value_counts[over_cards].cumsum()
            for i in range(len(over_cards)):
                val = over_cards[i]
                probas_df.loc[val, 0] = 3 * (deck_value_counts[val] / deck_n) * \
                                        ((tot_nr_cards - cumulative_overs.iloc[i]) / (deck_n - 1)) * \
                                        ((tot_nr_cards - cumulative_overs.iloc[i] - 1) / (deck_n - 2))

            # straight odds
            if max(ch_values) - min(ch_values) == 4:
                missing_vals = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values]
                probas_df.loc[max(ch_values), 4] = 3 * 2 * (deck_value_counts[missing_vals[0]] / deck_n) * \
                                                   (deck_value_counts[missing_vals[1]] / (deck_n - 1)) * \
                                                   (deck_value_counts[missing_vals[2]] / (deck_n - 2))

            elif max(ch_values) - min(ch_values) == 3:
                missing_vals = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values]
                probas_df.loc[max(ch_values) + 1, 4] = 3 * 2 * (deck_value_counts[missing_vals[0]] / deck_n) * \
                                                   (deck_value_counts[missing_vals[1]] / (deck_n - 1)) * \
                                                   (deck_value_counts[max(ch_values) + 1] / (deck_n - 2))
                probas_df.loc[max(ch_values), 4] = 3 * 2 * (deck_value_counts[missing_vals[0]] / deck_n) * \
                                                   (deck_value_counts[missing_vals[1]] / (deck_n - 1)) * \
                                                   (deck_value_counts[min(ch_values) - 1] / (deck_n - 2))

            elif max(ch_values) - min(ch_values) == 2:
                missing_val = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values][0]
                poss_vals = [-2, -1, 1, 2]
                needed_func = [min, min, max, max]
                for s in range(3):
                    probas_df.loc[max(ch_values) + (poss_vals[s+1] if poss_vals[s+1] > 0 else 0), 4] = 3 * 2 * \
                        deck_value_counts[needed_func[s](ch_values) + poss_vals[s]] * \
                        deck_value_counts[needed_func[s+1](ch_values) + poss_vals[s+1]] * \
                        deck_value_counts[missing_val] / (deck_n * (deck_n - 1) * (deck_n - 2))

                probas_df.loc[max(ch_values) + 1, 0] = max(0, probas_df.loc[max(ch_values) + 1, 0] -
                                                           probas_df.loc[max(ch_values) + 1, 4])
                probas_df.loc[max(ch_values) + 2, 0] = max(0, probas_df.loc[max(ch_values) + 2, 0] -
                                                           probas_df.loc[max(ch_values) + 2, 4])

            elif max(ch_values) - min(ch_values) == 1:
                poss_vals = [-3, -2, -1, 1, 2, 3]
                needed_func = [min, min, min, max, max, max]
                for s in range(4):
                    addition = (poss_vals[s+2] if poss_vals[s+2] > 0 else (poss_vals[s+1] if poss_vals[s+1] > 0 else 0))
                    probas_df.loc[(max(ch_values) + addition), 4] = 3 * 2 * \
                        deck_value_counts[needed_func[s](ch_values) + poss_vals[s]] * \
                        deck_value_counts[needed_func[s+1](ch_values) + poss_vals[s+1]] * \
                        deck_value_counts[needed_func[s+2](ch_values) + poss_vals[s+2]] / \
                        (deck_n * (deck_n - 1) * (deck_n - 2))

                probas_df.loc[max(ch_values) + 1, 0] = max(0, probas_df.loc[max(ch_values) + 1, 0] -
                                                           probas_df.loc[max(ch_values) + 1, 4])
                probas_df.loc[max(ch_values) + 2, 0] = max(0, probas_df.loc[max(ch_values) + 2, 0] -
                                                           probas_df.loc[max(ch_values) + 2, 4])
                probas_df.loc[max(ch_values) + 3, 0] = max(0, probas_df.loc[max(ch_values) + 3, 0] -
                                                           probas_df.loc[max(ch_values) + 3, 4])

            if 12 in ch_values:
                ch_values[ch_values == 12] = -1
                if max(ch_values) - min(ch_values) == 4:
                    missing_vals = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values]
                    probas_df.loc[max(ch_values), 4] = 3 * 2 * (deck_value_counts[missing_vals[0]] / deck_n) * \
                                                       (deck_value_counts[missing_vals[1]] / (deck_n - 1)) * \
                                                       (deck_value_counts[missing_vals[2]] / (deck_n - 2))

                elif max(ch_values) - min(ch_values) == 3:
                    missing_vals = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values]
                    probas_df.loc[max(ch_values) + 1, 4] = 3 * 2 * (deck_value_counts[missing_vals[0]] / deck_n) * \
                                                       (deck_value_counts[missing_vals[1]] / (deck_n - 1)) * \
                                                       (deck_value_counts[max(ch_values) + 1] / (deck_n - 2))
                    probas_df.loc[max(ch_values), 4] = 3 * 2 * (deck_value_counts[missing_vals[0]] / deck_n) * \
                                                       (deck_value_counts[missing_vals[1]] / (deck_n - 1)) * \
                                                       (deck_value_counts[min(ch_values) - 1] / (deck_n - 2))

                elif max(ch_values) - min(ch_values) == 2:
                    missing_val = [x for x in range(min(ch_values), max(ch_values)) if x not in ch_values][0]
                    poss_vals = [-2, -1, 1, 2]
                    needed_func = [min, min, max, max]
                    for s in range(3):
                        probas_df.loc[max(ch_values) + (poss_vals[s+1] if poss_vals[s+1] > 0 else 0), 4] = 3 * 2 * \
                            deck_value_counts[needed_func[s](ch_values) + poss_vals[s]] * \
                            deck_value_counts[needed_func[s+1](ch_values) + poss_vals[s+1]] * \
                            deck_value_counts[missing_val] / (deck_n * (deck_n - 1) * (deck_n - 2))

                    probas_df.loc[max(ch_values) + 1, 0] = max(0, probas_df.loc[max(ch_values) + 1, 0] -
                                                               probas_df.loc[max(ch_values) + 1, 4])
                    probas_df.loc[max(ch_values) + 2, 0] = max(0, probas_df.loc[max(ch_values) + 2, 0] -
                                                               probas_df.loc[max(ch_values) + 2, 4])

                elif max(ch_values) - min(ch_values) == 1:
                    poss_vals = [-3, -2, -1, 1, 2, 3]
                    needed_func = [min, min, min, max, max, max]
                    for s in range(4):
                        addition = (poss_vals[s+2] if poss_vals[s+2] > 0 else (poss_vals[s+1] if poss_vals[s+1] > 0 else 0))
                        probas_df.loc[(max(ch_values) + addition), 4] = 3 * 2 * \
                            deck_value_counts[needed_func[s](ch_values) + poss_vals[s]] * \
                            deck_value_counts[needed_func[s+1](ch_values) + poss_vals[s+1]] * \
                            deck_value_counts[needed_func[s+2](ch_values) + poss_vals[s+2]] / \
                            (deck_n * (deck_n - 1) * (deck_n - 2))

                    probas_df.loc[max(ch_values) + 1, 0] = max(0, probas_df.loc[max(ch_values) + 1, 0] -
                                                               probas_df.loc[max(ch_values) + 1, 4])
                    probas_df.loc[max(ch_values) + 2, 0] = max(0, probas_df.loc[max(ch_values) + 2, 0] -
                                                               probas_df.loc[max(ch_values) + 2, 4])
                    probas_df.loc[max(ch_values) + 3, 0] = max(0, probas_df.loc[max(ch_values) + 3, 0] -
                                                               probas_df.loc[max(ch_values) + 3, 4])

                ch_values[ch_values == -1] = 12

            # flush odds
            if len(np.unique(ch_suits)) == 1:
                n_tot_cards = deck_suit_counts[ch_suits[0]]
                over_cards = np.flip(np.sort(deck[(deck > max(current_hand)) & (deck < 13 * (ch_suits[0] + 1))]))
                for i in range(len(over_cards)):
                    val = over_cards[i] % 13
                    probas_df.loc[val, 5] = 3 * (1 / deck_n) * \
                                            (n_tot_cards - i - 1) / (deck_n - 1) * \
                                            (n_tot_cards - i - 2) / (deck_n - 2)
                    probas_df.loc[val, 0] = max(0, probas_df.loc[val, 0] - probas_df.loc[val, 5])

                probas_df.loc[max(ch_values), 5] = ((n_tot_cards - len(over_cards)) / deck_n) * \
                                                   (n_tot_cards - len(over_cards) - 1) / (deck_n - 1) * \
                                                   (n_tot_cards - len(over_cards) - 2) / (deck_n - 2)

            # pair odds
            for val in ch_values:
                probas_df.loc[val, 1] = 3 * (deck_value_counts[val] / deck_n) * extra_two_unrelated_odds
            for val in extra_pair_vec[extra_pair_vec > 0].index:
                probas_df.loc[val, 1] = extra_pair_vec[val]

            # tp odds - 2 existing cards
            sorted_ch_vals = np.flip(np.sort(ch_values.copy()))
            for i in range(len(sorted_ch_vals) - 1):
                probas_df.loc[sorted_ch_vals[i], 2] = 3 * 2 * (deck_value_counts[sorted_ch_vals[i]] / deck_n) * \
                          (sum(deck_value_counts[deck_value_counts.index.isin(sorted_ch_vals[(i+1):])]) / (deck_n - 1)) * \
                          (deck_value_counts[~deck_value_counts.index.isin(ch_values)].sum() / (deck_n - 2))

            # tp odds - 1 existing + extra pair
            for ch_val in ch_values:
                for extra_val in extra_pair_vec[(extra_pair_vec > 0)].index:
                    # we make up for 'extra_pair_vec' having already accounted for an extra unrelated card
                    if ch_val > extra_val:
                        probas_df.loc[ch_val, 2] += 3 * extra_pair_vec[extra_val] * \
                                                    (deck_value_counts[ch_val]) / \
                                                    (deck_value_counts[~deck_value_counts.index.isin(ch_values)].sum())
                    else:
                        probas_df.loc[extra_val, 2] += 3 * extra_pair_vec[extra_val] * \
                                                    (deck_value_counts[ch_val]) / \
                                                    (deck_value_counts[~deck_value_counts.index.isin(ch_values)].sum())

            # toak odds
            for val in ch_values:
                probas_df.loc[val, 3] = 3 * (deck_value_counts[val] / deck_n) * \
                                        ((deck_value_counts[val] - 1) / (deck_n - 1)) * \
                                        (deck_value_counts[~deck_value_counts.index.isin(ch_values)].sum() / (deck_n - 2))

            for val in extra_toak_vec[extra_toak_vec > 0].index:
                probas_df.loc[val, 3] = extra_toak_vec[val]

            # fh odds
            probas_df.loc[ch_values[0], 6] = 3 * (deck_value_counts[ch_values[0]] / deck_n) * \
                                             ((deck_value_counts[ch_values[0]] - 1) / (deck_n - 1)) * \
                                             ((deck_value_counts[ch_values[1]]) / (deck_n - 2))

            probas_df.loc[ch_values[1], 6] = 3 * (deck_value_counts[ch_values[1]] / deck_n) * \
                                             ((deck_value_counts[ch_values[1]] - 1) / (deck_n - 1)) * \
                                             ((deck_value_counts[ch_values[0]]) / (deck_n - 2))
            # foak odds
            for val in ch_values:
                probas_df.loc[val, 7] = (deck_value_counts[val] / deck_n) * \
                                        ((deck_value_counts[val] - 1) / (deck_n - 1)) * \
                                        ((deck_value_counts[val] - 2) / (deck_n - 2))

            # existing HC odds
            probas_df.loc[max(ch_values), 0] = 1 - probas_df.sum().sum()
            return probas_df

    def calc_extended_random_hand_probas(self, current_hand, deck):
        probas_list = [[0.0 for val in range(13)] for hand_type in range(9)]
        current_hand = current_hand[current_hand >= 0]
        ch_suits = current_hand // 13
        ch_values = current_hand % 13
        n_remaining = 5 - len(current_hand)
        
        deck_suits = deck // 13
        deck_values = deck % 13
        max_ch_values = max(ch_values)
        min_ch_values = min(ch_values)
        deck_n = len(deck)

        ch_mode = mode(ch_values)
        ch_mode_val, ch_mode_count = ch_mode.mode[0], ch_mode.count[0]
        sec_ch_mode = mode(ch_values[ch_values != ch_mode_val])
        if len(sec_ch_mode.mode) > 0:
            sec_ch_mode_val, sec_ch_mode_count = sec_ch_mode.mode[0], sec_ch_mode.count[0]

        if n_remaining == 0:
            if ch_mode_count >= 2:
                if ch_mode_count == 4:
                    probas_list[7][ch_mode_val] = 1
                    return probas_list
                elif ch_mode_count == 3:
                    if sec_ch_mode_count == 2:
                        probas_list[6][ch_mode_val] = 1
                        return probas_list
                    else:
                        probas_list[3][ch_mode_val] = 1
                        return probas_list
                if sec_ch_mode_count == 2:
                    probas_list[2][max(ch_mode_val, sec_ch_mode_val)] = 1
                    return probas_list
                probas_list[1][ch_mode_val] = 1
                return probas_list

            # straight
            if 12 in ch_values and sum(ch_values < 4) == (4 - n_remaining):
                ch_values[ch_values == 12] = -1
            if max(ch_values) - min_ch_values == 4:
                if len(np.unique(ch_suits)) == 1:
                    probas_list[8][max(ch_values)] = 1
                    return probas_list
                else:
                    probas_list[4][max(ch_values)] = 1
                    return probas_list

            # flush
            if len(np.unique(ch_suits)) == 1:
                probas_list[5][max_ch_values] = 1
                return probas_list

            probas_list[0][max_ch_values] = 1
            return probas_list

        if n_remaining == 1:
            if ch_mode_count >= 2:
                if ch_mode_count == 4:
                    probas_list[7][ch_mode_val] = 1
                    return probas_list
                if ch_mode_count == 3 and mode(ch_values[ch_values != ch_mode_val]).count[0] == 2:
                    probas_list[6][ch_mode_val] = 1
                    return probas_list
                if ch_mode_count == 2 and sec_ch_mode_count == 2:
                    probas_list[6][ch_mode_val] = sum([1 for val in deck_values if val == ch_mode_val])/deck_n
                    probas_list[6][sec_ch_mode_val] = sum([1 for val in deck_values if val == sec_ch_mode_val])/deck_n
                    probas_list[2][max(ch_mode_val, sec_ch_mode_val)] = 1 - sum([sum(l) for l in probas_list])
                    return probas_list
                if ch_mode_count == 3:
                    probas_list[6][ch_mode_val] = sum([1 for val in deck_values if val == sec_ch_mode_val])/deck_n
                    probas_list[7][ch_mode_val] = sum([1 for val in deck_values if val == ch_mode_val])/deck_n
                    probas_list[3][ch_mode_val] = 1 - sum([sum(l) for l in probas_list])
                    return probas_list
                # only remaining case is 1p
                probas_list[3][ch_mode_val] = sum([1 for val in deck_values if val == ch_mode_val])/deck_n
                for val in ch_values[ch_values != ch_mode_val]:
                    probas_list[2][val] = sum([1 for i in deck_values if i == val])/deck_n
                probas_list[1][ch_mode_val] = 1 - sum([sum(l) for l in probas_list])
                return probas_list

            # new HC odds
            for val in [i for i in range(13) if i > max_ch_values]:
                probas_list[0][val] = sum([1 for i in deck_values if i == val]) / deck_n

            # straight odds
            if 12 in ch_values and sum(ch_values < 4) == (4 - n_remaining):
                ch_values[ch_values == 12] = -1
                max_ch_values = max(ch_values)
                min_ch_values = min(ch_values)
            elif max_ch_values < 4:
                deck_values[deck_values == 12] = -1

            if max_ch_values - min_ch_values == 4:
                missing_val = [x for x in range(min_ch_values, max_ch_values) if x not in ch_values][0]
                probas_list[4][max_ch_values] = sum([1 for val in deck_values if val == missing_val])/deck_n

            elif max_ch_values - min_ch_values == 3:
                if min_ch_values > -1:
                    probas_list[4][max_ch_values] = sum([1 for val in deck_values if val == (min_ch_values - 1)]) / deck_n
                if max_ch_values < 12:
                    probas_list[4][max_ch_values + 1] = sum([1 for val in deck_values if val == (max_ch_values + 1)]) / deck_n
                    probas_list[0][max_ch_values + 1] = 0

            if -1 in ch_values:
                ch_values[ch_values == -1] = 12
                max_ch_values = max(ch_values)
                min_ch_values = min(ch_values)
            elif max_ch_values < 4:
                deck_values[deck_values == -1] = 12

            # flush odds
            if len(np.unique(ch_suits)) == 1:
                n_tot_cards = sum([1 for val in deck_suits if val == ch_suits[0]])
                over_cards = deck[(deck > max(current_hand)) & (deck < 13 * (ch_suits[0] + 1))]
                for over_card in over_cards:
                    val = over_card % 13
                    probas_list[5][val] = 1 / deck_n
                    probas_list[0][val] *= (sum([1 for i in deck_values if i == val]) - 1)/(sum([1 for i in deck_values if i == val]))
                probas_list[5][max_ch_values] = (n_tot_cards - len(over_cards)) / deck_n

            # pair odds
            for val in ch_values:
                probas_list[1][val] = sum([1 for i in deck_values if i == val]) / deck_n

            # existing HC odds
            probas_list[0][max_ch_values] = 1 - sum([sum(l) for l in probas_list])
            return probas_list

        if n_remaining == 2:
            extra_pair_odds_list = [sum([1 for i in deck_values if i == val]) for val in range(13)]
            extra_pair_odds_list = [val * (val - 1) / (deck_n * (deck_n - 1)) for val in extra_pair_odds_list]
            for val in ch_values:
                extra_pair_odds_list[val] = 0

            if ch_mode_count >= 2:
                if ch_mode_count == 3:
                    probas_list[7][ch_mode_val] = sum([1 for val in deck_values if val == ch_mode_val]) / deck_n + \
                                                    (1 - sum([1 for val in deck_values if val == ch_mode_val]) / deck_n) * \
                                                    (sum([1 for val in deck_values if val == ch_mode_val]) / (deck_n - 1))

                    probas_list[6][ch_mode_val] = sum(extra_pair_odds_list)
                    probas_list[3][ch_mode_val] = 1 - sum([sum(l) for l in probas_list])
                    return probas_list

                # only remaining case is 1p
                # tp
                probas_list[2][max(sec_ch_mode_val, ch_mode_val)] = 2 * \
                          (sum([1 for val in deck_values if val == sec_ch_mode_val]) / deck_n) * \
                          (sum([1 for x in deck_values if x not in ch_values]) / (deck_n - 1))

                for val in range(13):
                    if val not in ch_values:
                        if val > ch_mode_val:
                            probas_list[2][val] = extra_pair_odds_list[val]
                        else:
                            probas_list[2][ch_mode_val] += extra_pair_odds_list[val]

                # toak
                probas_list[3][ch_mode_val] = 2 * (sum([1 for val in deck_values if val == ch_mode_val]) / deck_n) * \
                          (sum([1 for x in deck_values if x not in ch_values]) / (deck_n - 1))
                # fh
                probas_list[6][sec_ch_mode_val] = (sum([1 for val in deck_values if val == sec_ch_mode_val]) / deck_n) * \
                          ((sum([1 for val in deck_values if val == sec_ch_mode_val]) - 1) / (deck_n - 1))

                probas_list[6][ch_mode_val] = 2 * (sum([1 for val in deck_values if val == sec_ch_mode_val]) / deck_n) * \
                          (sum([1 for val in deck_values if val == ch_mode_val]) / (deck_n - 1))
                # foak
                probas_list[7][ch_mode_val] = (sum([1 for val in deck_values if val == ch_mode_val]) / deck_n) * \
                          ((sum([1 for val in deck_values if val == ch_mode_val]) - 1) / (deck_n - 1))

                probas_list[1][ch_mode_val] = 1 - sum([sum(l) for l in probas_list])
                return probas_list

            # new HC odds
            tot_nr_cards = sum([1 for x in deck_values if x not in ch_values])
            over_cards = [i for i in range(12,-1,-1) if i > max_ch_values]
            cumulative_overs = 0
            for i in range(len(over_cards)):
                val = over_cards[i]
                n_overs = sum([1 for i in deck_values if i == val])
                probas_list[0][val] = 2 * (n_overs / deck_n) * ((tot_nr_cards - cumulative_overs) / (deck_n - 1))
                cumulative_overs += n_overs

            # straight odds
            if 12 in ch_values and sum(ch_values < 4) == (4 - n_remaining):
                ch_values[ch_values == 12] = -1
                max_ch_values = max(ch_values)
                min_ch_values = min(ch_values)
            elif max_ch_values < 4:
                deck_values[deck_values == 12] = -1

            if max_ch_values - min_ch_values == 4:
                missing_vals = [x for x in range(min_ch_values, max_ch_values) if x not in ch_values]
                probas_list[4][max_ch_values] = 2 * (sum([1 for val in deck_values if val == missing_vals[0]]) / deck_n) * \
                                                   (sum([1 for val in deck_values if val == missing_vals[1]]) / (deck_n - 1))

            elif max_ch_values - min_ch_values == 3:
                missing_val = [x for x in range(min_ch_values, max_ch_values) if x not in ch_values][0]
                if min_ch_values > -1:
                    probas_list[4][max_ch_values] = 2 * (sum([1 for val in deck_values if val == missing_val]) / deck_n) * \
                                                       (sum([1 for val in deck_values if val == (min_ch_values - 1)]) / (deck_n - 1))
                if max_ch_values < 12:
                    probas_list[4][max_ch_values + 1] = 2 * (sum([1 for val in deck_values if val == missing_val]) / deck_n) * \
                                                       (sum([1 for val in deck_values if val == (max_ch_values + 1)]) / (deck_n - 1))
                    probas_list[0][max_ch_values + 1] = probas_list[0][max_ch_values + 1] - probas_list[4][max_ch_values + 1]

            elif max_ch_values - min_ch_values == 2:
                if min_ch_values > 0:
                    probas_list[4][max_ch_values] = 2 * (sum([1 for val in deck_values if val == (min_ch_values - 2)]) / deck_n) * \
                                                       (sum([1 for val in deck_values if val == (min_ch_values - 1)]) / (deck_n - 1))
                if max_ch_values < 12 and min_ch_values > -1:
                    probas_list[4][max_ch_values + 1] = 2 * (sum([1 for val in deck_values if val == (min_ch_values - 1)]) / deck_n) * \
                                                       (sum([1 for val in deck_values if val == (max_ch_values + 1)]) / (deck_n - 1))
                    probas_list[0][max_ch_values + 1] = probas_list[0][max_ch_values + 1] - probas_list[4][max_ch_values + 1]
                if max_ch_values < 11:
                    probas_list[4][max_ch_values + 2] = 2 * (sum([1 for val in deck_values if val == (max_ch_values + 1)]) / deck_n) * \
                                                       (sum([1 for val in deck_values if val == (max_ch_values + 2)]) / (deck_n - 1))
                    probas_list[0][max_ch_values + 2] = probas_list[0][max_ch_values + 2] - probas_list[4][max_ch_values + 2]

            if -1 in ch_values:
                ch_values[ch_values == -1] = 12
                max_ch_values = max(ch_values)
                min_ch_values = min(ch_values)
            elif max_ch_values < 4:
                deck_values[deck_values == -1] = 12

            # flush odds
            if len(np.unique(ch_suits)) == 1:
                n_tot_cards = sum([1 for val in deck_suits if val == ch_suits[0]])
                over_cards = deck[(deck > max(current_hand)) & (deck < 13 * (ch_suits[0] + 1))]
                for i in range(len(over_cards)):
                    val = over_cards[i] % 13
                    probas_list[5][val] = 2 * (1 / deck_n) * (n_tot_cards - i - 1) / (deck_n - 1)
                    probas_list[0][val] = max(0, probas_list[0][val] - probas_list[5][val])

                probas_list[5][max_ch_values] = ((n_tot_cards - len(over_cards)) / deck_n) * \
                                                   (n_tot_cards - len(over_cards) - 1) / (deck_n - 1)

            # pair odds
            for val in ch_values:
                probas_list[1][val] = 2 * (sum([1 for i in deck_values if i == val]) / deck_n) * \
                            (sum([1 for x in deck_values if x not in ch_values]) / (deck_n - 1))
            for val in range(13):
                if val not in ch_values:
                    probas_list[1][val] = extra_pair_odds_list[val]

            # toak odds
            for val in ch_values:
                probas_list[3][val] = (sum([1 for i in deck_values if i == val]) / deck_n) * ((sum([1 for i in deck_values if i == val]) - 1) / (deck_n - 1))

            # tp odds
            probas_list[2][max_ch_values] = 2 * (sum([1 for val in deck_values if val == ch_values[0]]) / deck_n) * \
                      (sum([1 for val in deck_values if val == ch_values[1]]) / (deck_n - 1))
            probas_list[2][max_ch_values] += 2 * (sum([1 for val in deck_values if val == ch_values[1]]) / deck_n) * \
                      (sum([1 for val in deck_values if val == ch_values[2]]) / (deck_n - 1))
            probas_list[2][max_ch_values] += 2 * (sum([1 for val in deck_values if val == ch_values[0]]) / deck_n) * \
                      (sum([1 for val in deck_values if val == ch_values[2]]) / (deck_n - 1))

            # existing HC odds
            probas_list[0][max_ch_values] = 1 - sum([sum(l) for l in probas_list])
            return probas_list

        if n_remaining == 3:
            val_counts_list = [sum([1 for i in deck_values if i == val]) for val in range(13)]
            for val in ch_values:
                val_counts_list[val] = 0

            extra_pair_odds_list = [3 * val * (val - 1) * (sum([1 for x in deck_values if x not in ch_values and x != val]))
                                    / (deck_n * (deck_n - 1) * (deck_n - 2)) for val in val_counts_list]

            extra_toak_odds_list = [val * (val - 1) * (val - 2)
                                    / (deck_n * (deck_n - 1) * (deck_n - 2)) for val in val_counts_list]

            extra_two_unrelated_odds = sum([val * (sum([1 for x in deck_values if x not in ch_values]) - val)
                                            for val in val_counts_list]) / ((deck_n - 1) * (deck_n - 2))

            if ch_mode_count == 2:
                # tp odds
                for val in range(13):
                    if val not in ch_values:
                        if val > ch_mode_val:
                            probas_list[2][val] = extra_pair_odds_list[val]
                        else:
                            probas_list[2][ch_mode_val] += extra_pair_odds_list[val]

                # toak odds
                probas_list[3][ch_mode_val] = 3 * (sum([1 for val in deck_values if val == ch_mode_val]) / deck_n) * extra_two_unrelated_odds

                # fh odds
                for val in range(13):
                    if val not in ch_values:
                        probas_list[6][val] = extra_toak_odds_list[val]

                probas_list[6][ch_mode_val] = 3 * (sum([1 for val in deck_values if val == ch_mode_val]) / deck_n) * sum(extra_pair_odds_list)

                # foak odds
                probas_list[7][ch_mode_val] = 3 * (sum([1 for val in deck_values if val == ch_mode_val]) * (sum([1 for val in deck_values if val == ch_mode_val]) - 1) *
                     sum([1 for x in deck_values if x not in ch_values])) / (deck_n * (deck_n - 1) * (deck_n - 2))

                # current pair only odds
                probas_list[1][ch_mode_val] = 1 - sum([sum(l) for l in probas_list])
                return probas_list

            # new HC odds
            tot_nr_cards = sum([1 for x in deck_values if x not in ch_values])
            over_cards = [i for i in range(12,-1,-1) if i > max_ch_values]
            cumulative_overs = 0
            for i in range(len(over_cards)):
                val = over_cards[i]
                n_overs = sum([1 for i in deck_values if i == val])
                probas_list[0][val] = 2 * (n_overs / deck_n) * ((tot_nr_cards - cumulative_overs) / (deck_n - 1)) * \
                                      ((tot_nr_cards - cumulative_overs - 1) / (deck_n - 1))
                cumulative_overs += n_overs

            # straight odds
            if 12 in ch_values and sum(ch_values < 4) == (4 - n_remaining):
                ch_values[ch_values == 12] = -1
                max_ch_values = max(ch_values)
                min_ch_values = min(ch_values)
            elif max_ch_values < 4:
                deck_values[deck_values == 12] = -1

            if max_ch_values - min_ch_values == 4:
                missing_vals = [x for x in range(min_ch_values, max_ch_values) if x not in ch_values]
                probas_list[4][max_ch_values] = 3 * 2 * (sum([1 for val in deck_values if val == missing_vals[0]]) / deck_n) * \
                                                   (sum([1 for val in deck_values if val == missing_vals[1]]) / (deck_n - 1)) * \
                                                   (sum([1 for val in deck_values if val == missing_vals[2]]) / (deck_n - 2))

            elif max_ch_values - min_ch_values == 3:
                missing_vals = [x for x in range(min_ch_values, max_ch_values) if x not in ch_values]
                if min_ch_values > -1:
                    probas_list[4][max_ch_values] = 3 * 2 * (sum([1 for val in deck_values if val == missing_vals[0]]) / deck_n) * \
                                                   (sum([1 for val in deck_values if val == missing_vals[1]]) / (deck_n - 1)) * \
                                                   (sum([1 for val in deck_values if val == (min_ch_values - 1)]) / (deck_n - 2))
                if max_ch_values < 12:
                    probas_list[4][max_ch_values + 1] = 3 * 2 * (sum([1 for val in deck_values if val == missing_vals[0]]) / deck_n) * \
                                                   (sum([1 for val in deck_values if val == missing_vals[1]]) / (deck_n - 1)) * \
                                                   (sum([1 for val in deck_values if val == (max_ch_values + 1)]) / (deck_n - 2))
                    probas_list[0][max_ch_values + 1] = probas_list[0][max_ch_values + 1] - probas_list[4][max_ch_values + 1]

            elif max_ch_values - min_ch_values == 2:
                missing_val = [x for x in range(min_ch_values, max_ch_values) if x not in ch_values][0]

                if min_ch_values > 0:
                    probas_list[4][max_ch_values] = 3 * 2 * (sum([1 for val in deck_values if val == (min_ch_values - 2)]) / deck_n) * \
                                                       (sum([1 for val in deck_values if val == (min_ch_values - 1)]) / (deck_n - 1)) * \
                                                    (sum([1 for val in deck_values if val == missing_val]) / (deck_n - 2))
                if max_ch_values < 12 and min_ch_values > -1:
                    probas_list[4][max_ch_values + 1] = 2 * (sum([1 for val in deck_values if val == (min_ch_values - 1)]) / deck_n) * \
                                                       (sum([1 for val in deck_values if val == (max_ch_values + 1)]) / (deck_n - 1)) * \
                                                    (sum([1 for val in deck_values if val == missing_val]) / (deck_n - 2))
                    probas_list[0][max_ch_values + 1] = probas_list[0][max_ch_values + 1] - probas_list[4][max_ch_values + 1]
                if max_ch_values < 11:
                    probas_list[4][max_ch_values + 2] = 2 * (sum([1 for val in deck_values if val == (max_ch_values + 1)]) / deck_n) * \
                                                       (sum([1 for val in deck_values if val == (max_ch_values + 2)]) / (deck_n - 1)) * \
                                                    (sum([1 for val in deck_values if val == missing_val]) / (deck_n - 2))
                    probas_list[0][max_ch_values + 2] = probas_list[0][max_ch_values + 2] - probas_list[4][max_ch_values + 2]

            elif max_ch_values - min_ch_values == 1:
                if min_ch_values > 1:
                    probas_list[4][max_ch_values] = 3 * 2 * (sum([1 for val in deck_values if val == (min_ch_values - 3)]) / deck_n) * \
                                                       (sum([1 for val in deck_values if val == (min_ch_values - 2)]) / (deck_n - 1)) * \
                                                    (sum([1 for val in deck_values if val == (min_ch_values - 1)]) / (deck_n - 2))

                if max_ch_values < 12 and min_ch_values > 0:
                    probas_list[4][max_ch_values + 1] = 2 * (sum([1 for val in deck_values if val == (min_ch_values - 2)]) / deck_n) * \
                                                       (sum([1 for val in deck_values if val == (min_ch_values - 1)]) / (deck_n - 1)) * \
                                                    (sum([1 for val in deck_values if val == (max_ch_values + 1)]) / (deck_n - 2))
                    probas_list[0][max_ch_values + 1] = probas_list[0][max_ch_values + 1] - probas_list[4][max_ch_values + 1]

                if max_ch_values < 11 and min_ch_values > -1:
                    probas_list[4][max_ch_values + 2] = 2 * (sum([1 for val in deck_values if val == (min_ch_values - 1)]) / deck_n) * \
                                                       (sum([1 for val in deck_values if val == (max_ch_values + 1)]) / (deck_n - 1)) * \
                                                    (sum([1 for val in deck_values if val == (max_ch_values + 2)]) / (deck_n - 2))
                    probas_list[0][max_ch_values + 2] = probas_list[0][max_ch_values + 2] - probas_list[4][max_ch_values + 2]

                if max_ch_values < 10:
                    probas_list[4][max_ch_values + 3] = 2 * (sum([1 for val in deck_values if val == (max_ch_values + 1)]) / deck_n) * \
                                                       (sum([1 for val in deck_values if val == (max_ch_values + 2)]) / (deck_n - 1)) * \
                                                        (sum([1 for val in deck_values if val == (max_ch_values + 3)]) / (deck_n - 2))
                    probas_list[0][max_ch_values + 3] = probas_list[0][max_ch_values + 3] - probas_list[4][max_ch_values + 3]

            if -1 in ch_values:
                ch_values[ch_values == -1] = 12
                max_ch_values = max(ch_values)
                min_ch_values = min(ch_values)
            elif max_ch_values < 4:
                deck_values[deck_values == -1] = 12

            # flush odds
            if len(np.unique(ch_suits)) == 1:
                n_tot_cards = sum([1 for val in deck_suits if val == ch_suits[0]])
                over_cards = deck[(deck > max(current_hand)) & (deck < 13 * (ch_suits[0] + 1))]
                for i in range(len(over_cards)):
                    val = over_cards[i] % 13
                    probas_list[5][val] = 3 * (1 / deck_n) * \
                                            (n_tot_cards - i - 1) / (deck_n - 1) * \
                                            (n_tot_cards - i - 2) / (deck_n - 2)
                    probas_list[0][val] = max(0, probas_list[0][val] - probas_list[5][val])

                probas_list[5][max_ch_values] = ((n_tot_cards - len(over_cards)) / deck_n) * \
                                                   (n_tot_cards - len(over_cards) - 1) / (deck_n - 1) * \
                                                   (n_tot_cards - len(over_cards) - 2) / (deck_n - 2)

            # pair odds
            for val in range(13):
                if val in ch_values:
                    probas_list[1][val] = 3 * (sum([1 for i in deck_values if i == val]) / deck_n) * extra_two_unrelated_odds
                else:
                    probas_list[1][val] = extra_pair_odds_list[val]

            # tp odds - 2 existing cards
            probas_list[2][max_ch_values] = 3 * 2 * (sum([1 for val in deck_values if val == ch_values[0]]) / deck_n) * \
                                            (sum([1 for val in deck_values if val == ch_values[1]]) / (deck_n - 1)) * \
                                            (sum([1 for x in deck_values if x not in ch_values]) / (deck_n - 2))

            # tp odds - 1 existing + extra pair
            for ch_val in ch_values:
                for extra_val in range(13):
                    # we make up for 'extra_pair_vec' having already accounted for an extra unrelated card
                    if extra_val in ch_values:
                        continue
                    if ch_val > extra_val:
                        probas_list[2][ch_val] += 3 * extra_pair_odds_list[extra_val] * \
                                                    (sum([1 for val in deck_values if val == ch_val])) / \
                                                    (sum([1 for x in deck_values if x not in ch_values]))
                    else:
                        probas_list[2][extra_val] += 3 * extra_pair_odds_list[extra_val] * \
                                                    (sum([1 for val in deck_values if val == ch_val])) / \
                                                    (sum([1 for x in deck_values if x not in ch_values]))

            # toak odds
            for val in ch_values:
                probas_list[3][val] = 3 * (sum([1 for i in deck_values if i == val]) / deck_n) * \
                                        ((sum([1 for i in deck_values if i == val]) - 1) / (deck_n - 1)) * \
                                        (sum([1 for x in deck_values if x not in ch_values]) / (deck_n - 2))

            for val in range(13):
                if val not in ch_values:
                    probas_list[3][val] = extra_toak_odds_list[val]

            # fh odds
            probas_list[6][ch_values[0]] = 3 * (sum([1 for val in deck_values if val == ch_values[0]]) / deck_n) * \
                                             ((sum([1 for val in deck_values if val == ch_values[0]]) - 1) / (deck_n - 1)) * \
                                             ((sum([1 for val in deck_values if val == ch_values[1]])) / (deck_n - 2))

            probas_list[6][ch_values[1]] = 3 * (sum([1 for val in deck_values if val == ch_values[1]]) / deck_n) * \
                                             ((sum([1 for val in deck_values if val == ch_values[1]]) - 1) / (deck_n - 1)) * \
                                             ((sum([1 for val in deck_values if val == ch_values[0]])) / (deck_n - 2))
            # foak odds
            for val in ch_values:
                probas_list[7][val] = (sum([1 for i in deck_values if i == val]) / deck_n) * \
                                        ((sum([1 for i in deck_values if i == val]) - 1) / (deck_n - 1)) * \
                                        ((sum([1 for i in deck_values if i == val]) - 2) / (deck_n - 2))

            # existing HC odds
            probas_list[0][max_ch_values] = 1 - sum([sum(l) for l in probas_list])
            return probas_list

    def calc_hand_win_proba(self, cpu_probas, opp_probas, cpu_hand, opp_hand, deck):
        if self.proba_method == "extended":
            return self.calc_extended_hand_win_proba(cpu_probas, opp_probas, cpu_hand, opp_hand, deck)
        return self.calc_short_hand_win_proba(cpu_probas, opp_probas, cpu_hand, opp_hand, deck)

    def calc_extended_hand_win_proba(self, cpu_probas, opp_probas, cpu_hand, opp_hand, deck):
        # cpu_probas, opp_probas, cpu_hand, opp_hand, deck = cpu_decision_probas_list[3], opp_probas_list[3], vec, opp_vec, deck_left
        if sum(cpu_hand == -1) == 0:
            win_count = 0
            for i in range(len(deck)):
                poss_opp_hand = opp_hand.copy()
                poss_opp_hand[4] = deck[i]
                poss_opp_probas = self.calc_hand_probas(poss_opp_hand, deck)
                if np.argmax(cpu_probas) > np.argmax(poss_opp_probas):
                    win_count += 1
                elif np.argmax(cpu_probas) == np.argmax(poss_opp_probas):
                    if self.get_curr_leader(cpu_hand, poss_opp_hand) == "cpu":
                        win_count += 1
            return win_count / len(deck)

        opp_cum_probas = np.cumsum(opp_probas)
        cpu_win_proba = sum([cpu_probas[i] * opp_cum_probas[i - 1] + 0.5 * cpu_probas[i] * opp_probas[i] for i in range(1, 9*13)])
        return cpu_win_proba

    def calc_short_hand_win_proba(self, cpu_probas, opp_probas, cpu_hand, opp_hand, deck):
        if sum(cpu_hand == -1) == 0:
            win_count = 0
            for i in range(len(deck)):
                poss_opp_hand = opp_hand.copy()
                poss_opp_hand[4] = deck[i]
                poss_opp_probas = self.calc_hand_probas(poss_opp_hand, deck)
                if np.argmax(cpu_probas) > np.argmax(poss_opp_probas):
                    win_count += 1
                elif np.argmax(cpu_probas) == np.argmax(poss_opp_probas):
                    if self.get_curr_leader(cpu_hand, poss_opp_hand) == "cpu":
                        win_count += 1
            return win_count / len(deck)

        # tie_break_win_probas = self.win_tie_probas(cpu_probas, opp_probas, cpu_hand, opp_hand, deck)
        tie_break_win_probas = np.array(cpu_probas) * opp_probas * 0.5  # tie_break_win_probas
        opp_cum_probas = np.cumsum(opp_probas)
        cpu_win_proba = sum([cpu_probas[i] * opp_cum_probas[i - 1] for i in range(1, 9)]) + sum(tie_break_win_probas)
        return cpu_win_proba

    def win_tie_probas(self, cpu_probas, opp_probas, cpu_hand, opp_hand, deck):
        res_vec = [0.5] * 9
        deck_values = np.array(deck) % 13
        deck_n = len(deck)
        n_remaining = sum(cpu_hand == -1)

        cpu_hand_vals = np.array([val % 13 if val >= 0 else -1 for val in cpu_hand])
        opp_hand_vals = np.array([val % 13 if val >= 0 else -1 for val in opp_hand])

        lead = self.get_curr_leader(cpu_hand_vals, opp_hand_vals)

        ### both don't have pairs yet
        # high card calc:
        if cpu_probas[0] > 0 and opp_probas[0] > 0:
            if lead == "cpu":
                over_cards = sum(deck_values > max(cpu_hand_vals))
                res_vec[0] = 1 - ((over_cards / deck_n) ** (1 / n_remaining)) * 0.5
            elif lead == "opp":
                over_cards = sum(deck_values > max(opp_hand_vals))
                res_vec[0] = ((over_cards / deck_n) ** (1 / n_remaining)) * 0.5

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
        elif cpu_probas[0] > 0 and cpu_probas[1] > 0:
            if sum(np.isin(deck_values, cpu_hand_vals)) == 0:
                res_vec[1] = 0.2
            else:
                res_vec[1] = sum(np.isin(deck_values, cpu_hand_vals[cpu_hand_vals > opp_hand_vals[0]])) \
                             / sum(np.isin(deck_values, cpu_hand_vals))
                res_vec[1] += 0.5 * sum(np.isin(deck_values, cpu_hand_vals[cpu_hand_vals == opp_hand_vals[0]])) \
                              / sum(np.isin(deck_values, cpu_hand_vals))

        # cpu has pair, opp doesn't
        elif opp_probas[0] > 0 and opp_probas[1] > 0:
            if sum(np.isin(deck_values, opp_hand_vals)) == 0:
                res_vec[1] = 0.8
            else:
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

    def optimistic_probas(self, p_vec, level):
        new_vec = np.array(p_vec).copy()
        n = 5 - level
        if sum(new_vec > 0) == 1:
            return p_vec
        for i in np.flip(range(2, 9)):
            new_vec[i] = min((1 - (1 - (new_vec[i] ** (1 / n))) ** (2.5)) ** n, 1 - sum(new_vec[i:]))
            try:
                new_vec[:i] *= (1 - sum(new_vec[i:])) / sum(new_vec[:i])
            except Exception as exc:
                print("bla")
        return new_vec

    def convd_probas(self, p_vec, level):
        if level < 2:
            level = 2
        new_vec = [0] * 9
        for i in range(9):
            if p_vec[i] == 0 or p_vec[i] == 1:
                new_vec[i] = p_vec[i]
                continue
            v = round(max(p_vec[i], 0.0001), 4)
            d = (10 ** -(ceil(-log10(v))))
            ind = 10 * level + i + round((v // d) * d, 4)
            if ind not in self.proba_conv_df.index:
                print("bla")
            new_vec[i] = self.proba_conv_df[ind]
            new_vec = new_vec/sum(new_vec)
        return new_vec

    def rearrange_hand(self, hand):
        hand_vals = np.array([val % 13 if val >= 0 else -1 for val in hand])
        hand_vals = list(hand_vals[hand_vals >= 0])
        hand_order = np.flip(np.argsort([i + (hand_vals.count(i) - 1) * 13 for i in hand_vals]))
        rearrranged_hand = np.array(hand_vals)[hand_order]
        return np.concatenate([rearrranged_hand, [-1] * (5 - len(hand_vals))])

    def get_curr_leader(self, cpu_hand, opp_hand):
        cpu_hand_vals = self.rearrange_hand(cpu_hand)
        opp_hand_vals = self.rearrange_hand(opp_hand)
        for i in range(5):
            if cpu_hand_vals[i] > opp_hand_vals[i]:
                return "cpu"
            elif cpu_hand_vals[i] < opp_hand_vals[i]:
                return "opp"
        return "tie"

    def log_data(self, probas_df, dec_win_probas_list):
        probas_df["game_id"] = self.game_id
        probas_df["decision_id"] = self.decision_id
        self.probas_data[self.game_id * 12 + self.decision_id] = probas_df
        self.decisions_data[self.game_id * 12 + self.decision_id] = dec_win_probas_list

        self.decision_id += 1
        if self.decision_id == 12:
            self.decision_id = 0
            self.game_id += 1

    def get_data_as_dfs(self):
        return pd.concat(list(self.probas_data.values())), pd.DataFrame.from_dict(self.decisions_data)

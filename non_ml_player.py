import numpy as np
import pandas as pd
from poisson_binomial import PoissonBinomial
from scipy.stats import mode
from math import log10, ceil

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
            self.proba_conv_df["proba"] = self.proba_conv_df["proba"].apply(lambda x: round(x,4))
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
            if self.save_data:
                interwined_probas_list = [item for trio in zip(cpu_probas_list, cpu_decision_probas_list, opp_probas_list) for item in trio]
                probas_data = pd.DataFrame(interwined_probas_list, index=[["cpu", "cpu+card", "opp"][i % 3] + str(i//3) for i in range(15)])
                self.log_data(probas_data, dec_ovr_win_probas)

            return decision

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
            if max(ch_values) - min(ch_values) == 5:
                if len(np.unique(ch_suits)) == 1:
                    return [0, 0, 0, 0, 0, 0, 0, 0, 1]
                else:
                    return [0, 0, 0, 0, 1, 0, 0, 0, 0]
            if 12 in ch_values:
                temp_ch_values = ch_values.copy()
                temp_ch_values[temp_ch_values == 12] = -1
                if max(ch_values) - min(ch_values) == 5:
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
                           range(3)]) / 2
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
                               sum(unseen_cards_counts == 4) * 4 * 3) / (
                                          deck_n * (deck_n - 1) * (deck_n - 2)) - extra_toak_odds

            extra_two_unrelated_odds = sum([i * sum(unseen_cards_counts == i) *
                                            (sum([1 for x in deck_values if x not in ch_values]) - i) for i in
                                            range(1, 5)]) / \
                                       ((deck_n - 1) * (deck_n - 2))

            if ch_mode_count == 2:
                tp_odds = extra_pair_odds

                toak_odds = 3 * (sum([1 for x in deck_values if x == ch_mode_val]) / deck_n) * extra_two_unrelated_odds

                foak_odds = 3 * (sum([1 for x in deck_values if x == ch_mode_val]) / deck_n) * \
                            (max(sum([1 for x in deck_values if x == ch_mode_val]) - 1, 0) / (deck_n - 1)) * \
                            (sum([1 for x in deck_values if x != ch_mode_val]) / (deck_n - 2))

                fh_odds = 3 * (
                            sum([1 for x in deck_values if x == ch_mode_val]) / deck_n) * extra_pair_odds + extra_toak_odds

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
                    straight_odds += np.prod([sum([1 for x in deck_values if x == needed_func[i](ch_values) + i]) for i in
                                              poss_vals[s:(s + 3)]]) / \
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
            pair_odds = 3 * (sum(
                [1 for x in deck_values if x in ch_values]) / deck_n) * extra_two_unrelated_odds + extra_pair_odds

            tp_odds = 3 * 0.5 * sum([(sum([1 for x in deck_values if x == ch_values[i]]) / deck_n) * \
                                     (sum([1 for x in deck_values if x in np.delete(ch_values, i)]) / (deck_n - 1)) for i in
                                     range(2)]) * \
                      (sum([1 for x in deck_values if x not in ch_values]) / (deck_n - 2)) + \
                      3 * (sum([1 for x in deck_values if x in ch_values]) / deck_n) * extra_pair_odds

            toak_odds = 3 * sum([(sum([1 for x in deck_values if x == ch_values[i]]) / deck_n) * \
                                 (max(sum([1 for x in deck_values if x == ch_values[i]]) - 1, 0) / (deck_n - 1)) for i in
                                 range(2)]) * \
                        (sum([1 for x in deck_values if x not in ch_values]) / (deck_n - 2)) + extra_toak_odds

            fh_odds = 3 * sum([(sum([1 for x in deck_values if x == ch_values[i]]) / deck_n) * \
                               (max(sum([1 for x in deck_values if x == ch_values[i]]) - 1, 0) / (deck_n - 1)) * \
                               (sum([1 for x in deck_values if x == ch_values[1 - i]]) / (deck_n - 2)) for i in range(2)])

            foak_odds = sum([(sum([1 for x in deck_values if x == ch_values[i]]) / deck_n) * \
                             (max(sum([1 for x in deck_values if x == ch_values[i]]) - 1, 0) / (deck_n - 1)) * \
                             (max(sum([1 for x in deck_values if x == ch_values[i]]) - 2, 0) / (deck_n - 2)) for i in
                             range(2)])

            high_card_odds = 1 - straight_odds - flush_odds - pair_odds - tp_odds - toak_odds - fh_odds - foak_odds

            return [high_card_odds, pair_odds, tp_odds, toak_odds, straight_odds, flush_odds, fh_odds, foak_odds, straight_odds * flush_odds]

    def calc_hand_win_proba(self, cpu_probas, opp_probas, cpu_hand, opp_hand, deck):
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

        tie_break_win_probas = self.win_tie_probas(cpu_probas, opp_probas, cpu_hand, opp_hand, deck)
        tie_break_win_probas = np.array(cpu_probas) * opp_probas * tie_break_win_probas
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

        lead, lead_ind = self.get_curr_leader(cpu_hand_vals, opp_hand_vals)

        ### both don't have pairs yet
        # high card calc:
        if cpu_probas[0] > 0 and opp_probas[0] > 0:
            if lead == "cpu":
                over_cards = sum(deck_values > cpu_hand_vals[lead_ind])
                res_vec[0] = 1 - ((over_cards / deck_n) ** (1 / n_remaining)) * 0.5
            elif lead == "opp":
                over_cards = sum(deck_values > opp_hand_vals[lead_ind])
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

    def calc_hand_probas(self, current_hand, deck):
        if self.proba_method == "convd_probas":
            return self.convd_probas(self.calc_random_hand_probas(current_hand, deck), 4 - (len(deck) - 2) // 10)
        if self.proba_method == "optimistic":
            return self.optimistic_probas(self.calc_random_hand_probas(current_hand, deck), 4 - (len(deck) - 2) // 10)

        return self.calc_random_hand_probas(current_hand, deck)

    def rearrange_hand(self, hand):
        hand_vals = np.array([val % 13 if val > 0 else -1 for val in hand])
        hand_vals = list(hand_vals[hand_vals >= 0])
        hand_order = np.flip(np.argsort([i + (hand_vals.count(i) - 1) * 13 for i in hand_vals]))
        rearrranged_hand = np.array(hand_vals)[hand_order]
        return np.concatenate([rearrranged_hand, [-1] * (5 - len(hand_vals))])

    def get_curr_leader(self, cpu_hand, opp_hand):
        cpu_hand_vals = self.rearrange_hand(cpu_hand)
        opp_hand_vals = self.rearrange_hand(opp_hand)
        for i in range(5):
            if cpu_hand_vals[i] > opp_hand_vals[i]:
                return "cpu", i
            elif cpu_hand_vals[i] < opp_hand_vals[i]:
                return "opp", i
        return "tie", -1

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

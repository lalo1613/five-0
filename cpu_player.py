import numpy as np
from poisson_binomial import PoissonBinomial
from scipy.stats import mode


class CpuPlayer():
    """
    This is a shortened version, see full_cpu_player.py.
    """

    def __init__(self, proba_method="heuristic", fifth_card_exact_calc=False):
        self.proba_method = proba_method
        self.fifth_card_exact_calc = fifth_card_exact_calc
        self.decision_id = 0

    def get_choice(self, level, turn, player, deck, pot_card):
        # getting allowed decision options
        options = [x for x in range(5) if player[turn][level][x] == -1]

        # turning lists to np arrays, extracting player hands
        temp_player = np.array(player)
        deck_left = np.array(deck)
        cpu_hands = [temp_player[turn, :, i].copy() for i in range(5)]
        opp_hands = [temp_player[1 - turn, :, i].copy() for i in range(5)]

        # first card - currently placed heuristically in strongest potential column
        if level < 2:
            cpu_decision_probas_list = [[-1] * 9*13] * 5
            for i in range(5):
                vec = cpu_hands[i].copy()
                if i in options:
                    vec[level] = pot_card
                    cpu_decision_probas_list[i] = self.calc_hand_probas(vec, deck_left)

            expected_val = [sum([l[x] * (x//13)**1.5 for x in range(9*13)]) for l in cpu_decision_probas_list]
            decision = int(np.argmax(expected_val))
            return decision

        # calculating hand probabilities for all 10 existing columns
        cpu_probas_list = [self.calc_hand_probas(cpu_hands[i], deck_left) for i in range(5)]
        opp_probas_list = [self.calc_hand_probas(opp_hands[i], deck_left) for i in range(5)]

        # calculate probability of winning each hand as is,
        # and probability of winning each possible option hand if the card is placed there.
        cpu_decision_probas_list = [[-1] * 9*13] * 5
        as_is_hand_win_probas, decision_hand_win_probas = [-1] * 5, [-1] * 5
        for i in range(5):
            vec = cpu_hands[i].copy()
            opp_vec = opp_hands[i].copy()
            as_is_hand_win_probas[i] = self.calc_hand_win_proba(cpu_probas_list[i], opp_probas_list[i],
                                                                vec, opp_vec, deck_left)
            if i in options:
                vec[level] = pot_card
                cpu_decision_probas_list[i] = self.calc_hand_probas(vec, deck_left)
                decision_hand_win_probas[i] = self.calc_hand_win_proba(cpu_decision_probas_list[i], opp_probas_list[i],
                                                                       vec, opp_vec, deck_left)

        # look at the probability of winning the game with each decision made
        dec_game_win_probas = [-1] * 5
        for i in options:
            post_dec_vec = as_is_hand_win_probas.copy()
            post_dec_vec[i] = decision_hand_win_probas[i]
            pb = PoissonBinomial(post_dec_vec)
            dec_game_win_probas[i] = pb.x_or_more(3)

        # choose the option that maximized game win proba
        decision = int(np.argmax(dec_game_win_probas))

        # keeping track of decision order
        self.decision_id += 1
        if self.decision_id == 12:
            self.decision_id = 0

        return decision

    def calc_hand_probas(self, current_hand, deck):
        return self.calc_heuristic_adj_probas([item for sub_l in self.calc_extended_random_hand_probas(current_hand, deck) for item in sub_l],
                                                  sum(current_hand == -1))

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
                probas_list[0][12] = probas_list[0][12] - probas_list[4][3]
            elif max_ch_values < 4:
                deck_values[deck_values == -1] = 12
                probas_list[0][12] = probas_list[0][12] - probas_list[4][3]

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
                cumulative_overs += n_overs
                probas_list[0][val] = 2 * (n_overs / deck_n) * ((tot_nr_cards - cumulative_overs) / (deck_n - 1))

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
                probas_list[0][12] = probas_list[0][12] - probas_list[4][3]
            elif max_ch_values < 4:
                deck_values[deck_values == -1] = 12
                probas_list[0][12] = probas_list[0][12] - probas_list[4][3]

            # flush odds
            if len(np.unique(ch_suits)) == 1:
                n_tot_cards = sum([1 for val in deck_suits if val == ch_suits[0]])
                over_cards = deck[(deck > max(current_hand)) & (deck < 13 * (ch_suits[0] + 1))]
                for i in range(len(over_cards)):
                    val = over_cards[i] % 13
                    probas_list[5][val] = 2 * (1 / deck_n) * (n_tot_cards - i - 1) / (deck_n - 1)
                    probas_list[0][val] = probas_list[0][val] - probas_list[5][val]

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

            extra_pair_odds_list = [3 * val * (val - 1) * (sum([1 for x in deck_values if x not in ch_values]) - val) / (deck_n * (deck_n - 1) * (deck_n - 2)) for val in val_counts_list]

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
            over_cards = [i for i in range(12, -1, -1) if i > max_ch_values]
            cumulative_overs = 0
            for i in range(len(over_cards)):
                val = over_cards[i]
                n_overs = sum([1 for i in deck_values if i == val])
                cumulative_overs += n_overs
                probas_list[0][val] = 3 * (n_overs / deck_n) * ((tot_nr_cards - cumulative_overs) / (deck_n - 1)) * \
                                      ((tot_nr_cards - cumulative_overs - 1) / (deck_n - 2)) * \
                                (1 - sum([v * (v-1) / (max((tot_nr_cards - cumulative_overs), 1) * max((tot_nr_cards - cumulative_overs - 1), 1)) for v in val_counts_list[:val]]))

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
                    probas_list[4][max_ch_values] = 3 * (sum([1 for val in deck_values if val == (min_ch_values - 2)]) / deck_n) * \
                                                       (sum([1 for val in deck_values if val == (min_ch_values - 1)]) / (deck_n - 1)) * \
                                                    (sum([1 for val in deck_values if val == missing_val]) / (deck_n - 2))
                if max_ch_values < 12 and min_ch_values > -1:
                    probas_list[4][max_ch_values + 1] = 3 * (sum([1 for val in deck_values if val == (min_ch_values - 1)]) / deck_n) * \
                                                       (sum([1 for val in deck_values if val == (max_ch_values + 1)]) / (deck_n - 1)) * \
                                                    (sum([1 for val in deck_values if val == missing_val]) / (deck_n - 2))
                    probas_list[0][max_ch_values + 1] = probas_list[0][max_ch_values + 1] - probas_list[4][max_ch_values + 1]
                if max_ch_values < 11:
                    probas_list[4][max_ch_values + 2] = 3 * (sum([1 for val in deck_values if val == (max_ch_values + 1)]) / deck_n) * \
                                                       (sum([1 for val in deck_values if val == (max_ch_values + 2)]) / (deck_n - 1)) * \
                                                    (sum([1 for val in deck_values if val == missing_val]) / (deck_n - 2))
                    probas_list[0][max_ch_values + 2] = probas_list[0][max_ch_values + 2] - probas_list[4][max_ch_values + 2]

            elif max_ch_values - min_ch_values == 1:
                if min_ch_values > 1:
                    probas_list[4][max_ch_values] = 3 * 2 * (sum([1 for val in deck_values if val == (min_ch_values - 3)]) / deck_n) * \
                                                       (sum([1 for val in deck_values if val == (min_ch_values - 2)]) / (deck_n - 1)) * \
                                                    (sum([1 for val in deck_values if val == (min_ch_values - 1)]) / (deck_n - 2))

                if max_ch_values < 12 and min_ch_values > 0:
                    probas_list[4][max_ch_values + 1] = 3 * 2 * (sum([1 for val in deck_values if val == (min_ch_values - 2)]) / deck_n) * \
                                                       (sum([1 for val in deck_values if val == (min_ch_values - 1)]) / (deck_n - 1)) * \
                                                    (sum([1 for val in deck_values if val == (max_ch_values + 1)]) / (deck_n - 2))
                    probas_list[0][max_ch_values + 1] = probas_list[0][max_ch_values + 1] - probas_list[4][max_ch_values + 1]

                if max_ch_values < 11 and min_ch_values > -1:
                    probas_list[4][max_ch_values + 2] = 3 * 2 * (sum([1 for val in deck_values if val == (min_ch_values - 1)]) / deck_n) * \
                                                       (sum([1 for val in deck_values if val == (max_ch_values + 1)]) / (deck_n - 1)) * \
                                                    (sum([1 for val in deck_values if val == (max_ch_values + 2)]) / (deck_n - 2))
                    probas_list[0][max_ch_values + 2] = probas_list[0][max_ch_values + 2] - probas_list[4][max_ch_values + 2]

                if max_ch_values < 10:
                    probas_list[4][max_ch_values + 3] = 3 * 2 * (sum([1 for val in deck_values if val == (max_ch_values + 1)]) / deck_n) * \
                                                       (sum([1 for val in deck_values if val == (max_ch_values + 2)]) / (deck_n - 1)) * \
                                                        (sum([1 for val in deck_values if val == (max_ch_values + 3)]) / (deck_n - 2))
                    probas_list[0][max_ch_values + 3] = probas_list[0][max_ch_values + 3] - probas_list[4][max_ch_values + 3]

            if -1 in ch_values:
                ch_values[ch_values == -1] = 12
                max_ch_values = max(ch_values)
                min_ch_values = min(ch_values)
                probas_list[0][12] = probas_list[0][12] - probas_list[4][3]
            elif max_ch_values < 4:
                deck_values[deck_values == -1] = 12
                probas_list[0][12] = probas_list[0][12] - probas_list[4][3]

            # flush odds
            if len(np.unique(ch_suits)) == 1:
                n_tot_cards = sum([1 for val in deck_suits if val == ch_suits[0]])
                over_cards = deck[(deck > max(current_hand)) & (deck < 13 * (ch_suits[0] + 1))]
                for i in range(len(over_cards)):
                    val = over_cards[i] % 13
                    probas_list[5][val] = 3 * (1 / deck_n) * \
                                            (n_tot_cards - i - 1) / (deck_n - 1) * \
                                            (n_tot_cards - i - 2) / (deck_n - 2)
                    probas_list[0][val] = probas_list[0][val] - probas_list[5][val]

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
                        probas_list[2][ch_val] += extra_pair_odds_list[extra_val] * \
                                                    (sum([1 for val in deck_values if val == ch_val])) / \
                                                    (sum([1 for x in deck_values if x not in ch_values and x != extra_val]))
                    else:
                        probas_list[2][extra_val] += extra_pair_odds_list[extra_val] * \
                                                    (sum([1 for val in deck_values if val == ch_val])) / \
                                                    (sum([1 for x in deck_values if x not in ch_values and x != extra_val]))

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

    def calc_heuristic_adj_probas(self, p_vec, cards_left):
        htd_adj = 1/3
        future_turn_chances = 3
        new_vec = np.array(p_vec).copy()
        if sum(new_vec > 0) == 1:
            return p_vec

        turn_chances = 4 - (self.decision_id % 4) * int(cards_left == (3 - self.decision_id//4))
        inds = np.flip(np.array(range(9*13))[new_vec > 0])
        hand_types_desirability = np.cumsum(np.insert(new_vec, 0, 0))
        hand_types_desirability = hand_types_desirability - htd_adj
        for i in inds:
            htd = hand_types_desirability[i]
            if htd < 1/max(turn_chances, future_turn_chances):
                break
            per_turn_proba = (new_vec[i] ** (1 / cards_left))
            curr_turn_proba = (1 - (1 - per_turn_proba) ** (turn_chances * htd)) if htd > 1/turn_chances else per_turn_proba
            remaining_turns_proba = (1 - (1 - per_turn_proba) ** (future_turn_chances * htd)) if htd > 1/future_turn_chances else per_turn_proba
            new_vec[i] = curr_turn_proba * (remaining_turns_proba**(cards_left - 1))
            new_vec[:i] *= (1 - sum(new_vec[i:])) / sum(new_vec[:i])

        return list(new_vec)

    def calc_hand_win_proba(self, cpu_probas, opp_probas, cpu_hand, opp_hand, deck):
        return self.calc_extended_hand_win_proba(cpu_probas, opp_probas, cpu_hand, opp_hand, deck)

    def calc_extended_hand_win_proba(self, cpu_probas, opp_probas, cpu_hand, opp_hand, deck):
        if self.fifth_card_exact_calc and (sum(cpu_hand == -1) == 0):
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

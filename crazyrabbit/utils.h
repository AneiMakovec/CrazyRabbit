/*
    CrazyRabbit 2.2, a program for playing the chess variant Crazyhouse
    with the use of deep learning and domain knowledge.

    Copyright (C) 2022 Anei Makovec

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CRAZYRABBIT_UTILS_HPP
#define CRAZYRABBIT_UTILS_HPP

#include <string>
#include <fstream>
#include <chrono>
#include <format>
#include "surge/types.h"
#include "dirichlet/dirichlet.h"
#include "robin_hood/robin_hood.h"

namespace crazyrabbit
{

    // ------------------------------ GAME RELATED ------------------------------

    constexpr auto ACTION_SIZE = 5184;
    constexpr auto REPETITIONS_NORM = 500.0f;
    constexpr auto POCKET_COUNT_NORM = 32.0f;
    constexpr auto HALFMOVES_NORM = 40.0f;

    // ------------------------------ EVAL RELATED ------------------------------

    typedef uint16_t AttackInfo;

    constexpr int attackers_index = 11;
    constexpr AttackInfo clear_attackers = 0b0000011111111111;
    constexpr AttackInfo attack_mask[NPIECE_TYPES] = { 0b0000010000000000,
                                                   0b0000001000000000,
                                                   0b0000000100000000,
                                                   0b0000000010000000,
                                                   0b0000000001000000,
                                                   0b0000000000100000 };
    constexpr AttackInfo attacks_mask = 0b0000011111100000;

    constexpr AttackInfo drop_mask[NPIECE_TYPES] = { 0b0000000000010000,
                                                   0b0000000000001000,
                                                   0b0000000000000100,
                                                   0b0000000000000010,
                                                   0b0000000000000001,
                                                   0b0000000000000000 };
    constexpr AttackInfo drops_mask = 0b0000000000011111;

    constexpr void add_attack_info(AttackInfo& info, PieceType piece)
    {
        AttackInfo num_attackers = info >> attackers_index;
        info = (info & clear_attackers) | (++num_attackers << attackers_index) | attack_mask[piece];
    }

    constexpr void add_drop_info(AttackInfo& info, PieceType piece)
    {
        info |= drop_mask[piece];
    }

    constexpr bool can_attack(AttackInfo& info, PieceType piece)
    {
        return info & attack_mask[piece];
    }

    constexpr bool can_drop(AttackInfo& info, PieceType piece)
    {
        return info & drop_mask[piece];
    }

    constexpr int attacks_num(AttackInfo& info)
    {
        return static_cast<int>(info >> attackers_index);
    }

    inline void print_attack_info(AttackInfo& info)
    {
        int num_attacks = info >> attackers_index;
        if (num_attacks)
        {
            std::cout << num_attacks << " attacks by ";
            for (int i = 0; i < NPIECE_TYPES; i++)
            {
                if (info & attack_mask[i])
                    std::cout << PIECE_STR[i];
            }
        } else
        {
            std::cout << "no attacks, ";
        }

        if (info & drops_mask)
        {
            std::cout << ", drops: ";
        } else
        {
            std::cout << ", no drops";
            return;
        }

        if (info & drop_mask[PAWN])
            std::cout << "P";
        if (info & drop_mask[PAWN])
            std::cout << "N";
        if (info & drop_mask[PAWN])
            std::cout << "B";
        if (info & drop_mask[PAWN])
            std::cout << "R";
        if (info & drop_mask[PAWN])
            std::cout << "Q";

        std::cout << ", 0b";

        for (int i = 15; i >= 0; i--)
        {
            std::cout << ((info & (1 << i)) ? 1 : 0);
        }
    }

    typedef uint8_t EvalMask;

    constexpr EvalMask material_mask = 0b00000001;
    constexpr EvalMask pawn_structure_mask = 0b00000010;
    constexpr EvalMask king_safety_mask = 0b00000100;
    constexpr EvalMask piece_placement_mask = 0b00001000;
    constexpr EvalMask board_control_mask = 0b00010000;

    typedef uint8_t PolicyMask;

    constexpr PolicyMask dropping_moves_mask = 0b00000001;
    constexpr PolicyMask checking_moves_mask = 0b00000010;
    constexpr PolicyMask forking_moves_mask = 0b00000100;
    constexpr PolicyMask capturing_moves_mask = 0b00001000;

    struct ModMask
    {
        EvalMask eval_mask = 0U;
        PolicyMask policy_mask = 0U;
        bool use_dirichlet = true;
    };

    std::ostream& operator<<(std::ostream& os, const ModMask& m)
    {
        if (!m.eval_mask && !m.policy_mask)
        {
            os << "default";
            return os;
        }

        bool started = false;

        if (m.eval_mask & material_mask)
        {
            os << "MT";
            started = true;
        }

        if (m.eval_mask & pawn_structure_mask)
        {
            if (started)
                os << "-";
            else
                started = true;

            os << "PS";
        }

        if (m.eval_mask & king_safety_mask)
        {
            if (started)
                os << "-";
            else
                started = true;

            os << "KS";
        }

        if (m.eval_mask & piece_placement_mask)
        {
            if (started)
                os << "-";
            else
                started = true;

            os << "PP";
        }

        if (m.eval_mask & board_control_mask)
        {
            if (started)
                os << "-";
            else
                started = true;

            os << "BC";
        }

        if (m.policy_mask & dropping_moves_mask)
        {
            if (started)
                os << "-";
            else
                started = true;

            os << "DM";
        }

        if (m.policy_mask & checking_moves_mask)
        {
            if (started)
                os << "-";
            else
                started = true;

            os << "CHM";
        }

        if (m.policy_mask & forking_moves_mask)
        {
            if (started)
                os << "-";
            else
                started = true;

            os << "FM";
        }

        if (m.policy_mask & capturing_moves_mask)
        {
            if (started)
                os << "-";
            else
                started = true;

            os << "CPM";
        }

        return os;
    }

    inline ModMask parse_mod_mask(std::string mask)
    {
        ModMask mod_mask;

        bool end = false;
        size_t split_point;
        std::string mod;
        while (!end)
        {
            split_point = mask.find('-');
            if (split_point == std::string::npos)
            {
                mod = mask;
                end = true;
            }
            else
            {
                mod = mask.substr(0, split_point);
                mask = mask.substr(split_point + 1);
            }

            if (mod == "MT")
            {
                mod_mask.eval_mask |= material_mask;
            }
            else if (mod == "PS")
            {
                mod_mask.eval_mask |= pawn_structure_mask;
            }
            else if (mod == "KS")
            {
                mod_mask.eval_mask |= king_safety_mask;
            }
            else if (mod == "PP")
            {
                mod_mask.eval_mask |= piece_placement_mask;
            }
            else if (mod == "BC")
            {
                mod_mask.eval_mask |= board_control_mask;
            }
            else if (mod == "CHM")
            {
                mod_mask.policy_mask |= checking_moves_mask;
            }
            else if (mod == "FM")
            {
                mod_mask.policy_mask |= forking_moves_mask;
            }
            else if (mod == "DM")
            {
                mod_mask.policy_mask |= dropping_moves_mask;
            }
            else if (mod == "CPM")
            {
                mod_mask.policy_mask |= capturing_moves_mask;
            }
        }
        return mod_mask;
    }

    // --------------------- material -----------------------
    const double material_value[NPIECE_TYPES] =
        //       P,    N,    B,    R,    Q,   -
    { 1.26, 2.54, 3.00, 3.02, 4.83, 0.0 };
    const double material_value_hand[NPIECE_TYPES] =
        //       P,    N,    B,    R,    Q,   -
    { 1.03, 2.48, 2.38, 2.96, 4.47, 0.0 };
    constexpr double bishop_pair_bonus = 0.2;
    constexpr double knight_queen_bonus = 0.12;
    constexpr double bishop_rook_bonus = 0.1;
    constexpr double knight_pawn_bonus = 0.048;

    // ------------------- pawn structure --------------------
    const double doubled_pawn_pen[8] =
        //       A,     B,     C,      D,      E,     F,     G,    H
    { -0.2, -0.16, -0.16, -0.256, -0.256, -0.16, -0.16, -0.2 };

    const double passed_pawn_hi_supp[6] =
        //       R2,    R3,    R4,    R5,    R6,    R7
    { 0.148, 0.252, 0.500, 0.900, 1.400, 2.000 };
    const double passed_pawn_hi_nsupp[6] =
    { 0.100, 0.200, 0.300, 0.500, 0.900, 1.500 };
    const double passed_pawn_lo_supp[6] =
    { 0.148, 0.164, 0.236, 0.372, 0.872, 1.296 };
    const double passed_pawn_lo_nsupp[6] =
    { 0.136, 0.168, 0.148, 0.112, 0.100, 0.064 };

    //                            [num. supporters][can advance][is file half-open]
    const double isolated_pawn_pen[2][2][2] = {
                                                {                         // 0 supporters
                                                    {                         // can advance
                                                        -0.120,                 // closed file
                                                        -0.260 },               // open file
                                                    {                         // stopped
                                                        -0.220,                 // closed file
                                                        -0.360 } },             // open file
                                                {                         // 1 supporter
                                                    {                         // can advance
                                                        -0.040,                 // closed file
                                                        -0.120 },               // open file
                                                    {                         // stopped
                                                        -0.140,                 // closed file
                                                        -0.220 } } };           // open file

    // ----------------------- king safety ----------------------------

    const double king_square_vuln_w[NSQUARES] = {
         1.0,  0.0,  1.0,  3.0,  3.0,  1.0,  0.0,  1.0,
         2.0,  2.0,  3.0,  4.0,  4.0,  3.0,  2.0,  2.0,
         5.0,  6.0,  6.0,  8.0,  8.0,  6.0,  6.0,  5.0,
        10.0, 12.0, 14.0, 14.0, 14.0, 14.0, 12.0, 10.0,
        18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0,
        24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0,
        28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0,
        32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0
    };

    const double king_square_vuln_b[NSQUARES] = {
        32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0,
        28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0,
        24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0,
        18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0,
        10.0, 12.0, 14.0, 14.0, 14.0, 14.0, 12.0, 10.0,
         5.0,  6.0,  6.0,  8.0,  8.0,  6.0,  6.0,  5.0,
         2.0,  2.0,  3.0,  4.0,  4.0,  3.0,  2.0,  2.0,
         1.0,  0.0,  1.0,  3.0,  3.0,  1.0,  0.0,  1.0
    };

    // white side: [black][white], black side: [white][black]
    const double king_struct_vuln[4][4] = {
        {8.0, 1.0, 3.0, 6.0},
        {6.0, 0.0, 2.0, 3.0},
        {7.0, 0.0, 2.0, 3.0},
        {8.0, 1.0, 2.0, 4.0}
    };

    constexpr double empty_square_pen = 5.0;
    constexpr double check_pen = 200.0;
    constexpr double full_castling_bonus = 0.144;
    constexpr double ks_castling_bonus = 0.1;
    constexpr double qs_castling_bonus = 0.072;

    // --------------------- piece placement --------------------------

    enum KingZone
    {
        KZONE0, KZONE1, KZONE2, KZONE3, KZONE4, KZONE5, KZONE6, KZONE7
    };

    const int king_zone_w[NSQUARES] = {
        0, 0, 0, 1, 1, 2, 2, 2,
        0, 3, 3, 1, 1, 4, 4, 2,
        3, 3, 3, 3, 4, 4, 4, 4,
        5, 5, 5, 5, 6, 6, 6, 6,
        5, 5, 5, 5, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7
    };

    const int king_zone_b[NSQUARES] = {
        7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7,
        5, 5, 5, 5, 6, 6, 6, 6,
        5, 5, 5, 5, 6, 6, 6, 6,
        3, 3, 3, 3, 4, 4, 4, 4,
        0, 3, 3, 1, 1, 4, 4, 2,
        0, 0, 0, 1, 1, 2, 2, 2
    };

    const double knight_distance_bonus[8] = { 0.064, 0.048, 0.032, 0.016, 0.000, -0.016, -0.032, -0.048 };
    const double rook_distance_bonus[8] = { 0.032, 0.024, 0.016, 0.008, 0.000, -0.008, -0.016, -0.024 };
    const double queen_distance_bonus[8] = { 0.048, 0.036, 0.024, 0.012, 0.000, -0.012, -0.024, -0.036 };

    int diamond_distance_w[8][NSQUARES];
    int diamond_distance_b[8][NSQUARES];
    int cross_distance_w[8][NSQUARES];
    int cross_distance_b[8][NSQUARES];

    inline void init_diamond_distances()
    {
        for (int zone = 0; zone < 8; zone++)
        {
            for (int square = 0; square < 64; square++)
            {
                int s_x = file_of(static_cast<Square>(square));
                int s_y = rank_of(static_cast<Square>(square));

                int num_sq_zone_w = 0, num_sq_zone_b = 0;
                int d_sum_w = 0, d_sum_b = 0;

                for (int s = 0; s < 64; s++)
                {
                    int z_x = file_of(static_cast<Square>(s));
                    int z_y = rank_of(static_cast<Square>(s));

                    // white side
                    if (king_zone_w[s] == zone)
                    {
                        num_sq_zone_w++;
                        d_sum_w += 3 * abs(s_x - z_x) + 2 * abs(s_y - z_y);
                    }

                    // black side
                    if (king_zone_b[s] == zone)
                    {
                        num_sq_zone_b++;
                        d_sum_b += 3 * abs(s_x - z_x) + 2 * abs(s_y - z_y);
                    }
                }

                diamond_distance_w[zone][square] = fmin(7, fmax(0, d_sum_w / (3 * num_sq_zone_w) - 1));
                diamond_distance_b[zone][square] = fmin(7, fmax(0, d_sum_b / (3 * num_sq_zone_b) - 1));
            }
        }
    }

    inline void init_cross_distances()
    {
        for (int zone = 0; zone < 8; zone++)
        {
            for (int square = 0; square < 64; square++)
            {
                int s_x = file_of(static_cast<Square>(square));
                int s_y = rank_of(static_cast<Square>(square));

                int num_sq_zone_w = 0, num_sq_zone_b = 0;
                int d_sum_w = 0, d_sum_b = 0;

                for (int s = 0; s < 64; s++)
                {
                    int z_x = file_of(static_cast<Square>(s));
                    int z_y = rank_of(static_cast<Square>(s));

                    // white side
                    if (king_zone_w[s] == zone)
                    {
                        num_sq_zone_w++;
                        d_sum_w += fmin(4 * abs(s_x - z_x), 3 * abs(s_y - z_y));
                    }

                    // black side
                    if (king_zone_b[s] == zone)
                    {
                        num_sq_zone_b++;
                        d_sum_b += fmin(4 * abs(s_x - z_x), 3 * abs(s_y - z_y));
                    }
                }

                cross_distance_w[zone][square] = fmin(7, fmax(0, d_sum_w / (2 * num_sq_zone_w)));
                cross_distance_b[zone][square] = fmin(7, fmax(0, d_sum_b / (2 * num_sq_zone_b)));
            }
        }
    }

    // pawn
    const double pawn_square_score_w[NSQUARES] = {
         0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
         0.04,  0.06,  0.02, -0.04, -0.06,  0.04,  0.08,  0.04,
        -0.02, -0.01,  0.00,  0.00,  0.00,  0.00, -0.01, -0.02,
        -0.02, -0.01,  0.00,  0.02,  0.02,  0.00, -0.01, -0.02,
         0.00,  0.00,  0.00,  0.03,  0.03,  0.00,  0.00,  0.00,
         0.00,  0.00,  0.00,  0.03,  0.03,  0.00,  0.00,  0.00,
         0.05,  0.06,  0.05,  0.05,  0.05,  0.05,  0.06,  0.05,
         0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00
    };

    const double pawn_square_score_b[NSQUARES] = {
         0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
         0.05,  0.06,  0.05,  0.05,  0.05,  0.05,  0.06,  0.05,
         0.00,  0.00,  0.00,  0.03,  0.03,  0.00,  0.00,  0.00,
         0.00,  0.00,  0.00,  0.03,  0.03,  0.00,  0.00,  0.00,
        -0.02, -0.01,  0.00,  0.02,  0.02,  0.00, -0.01, -0.02,
        -0.02, -0.01,  0.00,  0.00,  0.00,  0.00, -0.01, -0.02,
         0.04,  0.06,  0.02, -0.04, -0.06,  0.04,  0.08,  0.04,
         0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00
    };

    // knight
    const double knight_square_score_w[NSQUARES] = {
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0
    };

    const double knight_square_score_b[NSQUARES] = {
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0
    };

    constexpr double strong_sq_bonus = 0.08;
    constexpr double strong_cent_sq_bonus = 0.14;

    // bishop
    const double bishop_square_score_w[NSQUARES] = {
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0
    };

    const double bishop_square_score_b[NSQUARES] = {
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0
    };

    constexpr double bishop_diag_penalty = -0.06;
    constexpr double bishop_diag_bonus = 0.08;

    // rook
    const double rook_square_score_w[NSQUARES] = {
        0.01,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.01,
        0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
        0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
        0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
        0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
        0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
        0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
        0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05
    };

    const double rook_square_score_b[NSQUARES] = {
        0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05,  0.05,
        0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
        0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
        0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
        0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
        0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
        0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
        0.01,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.01
    };

    const double rook_open_file_bonus = 0.1;
    const double rook_half_file_bonus = 0.06;
    const double rook_weak_pawn_bonus = 0.044;

    // queen
    const double queen_square_score_w[NSQUARES] = {
        -0.16,-0.14,-0.10,-0.04,-0.08,-0.10,-0.14,-0.16,
        -0.12,-0.12,-0.10,-0.08,-0.08,-0.10,-0.12,-0.12,
        -0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,
        -0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,
        -0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,
        -0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,
        -0.10,-0.10,-0.10,-0.10,-0.10,-0.10,-0.10,-0.10,
        -0.10,-0.10,-0.10,-0.10,-0.10,-0.10,-0.10,-0.10
    };

    const double queen_square_score_b[NSQUARES] = {
        -0.10,-0.10,-0.10,-0.10,-0.10,-0.10,-0.10,-0.10,
        -0.10,-0.10,-0.10,-0.10,-0.10,-0.10,-0.10,-0.10,
        -0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,
        -0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,
        -0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,
        -0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,-0.20,
        -0.12,-0.12,-0.10,-0.08,-0.08,-0.10,-0.12,-0.12,
        -0.16,-0.14,-0.10,-0.04,-0.08,-0.10,-0.14,-0.16
    };

    // king
    const double king_square_score_w[NSQUARES] = {
         0.01, 0.02, 0.01, 0.00, 0.01, 0.00, 0.02, 0.01,
        -0.02,-0.06,-0.11,-0.11,-0.11,-0.11,-0.06,-0.02,
        -0.10,-0.18,-0.25,-0.25,-0.25,-0.25,-0.25,-0.10,
        -0.18,-0.25,-0.35,-0.35,-0.35,-0.35,-0.25,-0.18,
        -0.25,-0.35,-0.35,-0.35,-0.35,-0.35,-0.35,-0.25,
        -0.25,-0.35,-0.35,-0.35,-0.35,-0.35,-0.35,-0.25,
        -0.18,-0.25,-0.25,-0.25,-0.25,-0.25,-0.25,-0.18,
        -0.10,-0.18,-0.25,-0.25,-0.25,-0.25,-0.18,-0.10,
    };

    const double king_square_score_b[NSQUARES] = {
        -0.10,-0.18,-0.25,-0.25,-0.25,-0.25,-0.18,-0.10,
        -0.18,-0.25,-0.25,-0.25,-0.25,-0.25,-0.25,-0.18,
        -0.25,-0.35,-0.35,-0.35,-0.35,-0.35,-0.35,-0.25,
        -0.25,-0.35,-0.35,-0.35,-0.35,-0.35,-0.35,-0.25,
        -0.18,-0.25,-0.35,-0.35,-0.35,-0.35,-0.25,-0.18,
        -0.10,-0.18,-0.25,-0.25,-0.25,-0.25,-0.25,-0.10,
        -0.02,-0.06,-0.11,-0.11,-0.11,-0.11,-0.06,-0.02,
         0.01, 0.02, 0.01, 0.00, 0.01, 0.00, 0.02, 0.01
    };

    // ---------------------- board control ---------------------------

    const double control_bonus_w[NSQUARES] = {
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0,
        1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0,
        2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0,
        2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0
    };

    const double control_bonus_b[NSQUARES] = {
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0,
        2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0,
        1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0,
        1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    };

    // ------------------------ smart drops ---------------------------

    const double drop_pawn_location_w[NSQUARES] = {
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
        0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
        0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    };

    const double drop_pawn_location_b[NSQUARES] = {
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
        0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
        0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    };

    constexpr double drop_king_def_bonus = 0.25;

    constexpr double drop_knight_attack_king_bonus = 0.10;
    constexpr double drop_knight_rank_bonus = 0.15;

    constexpr double drop_rook_rank_bonus = 0.2;

    // --------------------- initialisation functions --------------------------

    inline void initialise_eval_tables()
    {
        init_diamond_distances();
        init_cross_distances();
    }

    // ------------------------- MATE SEARCH RELATED ----------------------------

    constexpr int default_max_depth = 3;

    // ---------------------------- MCTS RELATED --------------------------------

    constexpr int cpuct = 1;
    constexpr float cpuct_init = 2.5f;
    constexpr long cpuct_base = 19652;
    constexpr double dirichlet_alpha = 0.2;
    constexpr double dirichlet_factor = 0.25;
    constexpr double u_min = 0.25;
    constexpr double u_init = 1.0;
    constexpr int u_base = 1965;
    constexpr double Q_init = 0.0;
    constexpr double Q_thresh_init = 0.5;
    constexpr double Q_thresh_max = 0.9;
    constexpr int Q_thresh_base = 1965;
    constexpr double Q_factor = 0.7;
    constexpr double check_thresh = 0.1;
    constexpr double check_factor = 0.5;
    constexpr double EPS = 1e-8;
    constexpr int moves_per_game = 50;
    constexpr int move_thresh = 40;
    constexpr double original_time_amount = 0.7;
    constexpr double increment_amount = 0.7;
    constexpr double time_proportion = 0.2; //0.05;
    constexpr double eval_factor = 0.25;

    typedef robin_hood::unordered_map<std::string, move_vector<Move>> MD_t;

    enum class BestMoveStrat
    {
        Default,
        Q_value,
        NUM
    };

    enum class PolicyEnhancementStrat
    {
        Dirichlet,
        DroppingMoves,
        CheckingMoves,
        CapturingMoves,
        ForkingMoves,
        NUM
    };

    enum class NodeExpansionStrat
    {
        Default,
        Exploration,
        NUM
    };

    enum class BackpropStrat
    {
        Default,
        SMA,
        NUM
    };

    struct MCTS_config
    {
        int num_sims;
        BestMoveStrat best_move_strategy;
        NodeExpansionStrat node_expansion_strategy;
        BackpropStrat backprop_strategy;
        bool use_dirichlet;

        bool config_switch;
        ModMask config;
        ModMask config_ts;

        MCTS_config() : num_sims(100), best_move_strategy(BestMoveStrat::Default), node_expansion_strategy(NodeExpansionStrat::Default), backprop_strategy(BackpropStrat::Default),
                        use_dirichlet(true), config_switch(false), config(), config_ts() {}

        ~MCTS_config() = default;
    };


    // ---------------------------- UTILITY CLASSES --------------------------------
    class Dirichlet
    {
    private:
        dirichlet_distribution<std::mt19937>* d;
        std::mt19937 gen;

        Dirichlet()
        {
            std::random_device rd;
            gen = std::mt19937(rd());
            std::vector<double> alpha(ACTION_SIZE, dirichlet_alpha);
            d = new dirichlet_distribution<std::mt19937>(alpha);
        }

        ~Dirichlet() { delete d; }

    public:
        static Dirichlet& get_instance()
        {
            static Dirichlet instance;
            return instance;
        }

        Dirichlet(const Dirichlet&) = delete;
        void operator=(const Dirichlet&) = delete;

        //Returns a Dirichlet noise sample.
        inline std::vector<double> get_noise() { return (d->operator())(gen); }
    };

    class Elo
    {
    private:
        int m_wins;
        int m_losses;
        int m_draws;
        double m_mu;
        double m_stdev;

    public:
        Elo(int wins, int losses, int draws)
        {
            m_wins = wins;
            m_losses = losses;
            m_draws = draws;

            double n = wins + losses + draws;
            double w = wins / n;
            double l = losses / n;
            double d = draws / n;
            m_mu = w + d / 2.0;

            double devW = w * std::pow(1.0 - m_mu, 2.0);
            double devL = l * std::pow(0.0 - m_mu, 2.0);
            double devD = d * std::pow(0.5 - m_mu, 2.0);
            m_stdev = std::sqrt(devW + devL + devD) / std::sqrt(n);
        }

        inline double point_ratio()
        {
            double total = (m_wins + m_losses + m_draws) * 2;
            return ((m_wins * 2) + m_draws) / total;
        }

        inline double draw_ratio()
        {
            double n = m_wins + m_losses + m_draws;
            return m_draws / n;
        }

        inline double diff(double p) { return -400.0 * std::log10(1.0 / p - 1.0); }

        inline double diff() { return diff(m_mu); }

        inline double erf_inv(double x)
        {
            const double pi = 3.1415926535897;

            double a = 8.0 * (pi - 3.0) / (3.0 * pi * (4.0 - pi));
            double y = std::log(1.0 - x * x);
            double z = 2.0 / (pi * a) + y / 2.0;

            double ret = std::sqrt(std::sqrt(z * z - y / a) - z);

            if (x < 0.0)
                return -ret;
            return ret;
        }

        inline double phi_inv(double p) { return std::sqrt(2.0) * erf_inv(2.0 * p - 1.0); }

        inline double error_margin()
        {
            double muMin = m_mu + phi_inv(0.025) * m_stdev;
            double muMax = m_mu + phi_inv(0.975) * m_stdev;
            return (diff(muMax) - diff(muMin)) / 2.0;
        }

        inline double LOS() const { return 100 * (0.5 + 0.5 * std::erf((static_cast<double>(m_wins) - static_cast<double>(m_losses)) / std::sqrt(2.0 * (static_cast<double>(m_wins) + static_cast<double>(m_losses))))); }
    };

    class PGN_writer
    {
    public:
        std::string event_name;
        std::string date;
        int round;
        std::string white;
        std::string black;

        std::ofstream pgn_file;

        std::vector<std::string> moves;

        PGN_writer() = delete;

        PGN_writer(const std::string file_path) { pgn_file = std::ofstream(file_path, std::ios_base::out | std::ios_base::app); }

        inline void new_game(const std::string event_n, const int r, const std::string white_name, const std::string black_name)
        {
            event_name = event_n;
            auto const time = std::chrono::current_zone()
                ->to_local(std::chrono::system_clock::now());
            date = std::format("{:%Y-%m-%d}", time);
            round = r;
            white = white_name;
            black = black_name;
            moves.clear();
        }

        inline void add_move(const std::string move) { moves.emplace_back(move); }

        inline void flush(const Color winner)
        {
            pgn_file << "[Event \"" << event_name << "\"]\n";
            pgn_file << "[Site \"Ljubljana, Slovenia\"]\n";
            pgn_file << "[Date \"" << date << "\"]\n";
            pgn_file << "[Round \"" << round << "\"]\n";
            pgn_file << "[White \"" << white << "\"]\n";
            pgn_file << "[Black \"" << black << "\"]\n";
            std::string result;
            if (winner == WHITE)
                result = "1-0";
            else if (winner == BLACK)
                result = "0-1";
            else
                result = "1/2-1/2";
            pgn_file << "[Result \"" << result << "\"]\n";
            pgn_file << "[Variant \"crazyhouse\"]\n\n";

            int num_moves = moves.size();
            int move_index = 0;
            Color player = WHITE;
            int move_counter = 0;
            for (const std::string move : moves)
            {
                move_index++;

                if (player == WHITE)
                {
                    move_counter++;
                    pgn_file << move_counter << ". ";
                }

                pgn_file << move;

                if (move_index == num_moves)
                    pgn_file << "# ";
                else
                    pgn_file << " ";

                if (move_index == num_moves)
                    pgn_file << result << "\n\n";
                else if (player == BLACK && move_counter % 2 == 0)
                    pgn_file << "\n";

                player = ~player;
            }
            pgn_file.flush();
        }

        inline void close() { pgn_file.close(); }
    };

    class PGN_reader
    {
    public:
        std::string event_name;
        std::string site;
        std::string date;
        int round;
        std::string white;
        std::string black;
        Color result;
        std::string variant;

        std::ifstream pgn_file;

        PGN_reader() = delete;

        PGN_reader(const std::string file_path) { pgn_file = std::ifstream(file_path); }

        inline bool read_game()
        {
            if (pgn_file.is_open() && pgn_file)
            {
                bool reading_tags = true;

                std::string line;
                while (line.empty())
                {
                    if (!pgn_file)
                        return false;

                    std::getline(pgn_file, line);
                }

                while (true)
                {
                    if (!pgn_file)
                        return true;

                    if (reading_tags)
                    {
                        if (line.empty())
                        {
                            std::getline(pgn_file, line);
                            reading_tags = false;
                            continue;
                        }

                        line = line.substr(1, line.find(']'));
                        size_t split_point = line.find(' ');
                        std::string tag = line.substr(0, split_point);
                        std::string value = line.substr(split_point + 1);
                        value = value.substr(1);
                        value = value.substr(0, value.find('"'));

                        if (tag == "Event")
                        {
                            event_name = value;
                        }
                        else if (tag == "Site")
                        {
                            site = value;
                        }
                        else if (tag == "Date")
                        {
                            date = value;
                        }
                        else if (tag == "Round")
                        {
                            round = std::stoi(value);
                        }
                        else if (tag == "White")
                        {
                            white = value;
                        }
                        else if (tag == "Black")
                        {
                            black = value;
                        }
                        else if (tag == "Result")
                        {
                            if (value == "1-0")
                                result = WHITE;
                            else if (value == "0-1")
                                result = BLACK;
                            else
                                result = NO_COLOR;
                        }
                        else if (tag == "Variant")
                        {
                            variant = value;
                        }
                    }
                    else
                    {
                        if (line.empty())
                            return true;
                        //TODO: read moves
                    }

                    std::getline(pgn_file, line);
                }
            }

            return false;
        }
    };
}

#endif

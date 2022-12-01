#ifndef CRAZYZERO_UTILS_HPP
#define CRAZYZERO_UTILS_HPP

#include <stdlib.h>
#include <string>
#include <set>
#include <vector>
#include <unordered_map>
#include <map>
#include <robin_hood.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include "surge/types.h"
#include "dirichlet/dirichlet.h"

namespace crazyzero
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
    const AttackInfo attack_mask[NPIECE_TYPES] = { 0b0000010000000000,
                                                   0b0000001000000000,
                                                   0b0000000100000000,
                                                   0b0000000010000000,
                                                   0b0000000001000000,
                                                   0b0000000000100000 };
    constexpr AttackInfo attacks_mask = 0b0000011111100000;

    const AttackInfo drop_mask[NPIECE_TYPES] = { 0b0000000000010000,
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

    constexpr void print_attack_info(AttackInfo& info)
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

    typedef uint16_t TestMask;

    constexpr TestMask checking_moves_mask = 0b0000000000100000;
    constexpr TestMask forking_moves_mask = 0b0000000001000000;
    constexpr TestMask dropping_moves_mask = 0b0000000010000000;
    constexpr TestMask capturing_moves_mask = 0b0000000100000000;

    // --------------------- material -----------------------
    //const double material_value[NPIECE_TYPES] =
    ////      P,   N,   B,   R,   Q,   -
    //    { 2.0, 3.5, 3.0, 4.0, 6.0, 0.0 };
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
                                                        {-0.120},                 // closed file
                                                        {-0.260} },               // open file
                                                    {                         // stopped
                                                        {-0.220},                 // closed file
                                                        {-0.360} } },             // open file
                                                {                         // 1 supporter
                                                    {                         // can advance
                                                        {-0.040},                 // closed file
                                                        {-0.120} },               // open file
                                                    {                         // stopped
                                                        {-0.140},                 // closed file
                                                        {-0.220} } } };           // open file

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

    constexpr void init_diamond_distances()
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

    constexpr void init_cross_distances()
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
    /*const double pawn_square_score_w[NSQUARES] = {
        1.07, 1.07, 1.07, 1.07, 1.07, 1.07, 1.07, 1.07,
        0.66, 0.99, 1.21, 0.93, 1.16, 2.12, 1.93, 1.11,
        0.98, 0.92, 0.94, 1.14, 1.29, 1.31, 1.94, 1.34,
        0.73, 0.81, 0.77, 0.85, 1.11, 1.04, 1.18, 0.75,
        0.78, 0.85, 1.05, 1.24, 1.26, 1.51, 1.50, 1.04,
        0.85, 1.03, 1.26, 1.47, 1.61, 2.13, 1.82, 1.77,
        0.98, 1.17, 1.32, 1.36, 1.54, 1.48, 2.14, 1.28,
        1.07, 1.07, 1.07, 1.07, 1.07, 1.07, 1.07, 1.07
    };

    const double pawn_square_score_b[NSQUARES] = {
        1.07, 1.07, 1.07, 1.07, 1.07, 1.07, 1.07, 1.07,
        0.98, 1.17, 1.32, 1.36, 1.54, 1.48, 2.14, 1.28,
        0.85, 1.03, 1.26, 1.47, 1.61, 2.13, 1.82, 1.77,
        0.78, 0.85, 1.05, 1.24, 1.26, 1.51, 1.50, 1.04,
        0.73, 0.81, 0.77, 0.85, 1.11, 1.04, 1.18, 0.75,
        0.98, 0.92, 0.94, 1.14, 1.29, 1.31, 1.94, 1.34,
        0.66, 0.99, 1.21, 0.93, 1.16, 2.12, 1.93, 1.11,
        1.07, 1.07, 1.07, 1.07, 1.07, 1.07, 1.07, 1.07
    };*/

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
    /*const double knight_square_score_w[NSQUARES] = {
        1.93, 1.53, 1.99, 2.30, 2.30, 2.23, 1.91, 2.20,
        1.95, 2.01, 2.26, 2.41, 2.35, 2.60, 2.27, 2.28,
        1.91, 2.25, 2.47, 2.58, 2.78, 2.94, 2.75, 2.48,
        2.06, 2.31, 2.56, 2.71, 2.89, 3.02, 2.84, 2.56,
        2.28, 2.44, 2.64, 3.20, 3.65, 3.66, 3.33, 3.43,
        2.18, 2.53, 2.97, 2.99, 3.37, 3.97, 3.45, 2.95,
        2.09, 2.22, 2.42, 2.54, 2.67, 3.36, 2.32, 2.46,
        2.04, 2.31, 2.20, 2.56, 2.43, 2.51, 2.37, 2.42
    };

    const double knight_square_score_b[NSQUARES] = {
        2.04, 2.31, 2.20, 2.56, 2.43, 2.51, 2.37, 2.42,
        2.09, 2.22, 2.42, 2.54, 2.67, 3.36, 2.32, 2.46,
        2.18, 2.53, 2.97, 2.99, 3.37, 3.97, 3.45, 2.95,
        2.28, 2.44, 2.64, 3.20, 3.65, 3.66, 3.33, 3.43,
        2.06, 2.31, 2.56, 2.71, 2.89, 3.02, 2.84, 2.56,
        1.91, 2.25, 2.47, 2.58, 2.78, 2.94, 2.75, 2.48,
        1.95, 2.01, 2.26, 2.41, 2.35, 2.60, 2.27, 2.28,
        1.93, 1.53, 1.99, 2.30, 2.30, 2.23, 1.91, 2.20
    };*/

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
    /*const double bishop_square_score_w[NSQUARES] = {
        2.64, 2.38, 2.47, 2.92, 2.75, 2.77, 2.49, 2.61,
        2.78, 3.08, 2.86, 3.09, 3.21, 3.27, 3.87, 2.81,
        2.84, 2.94, 3.15, 3.22, 3.22, 3.39, 3.20, 3.36,
        2.58, 2.81, 2.91, 3.12, 3.29, 3.09, 3.05, 3.06,
        2.69, 2.83, 3.01, 3.20, 3.40, 3.09, 3.01, 3.02,
        2.44, 2.72, 3.26, 3.17, 3.46, 3.86, 3.31, 3.28,
        2.64, 2.93, 2.98, 2.77, 3.02, 3.19, 3.87, 2.79,
        2.74, 2.65, 2.70, 2.78, 2.78, 2.89, 2.89, 3.03
    };

    const double bishop_square_score_b[NSQUARES] = {
        2.74, 2.65, 2.70, 2.78, 2.78, 2.89, 2.89, 3.03,
        2.64, 2.93, 2.98, 2.77, 3.02, 3.19, 3.87, 2.79,
        2.44, 2.72, 3.26, 3.17, 3.46, 3.86, 3.31, 3.28,
        2.69, 2.83, 3.01, 3.20, 3.40, 3.09, 3.01, 3.02,
        2.58, 2.81, 2.91, 3.12, 3.29, 3.09, 3.05, 3.06,
        2.84, 2.94, 3.15, 3.22, 3.22, 3.39, 3.20, 3.36,
        2.78, 3.08, 2.86, 3.09, 3.21, 3.27, 3.87, 2.81,
        2.64, 2.38, 2.47, 2.92, 2.75, 2.77, 2.49, 2.61
    };*/

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
    /*const double rook_square_score_w[NSQUARES] = {
        2.68, 2.82, 2.87, 3.16, 3.08, 3.39, 2.96, 3.19,
        2.30, 2.74, 3.01, 2.99, 3.15, 3.54, 3.23, 2.99,
        2.67, 2.82, 2.79, 2.97, 2.97, 2.98, 3.10, 2.94,
        2.75, 2.76, 2.78, 2.86, 2.84, 2.98, 2.93, 2.95,
        2.72, 2.80, 2.90, 2.82, 2.91, 2.98, 2.98, 3.03,
        2.79, 2.81, 2.87, 2.93, 2.97, 2.91, 3.04, 3.07,
        2.90, 2.95, 3.28, 3.05, 3.26, 3.80, 3.80, 4.27,
        3.27, 3.52, 3.22, 3.64, 4.14, 3.99, 4.80, 4.75
    };

    const double rook_square_score_b[NSQUARES] = {
        3.27, 3.52, 3.22, 3.64, 4.14, 3.99, 4.80, 4.75,
        2.90, 2.95, 3.28, 3.05, 3.26, 3.80, 3.80, 4.27,
        2.79, 2.81, 2.87, 2.93, 2.97, 2.91, 3.04, 3.07,
        2.72, 2.80, 2.90, 2.82, 2.91, 2.98, 2.98, 3.03,
        2.75, 2.76, 2.78, 2.86, 2.84, 2.98, 2.93, 2.95,
        2.67, 2.82, 2.79, 2.97, 2.97, 2.98, 3.10, 2.94,
        2.30, 2.74, 3.01, 2.99, 3.15, 3.54, 3.23, 2.99,
        2.68, 2.82, 2.87, 3.16, 3.08, 3.39, 2.96, 3.19
    };*/

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
    /*const double queen_square_score_w[NSQUARES] = {
        4.55, 4.38, 4.63, 4.75, 4.62, 4.75, 4.64, 4.57,
        4.47, 4.54, 4.87, 4.85, 4.89, 5.00, 4.92, 4.63,
        4.50, 4.60, 4.72, 4.77, 4.74, 4.83, 4.89, 4.77,
        4.55, 4.60, 4.68, 4.79, 4.87, 4.82, 4.80, 4.70,
        4.57, 4.60, 4.75, 4.79, 5.00, 5.02, 4.95, 4.96,
        4.46, 4.77, 4.97, 4.99, 5.66, 5.60, 5.68, 5.55,
        4.62, 5.01, 5.42, 5.53, 5.56, 6.80, 6.91, 5.86,
        4.94, 5.09, 5.54, 5.74, 6.22, 6.42, 6.07, 5.99
    };

    const double queen_square_score_b[NSQUARES] = {
        4.94, 5.09, 5.54, 5.74, 6.22, 6.42, 6.07, 5.99,
        4.62, 5.01, 5.42, 5.53, 5.56, 6.80, 6.91, 5.86,
        4.46, 4.77, 4.97, 4.99, 5.66, 5.60, 5.68, 5.55,
        4.57, 4.60, 4.75, 4.79, 5.00, 5.02, 4.95, 4.96,
        4.55, 4.60, 4.68, 4.79, 4.87, 4.82, 4.80, 4.70,
        4.50, 4.60, 4.72, 4.77, 4.74, 4.83, 4.89, 4.77,
        4.47, 4.54, 4.87, 4.85, 4.89, 5.00, 4.92, 4.63,
        4.55, 4.38, 4.63, 4.75, 4.62, 4.75, 4.64, 4.57
    };*/

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
    /*const double king_square_score_w[NSQUARES] = {
        1.47, 1.61, 1.14, 0.11, 2.14, 0.95, 2.90, 4.25,
        2.53, 1.57, 1.20, 0.75, 0.46, 1.30, 1.63, 3.29,
        1.04, 0.93, 0.57, -0.29, -0.74, -0.97, 0.32, 0.30,
        0.73, 0.77, 0.48, 0.16, -0.89, -0.77, -0.54, -0.68,
        0.87, 0.91, 0.87, 0.77, 0.63, 0.71, 0.57, 0.44,
        0.97, 1.04, 1.00, 0.93, 0.93, 0.93, 0.93, 0.97,
        1.08, 1.00, 1.02, 0.97, 0.97, 1.00, 1.04, 1.04,
        1.02, 1.04, 1.00, 1.00, 1.00, 0.97, 1.02, 1.06
    };

    const double king_square_score_b[NSQUARES] = {
        1.02, 1.04, 1.00, 1.00, 1.00, 0.97, 1.02, 1.06,
        1.08, 1.00, 1.02, 0.97, 0.97, 1.00, 1.04, 1.04,
        0.97, 1.04, 1.00, 0.93, 0.93, 0.93, 0.93, 0.97,
        0.87, 0.91, 0.87, 0.77, 0.63, 0.71, 0.57, 0.44,
        0.73, 0.77, 0.48, 0.16, -0.89, -0.77, -0.54, -0.68,
        1.04, 0.93, 0.57, -0.29, -0.74, -0.97, 0.32, 0.30,
        2.53, 1.57, 1.20, 0.75, 0.46, 1.30, 1.63, 3.29,
        1.47, 1.61, 1.14, 0.11, 2.14, 0.95, 2.90, 4.25
    };*/

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

    /*
    constexpr int max_visits = 10000;
    double SQRT_NS[max_visits];
    double SQRT_NS_NSA[max_visits][max_visits];

    constexpr void init_sqrt()
    {
        for (int ns = 0; ns < max_visits; ns++)
        {
            SQRT_NS[ns] = sqrt(static_cast<double>(ns));
            for (int nsa = 0; nsa < max_visits; nsa++)
            {
                SQRT_NS_NSA[ns][nsa] = SQRT_NS[ns] / (1.0 + static_cast<double>(nsa));
            }
        }
    }
    */

    constexpr void initialise_eval_tables()
    {
        init_diamond_distances();
        init_cross_distances();
        //init_sqrt();
    }

    // ------------------------- MATE SEARCH RELATED ----------------------------

    constexpr int max_depth = 3;

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
    constexpr int moves_per_game = 10;
    constexpr int move_thresh = 20;
    constexpr double original_time_amount = 0.7;
    constexpr double increment_amount = 0.7;
    constexpr double time_proportion = 0.05;
    constexpr double eval_factor = 0.25;

    typedef robin_hood::unordered_map<long long, move_vector<Move>> MD_t;

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

        bool changed;

        bool time_control;
        bool variable_time_control;
        int num_sims;
        BestMoveStrat best_move_strategy;
        NodeExpansionStrat node_expansion_strategy;
        BackpropStrat backprop_strategy;

        EvalMask eval_types;
        std::set<PolicyEnhancementStrat> policy_strategies;

        bool time_saving_config;
        EvalMask ts_eval_types;
        std::set<PolicyEnhancementStrat> ts_policy_strategies;

        MCTS_config() : changed(false), time_control(true), variable_time_control(false), num_sims(100),
            best_move_strategy(BestMoveStrat::Default), node_expansion_strategy(NodeExpansionStrat::Default), backprop_strategy(BackpropStrat::Default),
            eval_types(0), time_saving_config(false), ts_eval_types(0)
        {}

        ~MCTS_config() = default;
    };


    // ---------------------------- UTILITY CLASSES --------------------------------
    class Dirichlet
    {
    private:
        dirichlet_distribution<std::mt19937>* d;
        std::mt19937 gen;

    public:
        Dirichlet()
        {
            std::random_device rd;
            gen = std::mt19937(rd());
            std::vector<double> alpha(ACTION_SIZE, dirichlet_alpha);
            d = new dirichlet_distribution<std::mt19937>(alpha);
        }

        ~Dirichlet() { delete d; }

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

    class PGN
    {
    public:
        std::string event_name;
        std::string date;
        int round;
        std::string white;
        std::string black;

        std::ofstream pgn_file;

        std::vector<std::string> moves;

        PGN(const std::string file_name) { pgn_file = std::ofstream(file_name, std::ios_base::out | std::ios_base::app); }

        inline void new_game(const std::string event_n, const int r, const std::string d, const std::string white_name, const std::string black_name)
        {
            event_name = event_n;
            date = d;
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
}

#endif

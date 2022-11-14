#pragma once

#include <string>
#include <set>
#include <unordered_map>
#include "surge/types.h"

namespace crazyzero {

    // ------------------------------ GAME RELATED ------------------------------

    constexpr auto ACTION_SIZE = 5184;
    constexpr auto REPETITIONS_NORM = 500.0f;
    constexpr auto POCKET_COUNT_NORM = 32.0f;
    constexpr auto HALFMOVES_NORM = 40.0f;

    // ------------------------------ EVAL RELATED ------------------------------

    typedef uint16_t AttackInfo;

    constexpr int attackers_index = 11;
    constexpr AttackInfo clear_attackers         = 0b0000011111111111;
    const AttackInfo attack_mask[NPIECE_TYPES] = { 0b0000010000000000,
                                                   0b0000001000000000,
                                                   0b0000000100000000,
                                                   0b0000000010000000,
                                                   0b0000000001000000,
                                                   0b0000000000100000 };
    constexpr AttackInfo attacks_mask            = 0b0000011111100000;

    const AttackInfo drop_mask[NPIECE_TYPES] =   { 0b0000000000010000,
                                                   0b0000000000001000,
                                                   0b0000000000000100,
                                                   0b0000000000000010,
                                                   0b0000000000000001,
                                                   0b0000000000000000 };
    constexpr AttackInfo drops_mask              = 0b0000000000011111;

    constexpr void add_attack_info(AttackInfo& info, PieceType piece) {
        AttackInfo num_attackers = info >> attackers_index;
        info = (info & clear_attackers) | (++num_attackers << attackers_index) | attack_mask[piece];
    }

    constexpr void add_drop_info(AttackInfo& info, PieceType piece) {
        info |= drop_mask[piece];
    }

    constexpr bool can_attack(AttackInfo& info, PieceType piece) {
        return info & attack_mask[piece];
    }

    constexpr bool can_drop(AttackInfo& info, PieceType piece) {
        return info & drop_mask[piece];
    }

    constexpr int attacks_num(AttackInfo& info) {
        return static_cast<int>(info >> attackers_index);
    }

    constexpr void print_attack_info(AttackInfo& info) {
        int num_attacks = info >> attackers_index;
        if (num_attacks) {
            std::cout << num_attacks << " attacks by ";
            for (int i = 0; i < NPIECE_TYPES; i++) {
                if (info & attack_mask[i])
                    std::cout << PIECE_STR[i];
            }
        } else {
            std::cout << "no attacks, ";
        }

        if (info & drops_mask) {
            std::cout << ", drops: ";
        } else {
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

        for (int i = 15; i >= 0; i--) {
            std::cout << ((info & (1 << i)) ? 1 : 0);
        }
    }

    typedef uint8_t EvalMask;

    constexpr EvalMask material_mask        = 0b00000001;
    constexpr EvalMask pawn_structure_mask  = 0b00000010;
    constexpr EvalMask king_safety_mask     = 0b00000100;
    constexpr EvalMask piece_placement_mask = 0b00001000;
    constexpr EvalMask board_control_mask   = 0b00010000;

    typedef uint16_t TestMask;

    constexpr TestMask checking_moves_mask  = 0b0000000000100000;
    constexpr TestMask forking_moves_mask   = 0b0000000001000000;
    constexpr TestMask dropping_moves_mask  = 0b0000000010000000;
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
    
    enum KingZone {
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
    const double rook_distance_bonus[8] =   { 0.032, 0.024, 0.016, 0.008, 0.000, -0.008, -0.016, -0.024 };
    const double queen_distance_bonus[8] =  { 0.048, 0.036, 0.024, 0.012, 0.000, -0.012, -0.024, -0.036 };

    int diamond_distance_w[8][NSQUARES];
    int diamond_distance_b[8][NSQUARES];
    int cross_distance_w[8][NSQUARES];
    int cross_distance_b[8][NSQUARES];

    constexpr void init_diamond_distances() {
        for (int zone = 0; zone < 8; zone++) {
            for (int square = 0; square < 64; square++) {
                int s_x = file_of(static_cast<Square>(square));
                int s_y = rank_of(static_cast<Square>(square));

                int num_sq_zone_w = 0, num_sq_zone_b = 0;
                int d_sum_w = 0, d_sum_b = 0;

                for (int s = 0; s < 64; s++) {
                    int z_x = file_of(static_cast<Square>(s));
                    int z_y = rank_of(static_cast<Square>(s));

                    // white side
                    if (king_zone_w[s] == zone) {
                        num_sq_zone_w++;
                        d_sum_w += 3 * abs(s_x - z_x) + 2 * abs(s_y - z_y);
                    }

                    // black side
                    if (king_zone_b[s] == zone) {
                        num_sq_zone_b++;
                        d_sum_b += 3 * abs(s_x - z_x) + 2 * abs(s_y - z_y);
                    }
                }

                diamond_distance_w[zone][square] = fmin(7, fmax(0, d_sum_w / (3 * num_sq_zone_w) - 1));
                diamond_distance_b[zone][square] = fmin(7, fmax(0, d_sum_b / (3 * num_sq_zone_b) - 1));
            }
        }
    }

    constexpr void init_cross_distances() {
        for (int zone = 0; zone < 8; zone++) {
            for (int square = 0; square < 64; square++) {
                int s_x = file_of(static_cast<Square>(square));
                int s_y = rank_of(static_cast<Square>(square));

                int num_sq_zone_w = 0, num_sq_zone_b = 0;
                int d_sum_w = 0, d_sum_b = 0;

                for (int s = 0; s < 64; s++) {
                    int z_x = file_of(static_cast<Square>(s));
                    int z_y = rank_of(static_cast<Square>(s));

                    // white side
                    if (king_zone_w[s] == zone) {
                        num_sq_zone_w++;
                        d_sum_w += fmin(4 * abs(s_x - z_x), 3 * abs(s_y - z_y));
                    }

                    // black side
                    if (king_zone_b[s] == zone) {
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

    constexpr void initialise_eval_tables() {
        init_diamond_distances();
        init_cross_distances();
    }

    // ------------------------------ MCTS RELATED ------------------------------

    constexpr int cpuct = 1;
    constexpr double cpuct_init = 2.5;
    constexpr int cpuct_base = 19652;
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

    typedef std::unordered_map<std::string, std::unordered_map<int, double>> Qsa_t;
    typedef std::unordered_map<std::string, std::unordered_map<int, long>> Nsa_t;
    typedef std::unordered_map<std::string, long> Ns_t;
    typedef std::unordered_map<std::string, std::unordered_map<int, double>> Ps_t;
    typedef std::unordered_map<std::string, double> Es_t;
    typedef std::unordered_map<std::string, std::unordered_map<int, Move>> Ls_t;

    enum class BestMoveStrat {
        Default,
        Q_value,
        NUM
    };

    enum class PolicyEnhancementStrat {
        Dirichlet,
        DroppingMoves,
        CheckingMoves,
        CapturingMoves,
        ForkingMoves,
        NUM
    };

    enum class NodeExpansionStrat {
        Default,
        Exploration,
        NUM
    };

    enum class BackpropStrat {
        Default,
        SMA,
        NUM
    };

    struct MCTS_config {

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
                        eval_types(0), time_saving_config(false), ts_eval_types(0) {}

        ~MCTS_config() = default;
    };
}

#ifndef CRAZYZERO_HPP
#define CRAZYZERO_HPP

#include <stdint.h>
#include <string>
#include <list>
#include <vector>
#include <unordered_map>
#include <map>
#include <math.h>
#include <time.h>
#include <random>
#include <limits>
#include <chrono>
#include <algorithm>
#include <execution>
#include <functional>
#include "surge/position.h"
#include "surge/tables.h"
#include "surge/types.h"
#include "utils.h"
#include "cppflow/cppflow.h"

#define NNET_MODEL_PATH "./model"

namespace crazyzero
{
    //////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////// MAIN CLASSES //////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////

    //Main representation of the game board.
    class Board
    {
    private:
        std::string starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1";
    public:
        Position p;
        long long hash;

        Board()
        {
            Position::set(starting_fen, p);
            calc_hash();
        }

        ~Board() = default;

        inline void reset();
        inline std::string fen();
        inline void set_fen(const std::string& fen);
        inline void push(Move& move);
        inline void push_encoded(const int move);
        inline void pop(Move& move);
        inline move_vector<Move> legal_moves();
        inline double end_score(const Color c);
        inline cppflow::tensor input_representation();
        inline std::string san(Move& move);
        inline bool gives_check(Move& move);
        inline bool gives_fork(Move& move);
        inline double eval_drop(Move& move);

    private:
        inline void calc_hash();
    };

    //Neural network implementation.
    class NNet
    {
    public:
        cppflow::model* model = nullptr;

        NNet() = default;

        ~NNet()
        {
            if (model != nullptr)
                delete model;
        }

        //Initializes the neural network from a file.
        inline void init() { model = new cppflow::model(NNET_MODEL_PATH); }

        //Returns the neural network prediction of the given board's position.
        std::pair<std::vector<float>, float> predict(Board& board)
        {
            cppflow::tensor input = board.input_representation();
            auto output = (model->operator())({ {"serving_default_input_1:0", input} }, { "StatefulPartitionedCall:0", "StatefulPartitionedCall:1" });
            std::pair<std::vector<float>, float> prediction(output[0].get_data<float>(), output[1].get_data<float>()[0]);
            return prediction;
        }
    };

    //Evaluation function implementation.
    class Evaluator
    {
    private:
        AttackInfo WT[NSQUARES];
        AttackInfo BT[NSQUARES];

    public:
        EvalMask eval_types;

        constexpr Evaluator() : eval_types(0U) {}
        ~Evaluator() = default;

        inline int q_to_cp(double q);
        inline double cp_to_q(double cp);
        inline void add_eval(EvalMask mask);
        inline void remove_eval(EvalMask mask);
        inline bool has_eval(EvalMask mask);
        inline void update_tables(Board& board);
        inline double material(Board& board);
        inline double pawn_structure(Board& board);
        inline double king_safety(Board& board);
        inline double piece_placement(Board& board);
        inline double board_control(Board& board);
        inline double eval(Board& board);
    };

    //Depth-first mate search implementation.
    class MateSearch
    {
    private:
        //Recursive DFS implementation.
        inline bool find_mate(Board board, Move move, const int depth)
        {
            board.push(move);
            EndType end = board.p.is_checkmate();

            if (depth == max_depth)
                return (end == CHECKMATE);

            if (end != NONE)
                return (board.p.turn() != player && end == CHECKMATE);

            for (Move& move : board.legal_moves())
            {
                if (find_mate(board, move, depth + 1))
                    return true;
            }

            return false;
        }

    public:
        Color player;

        MateSearch() = default;
        ~MateSearch() = default;

        //Returns the move that leads to a direct mate. If no move is found, an empty move is returned.
        inline Move mate_move(Board& board)
        {
            player = board.p.turn();
            for (Move& move : board.legal_moves())
            {
                if (find_mate(board, move, 1))
                    return move;
            }

            return Move();
        }
    };

    //Monte-Carlo tree search implementation.
    class MCTS
    {
    public:
        MD_t move_data;
        
        NNet nnet;
        Evaluator eval;
        MateSearch mate_search;

        Dirichlet dirichlet;

        MCTS_config config;
        bool time_control;
        bool variable_time_control;
        int num_sims;
        bool initialized = false;
        Color player;
        long long time_per_move;
        long long original_time;
        bool time_saving_mode;
        long long time_simulating;
        int executed_moves;
        int explored_nodes;
        int best_move_cp;
        bool mode_switch;

        MCTS()
        {
            time_control = true;
            variable_time_control = false;
            num_sims = 100;

            player = NO_COLOR;
            time_per_move = -1LL;
            original_time = -1LL;
            time_saving_mode = false;
            time_simulating = 0LL;
            executed_moves = 0;
            explored_nodes = 0;
            best_move_cp = 0;
            mode_switch = false;

            // initialize playing strategies
            set_best_move_strategy(BestMoveStrat::Default);
            set_node_expansion_strategy(NodeExpansionStrat::Default);
            set_backprop_strategy(BackpropStrat::Default);

            add_policy_enhancement_strategy(PolicyEnhancementStrat::Dirichlet);
        }

        ~MCTS() = default;

        inline void init(Board& board);
        inline void init_time(const int available_time, const int increment);
        inline void update_time(const int remaining_time);
        inline void update_config();
        inline void reset();
        inline void soft_reset();
        inline void add_policy_enhancement_strategy(const PolicyEnhancementStrat policy_type);

        inline Move best_move(Board& board);
        inline void search(Board board);

    private:
        inline void on_mode_switch(bool state);
        inline void set_best_move_strategy(const BestMoveStrat best_move_type);
        inline void set_node_expansion_strategy(const NodeExpansionStrat expansion_type);
        inline void set_backprop_strategy(const BackpropStrat backprop_type);
        inline void remove_policy_enhancement_strategies();

        // ----------------------- STRATEGY INSTANCES ------------------------
        Move& (*best_move_strat)(move_vector<Move>&);
        std::vector<void (*)(MCTS*, Board&, move_vector<Move>&)> policy_strats;
        Move& (*expansion_strat)(move_vector<Move>&);
        double (*backprop_strat)(const Move&, const double&);
    };

    //////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////// MCTS STRATEGIES /////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////

    //Strategy to choose the best move to make.
    inline Move& best_move_nvisits(move_vector<Move>& moves);
    inline Move& best_move_qvalue(move_vector<Move>& moves);

    //Strategy to choose the next move to expand during a MCTS simulation.
    inline Move& move_to_expand_default(move_vector<Move>& moves);
    inline Move& move_to_expand_inc(move_vector<Move>& moves);

    //Strategy to calculate Q-values during backpropagation.
    inline double backprop_nvisits_qvalue(const Move& move, const double& v);
    inline double backprop_sma(const Move& move, const double& v);

    //Strategy for enhancing the prior probability of legal moves returned by the neural network.
    inline void enhance_policy_dirichlet(MCTS* mcts, Board& board, move_vector<Move>& moves);
    inline void enhance_policy_checking_moves(MCTS* mcts, Board& board, move_vector<Move>& moves);
    inline void enhance_policy_forking_moves(MCTS* mcts, Board& board, move_vector<Move>& moves);
    inline void enhance_policy_dropping_moves(MCTS* mcts, Board& board, move_vector<Move>& moves);
    inline void enhance_policy_capturing_moves(MCTS* mcts, Board& board, move_vector<Move>& moves);

    //////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////// BOARD CLASS MEMBERS ///////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////

    //Resets the board to the starting position.
    inline void Board::reset()
    {
        p = Position();
        Position::set(starting_fen, p);
        calc_hash();
    }

    //Returns the FEN string of the current board position.
    inline std::string Board::fen() { return p.fen(); }

    //Sets the board position to the one described by the given FEN string.
    inline void Board::set_fen(const std::string& fen)
    {
        Position::set(fen, p);
        calc_hash();
    }

    //Executes a move and updates the board position.
    inline void Board::push(Move& move)
    {
        if (p.turn() == WHITE)
            p.play<WHITE>(move);
        else
            p.play<BLACK>(move);
        calc_hash();
    }

    //Executes a move represented by its encoded value and updates the board position.
    inline void Board::push_encoded(const int move)
    {
        Move m = legal_moves()[move];
        push(m);
    }

    //Undoes a given move and updates the board position.
    inline void Board::pop(Move& move)
    {
        if (p.turn() == WHITE)
            p.undo<BLACK>(move);
        else
            p.undo<WHITE>(move);
        calc_hash();
    }

    inline move_vector<Move> Board::legal_moves() { return (p.turn() == WHITE) ? p.generate_legals<WHITE>() : p.generate_legals<BLACK>(); }

    //Returns a value representing weather the given player won or lost, a draw occured or the game did not end.
    inline double Board::end_score(const Color c)
    {
        switch (c)
        {
        case WHITE:
            return p.end_score<WHITE>();
        case BLACK:
            return p.end_score<BLACK>();
        }
    }

    //Returns a representation of the current board position that can be used as an input to the neural network.
    inline cppflow::tensor Board::input_representation()
    {
        int size = 34 * 64;
        std::vector<float> input_rep(size);

        int start_index = 0;

        // pieces positions for each player (12 layers)
        // white (6 layers)
        for (int piece = PAWN; piece <= KING; piece++)
        {
            Bitboard squares = p.bitboard_of(WHITE, (PieceType)piece);
            Square s;
            while (squares)
            {
                s = pop_lsb(&squares);
                input_rep[start_index + s] = 1.0f;
            }
            start_index += 64;
        }

        // black (6 layers)
        for (int piece = PAWN; piece <= KING; piece++)
        {
            Bitboard squares = p.bitboard_of(BLACK, (PieceType)piece);
            Square s;
            while (squares)
            {
                s = pop_lsb(&squares);
                input_rep[start_index + s] = 1.0f;
            }
            start_index += 64;
        }

        // how often the board position has occured (2 layers)
        float reps = static_cast<float>(p.repetitions[p.fen_board()]) / REPETITIONS_NORM;
        for (int i = 0; i < 128; i++)
        {
            input_rep[start_index + i] = reps;
        }
        start_index += 128;

        // pocket counts (10 layers)
        // white (5 layers)
        for (int piece = PAWN; piece <= QUEEN; piece++)
        {
            float count = static_cast<float>(p.pocket_count(WHITE, (PieceType)piece)) / POCKET_COUNT_NORM;
            for (int i = 0; i < 64; i++)
            {
                input_rep[start_index + i] = count;
            }
            start_index += 64;
        }

        // black (5 layers)
        for (int piece = PAWN; piece <= QUEEN; piece++)
        {
            float count = static_cast<float>(p.pocket_count(BLACK, (PieceType)piece)) / POCKET_COUNT_NORM;
            for (int i = 0; i < 64; i++)
            {
                input_rep[start_index + i] = count;
            }
            start_index += 64;
        }

        // promoted pieces (pawns) (2 layers)
        Bitboard promoted_pawns = p.promoted;
        int black_start_index = start_index + 64;
        while (promoted_pawns)
        {
            int square = pop_lsb(&promoted_pawns);
            Color piece = color_of(p.at((Square)square));
            if (piece == WHITE)
            {
                input_rep[start_index + square] = 1.0f;
            } else
            {
                input_rep[black_start_index + square] = 1.0f;
            }
        }
        start_index = black_start_index + 64;

        // en-passant square (1 layer)
        int en_pass = p.en_passant();
        if (en_pass != NO_SQUARE)
            input_rep[start_index + en_pass] = 1.0f;
        start_index += 64;

        // color (1 layer)
        if (p.turn() == WHITE)
        {
            for (int i = 0; i < 64; i++)
                input_rep[start_index + i] = 1.0f;
        }
        start_index += 64;

        // total move count (1 layer)
        float total_moves = static_cast<float>(p.fullmove_number()) / REPETITIONS_NORM;
        for (int i = 0; i < 64; i++)
            input_rep[start_index + i] = total_moves;
        start_index += 64;

        // castling rights (4 layers)
        // white (2 layers)
        if (p.has_kingside_castling_rights(WHITE))
        {
            for (int i = 0; i < 64; i++)
                input_rep[start_index + i] = 1.0f;
        }
        start_index += 64;
        if (p.has_queenside_castling_rights(WHITE))
        {
            for (int i = 0; i < 64; i++)
                input_rep[start_index + i] = 1.0f;
        }
        start_index += 64;

        // black (2 layers)
        if (p.has_kingside_castling_rights(BLACK))
        {
            for (int i = 0; i < 64; i++)
                input_rep[start_index + i] = 1.0f;
        }
        start_index += 64;
        if (p.has_queenside_castling_rights(BLACK))
        {
            for (int i = 0; i < 64; i++)
                input_rep[start_index + i] = 1.0f;
        }
        start_index += 64;

        // no-progress count (halfmove count) (1 layer)
        float half_moves = static_cast<float>(p.halfmove_clock()) / HALFMOVES_NORM;
        for (int i = 0; i < 64; i++)
            input_rep[start_index + i] = half_moves;

        return cppflow::tensor(input_rep, { 1, 34, 64 });
    }

    //Returns the given move represented in the SAN notation.
    inline std::string Board::san(Move& move)
    {
        std::stringstream os;
        int flag = move.flags();
        if (flag == OO)
        {
            os << "O-O";
            return os.str();
        } else if (flag == OOO)
        {
            os << "O-O-O";
            return os.str();
        } else if (flag >= DROP_PAWN && flag <= DROP_QUEEN)
        {
            // encode drop
            switch (flag - DROP_PAWN)
            {
            case PAWN:
                os << "P@" << SQSTR[move.from()];
                break;
            case KNIGHT:
                os << "N@" << SQSTR[move.from()];
                break;
            case BISHOP:
                os << "B@" << SQSTR[move.from()];
                break;
            case ROOK:
                os << "R@" << SQSTR[move.from()];
                break;
            case QUEEN:
                os << "Q@" << SQSTR[move.from()];
                break;
            default:
                os << "Unknown drop";
                break;
            }
            return os.str();
        }

        bool multiple = false, same_rank = false, same_file = false;
        for (const Move& legal_m : legal_moves())
        {
            if (legal_m.from() != legal_m.to() && legal_m.to() == move.to() && legal_m.from() != move.from())
            {
                if (rank_of(legal_m.from()) == rank_of(move.from()))
                    same_rank = true;
                if (file_of(legal_m.from()) == file_of(move.from()))
                    same_file = true;
                multiple = true;
            }
        }

        PieceType piece = type_of(p.at(move.from()));

        switch (flag)
        {
        case CAPTURE:
            if (piece != PAWN)
            {
                os << PIECE_STR[piece];
                if (same_rank && same_file)
                    os << SQSTR[move.from()];
                else if (same_rank)
                    os << FILE_STR[file_of(move.from())];
                else if (same_file)
                    os << RANK_STR[rank_of(move.from())];
                else if (multiple)
                    os << FILE_STR[file_of(move.from())];
            } else
            {
                os << FILE_STR[file_of(move.from())];
                if (same_file)
                    os << RANK_STR[rank_of(move.from())];
            }
            os << "x" << SQSTR[move.to()];
            break;
        case PC_KNIGHT:
            os << FILE_STR[file_of(move.from())];
            if (same_file)
                os << RANK_STR[rank_of(move.from())];
            os << "x" << SQSTR[move.to()] << "=N";
            break;
        case PC_BISHOP:
            os << FILE_STR[file_of(move.from())];
            if (same_file)
                os << RANK_STR[rank_of(move.from())];
            os << "x" << SQSTR[move.to()] << "=B";
            break;
        case PC_ROOK:
            os << FILE_STR[file_of(move.from())];
            if (same_file)
                os << RANK_STR[rank_of(move.from())];
            os << "x" << SQSTR[move.to()] << "=R";
            break;
        case PC_QUEEN:
            os << FILE_STR[file_of(move.from())];
            if (same_file)
                os << RANK_STR[rank_of(move.from())];
            os << "x" << SQSTR[move.to()] << "=Q";
            break;
        case PR_KNIGHT:
            os << SQSTR[move.to()] << "=N";
            break;
        case PR_BISHOP:
            os << SQSTR[move.to()] << "=B";
            break;
        case PR_ROOK:
            os << SQSTR[move.to()] << "=R";
            break;
        case PR_QUEEN:
            os << SQSTR[move.to()] << "=Q";
            break;
        case EN_PASSANT:
            os << FILE_STR[file_of(move.from())];
            if (same_file)
                os << RANK_STR[rank_of(move.from())];
            os << "x" << SQSTR[move.to()];
            break;
        default:
            // quiet moves and double push
            if (piece != PAWN)
            {
                os << PIECE_STR[piece];
                if (same_rank && same_file)
                    os << SQSTR[move.from()];
                else if (same_rank)
                    os << FILE_STR[file_of(move.from())];
                else if (same_file)
                    os << RANK_STR[rank_of(move.from())];
                else if (multiple)
                    os << FILE_STR[file_of(move.from())];
            }
            os << SQSTR[move.to()];
            break;
        }
        return os.str();
    }

    //Returns true, if the given move results in a check.
    inline bool Board::gives_check(Move& move)
    {
        bool check = false;
        switch (p.turn())
        {
        case WHITE:
            p.play<WHITE>(move);
            check = p.in_check<BLACK>();
            p.undo<WHITE>(move);
            break;
        case BLACK:
            p.play<BLACK>(move);
            check = p.in_check<WHITE>();
            p.undo<BLACK>(move);
            break;
        }
        return check;
    }

    //Returns true, if the given move results in a fork.
    inline bool Board::gives_fork(Move& move)
    {
        bool fork = false;
        switch (p.turn())
        {
        case WHITE:
            p.play<WHITE>(move);
            break;
        case BLACK:
            p.play<BLACK>(move);
            break;
        }

        Bitboard b = attacks(type_of(p.at(move.to())), move.to(), p.all_pieces<WHITE>() | p.all_pieces<BLACK>());
        if (p.turn() == WHITE)
            b &= p.all_pieces<WHITE>();
        else
            b &= p.all_pieces<BLACK>();

        if (pop_count(b) >= 2)
            fork = true;

        switch (~p.turn())
        {
        case WHITE:
            p.undo<WHITE>(move);
            break;
        case BLACK:
            p.undo<BLACK>(move);
            break;
        }
        return fork;
    }

    //Returns an evaluation of the given dropping move, used for the Dropping moves policy enhancement.
    inline double Board::eval_drop(Move& move)
    {
        Square w_king = bsf(p.bitboard_of(WHITE_KING));
        Square b_king = bsf(p.bitboard_of(BLACK_KING));
        //Bitboard w_all = p.all_pieces<WHITE>();
        //Bitboard b_all = p.all_pieces<BLACK>();
        double factor = 0.0;

        // all drops should defend own king
        if (p.turn() == WHITE && attacks<KING>(w_king, 0ULL) & SQUARE_BB[move.to()])
            factor += drop_king_def_bonus;
        else if (p.turn() == BLACK && attacks<KING>(b_king, 0ULL) & SQUARE_BB[move.to()])
            factor += drop_king_def_bonus;

        switch (move.flags())
        {
            // pawn
            // 1. drops deep in enemy territory
        case DROP_PAWN:
            factor += (p.turn() == WHITE) ? drop_pawn_location_w[move.to()] : drop_pawn_location_b[move.to()];
            break;
            // knight
            // 1. attack weak spaces arround enemy king
            // 2. bonus if on rank 5 (white) or rank 4 (black)
        case DROP_KNIGHT:
            if (p.turn() == WHITE)
            {
                if (attacks<KING>(b_king, 0ULL) & attacks<KNIGHT>(move.to(), 0ULL))
                    factor += drop_knight_attack_king_bonus;
                if (rank_of(move.to()) == RANK5)
                    factor += drop_knight_rank_bonus;
            } else
            {
                if (attacks<KING>(w_king, 0ULL) & attacks<KNIGHT>(move.to(), 0ULL))
                    factor += drop_knight_attack_king_bonus;
                if (rank_of(move.to()) == RANK4)
                    factor += drop_knight_rank_bonus;
            }
            break;
            // bishop
            // no use, other than defending the king
        case DROP_BISHOP:
            break;
            // rook
            // 1. exploit enemy's back rank
        case DROP_ROOK:
            if (p.turn() == WHITE && rank_of(move.to()) == RANK8)
                factor += drop_rook_rank_bonus;
            else if (p.turn() == BLACK && rank_of(move.to()) == RANK1)
                factor += drop_rook_rank_bonus;
            break;
            // queen
            // no use, other than defending the king
        case DROP_QUEEN:
            break;
        }
        return factor;
    }

    inline void Board::calc_hash()
    {
        const int p = 31;
        const int m = 1e9 + 9;
        this->hash = 0LL;
        long long p_pow = 1LL;
        for (char c : this->p.fen())
        {
            this->hash = (this->hash + (c - 'a' + 1) * p_pow) % m;
            p_pow = (p_pow * p) % m;
        }
        //this->hash = p.fen();
    }

    //////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////// MCTS CLASS MEMBERS ///////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////

    //Initializes and warms up the neural network.
    inline void MCTS::init(Board& board)
    {
        if (!initialized)
        {
            nnet.init();
            nnet.predict(board);    // warm up the nnet
            initialized = true;
        }
    }

    //Initializes the time control system.
    inline void MCTS::init_time(const int available_time, const int increment)
    {
        time_per_move = static_cast<long long>(available_time / moves_per_game) + static_cast<long long>(increment * increment_amount);
        original_time = time_per_move;
    }

    //Updates the time control system.
    inline void MCTS::update_time(const int remaining_time)
    {
        if (!time_saving_mode && executed_moves >= moves_per_game - 1)
            time_saving_mode = true;

        if (time_saving_mode)
        {
            if (remaining_time <= original_time + original_time / 2LL)
            {
                time_per_move = static_cast<long long>(static_cast<double>(remaining_time) * time_proportion);

                if (!mode_switch)
                {
                    on_mode_switch(true);
                    mode_switch = true;
                }
            } else
            {
                time_per_move = static_cast<long long>(static_cast<double>(original_time) * original_time_amount);

                if (mode_switch)
                {
                    on_mode_switch(false);
                    mode_switch = false;
                }
            }
        }
    }

    //Updates the configuration of modifications.
    inline void MCTS::update_config()
    {
        time_control = config.time_control;
        variable_time_control = config.variable_time_control;
        num_sims = config.num_sims;

        set_best_move_strategy(config.best_move_strategy);
        set_node_expansion_strategy(config.node_expansion_strategy);
        set_backprop_strategy(config.backprop_strategy);

        eval.eval_types = config.eval_types;

        remove_policy_enhancement_strategies();
        for (auto strat : config.policy_strategies)
        {
            add_policy_enhancement_strategy(strat);
        }

        config.changed = false;
    }

    //Switches the MCTS modification configuration. Called when entering/leaving the time saving mode.
    inline void MCTS::on_mode_switch(bool state)
    {
        if (variable_time_control)
            time_control = state;

        if (config.time_saving_config && state)
        {
            eval.eval_types = config.ts_eval_types;

            remove_policy_enhancement_strategies();
            for (auto strat : config.ts_policy_strategies)
                add_policy_enhancement_strategy(strat);
        } else if (config.time_saving_config && !state)
        {
            eval.eval_types = config.eval_types;

            remove_policy_enhancement_strategies();
            for (auto strat : config.policy_strategies)
                add_policy_enhancement_strategy(strat);
        }
    }

    //Sets the best move determination function to use.
    inline void MCTS::set_best_move_strategy(const BestMoveStrat best_move_type)
    {
        switch (best_move_type)
        {
        case BestMoveStrat::Q_value:
            best_move_strat = &best_move_qvalue;
            break;
        case BestMoveStrat::Default:
            best_move_strat = &best_move_nvisits;
            break;
        default:
            throw std::runtime_error("MCTS ERROR: BestMoveStrategy is of unknown type.");
        }
    }

    //Sets the node expansion function to use.
    inline void MCTS::set_node_expansion_strategy(const NodeExpansionStrat expansion_type)
    {
        switch (expansion_type)
        {
        case NodeExpansionStrat::Exploration:
            expansion_strat = &move_to_expand_inc;
            break;
        case NodeExpansionStrat::Default:
            expansion_strat = &move_to_expand_default;
            break;
        default:
            throw std::runtime_error("MCTS ERROR: NodeExpansionStrategy is of unknown type.");
        }
    }

    //Sets the backpropagation function to use.
    inline void MCTS::set_backprop_strategy(const BackpropStrat backprop_type)
    {
        switch (backprop_type)
        {
        case BackpropStrat::SMA:
            backprop_strat = &backprop_sma;
            break;
        case BackpropStrat::Default:
            backprop_strat = &backprop_nvisits_qvalue;
            break;
        default:
            throw std::runtime_error("MCTS ERROR: BackpropStrategy is of unknown type.");
        }
    }

    //Adds a given policy enhancement modification to the configuration.
    inline void MCTS::add_policy_enhancement_strategy(const PolicyEnhancementStrat policy_type)
    {
        switch (policy_type)
        {
        case PolicyEnhancementStrat::Dirichlet:
            policy_strats.push_back(&enhance_policy_dirichlet);
            break;
        case PolicyEnhancementStrat::CheckingMoves:
            policy_strats.push_back(&enhance_policy_checking_moves);
            break;
        case PolicyEnhancementStrat::ForkingMoves:
            policy_strats.push_back(&enhance_policy_forking_moves);
            break;
        case PolicyEnhancementStrat::DroppingMoves:
            policy_strats.push_back(&enhance_policy_dropping_moves);
            break;
        case PolicyEnhancementStrat::CapturingMoves:
            policy_strats.push_back(&enhance_policy_capturing_moves);
            break;
        default:
            throw std::runtime_error("MCTS ERROR: PolicyEnhancementStrategy is of unknown type.");
        }
    }

    //Removes all function pointers to policy enhancement strategies.
    inline void MCTS::remove_policy_enhancement_strategies() { policy_strats.clear(); }

    //Resets and gets ready for a new game.
    inline void MCTS::reset()
    {
        move_data.clear();

        player = NO_COLOR;
        time_per_move = -1LL;
        original_time = -1LL;
        time_saving_mode = false;
        time_simulating = 0LL;
        executed_moves = 0;
        explored_nodes = 0;
        best_move_cp = 0;
        mode_switch = false;

        update_config();
    }

    //Resets, but keeps the configuration of modifications.
    inline void MCTS::soft_reset()
    {
        move_data.clear();

        player = NO_COLOR;
        time_per_move = -1LL;
        original_time = -1LL;
        time_saving_mode = false;
        time_simulating = 0LL;
        executed_moves = 0;
        explored_nodes = 0;
        best_move_cp = 0;
        mode_switch = false;
    }

    //Returns the best move in the given board's position.
    inline Move MCTS::best_move(Board& board)
    {
        //If only one move is available, choose it.
        move_vector<Move> moves = board.legal_moves();
        if (moves.size() == 1)
            return moves.front();

        //Perform a quick mate search to see if a mate can be forced.
        Move best_move = mate_search.mate_move(board);
        if (best_move.from() != NO_SQUARE)
            return best_move;

        //Perform simulations.
        explored_nodes = 0;
        if (time_control)
        {
            long long sim_time = time_per_move;
            bool half_time = false;
            while (sim_time > 0LL)
            {
                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                search(board);
                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                sim_time -= std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
                explored_nodes++;
            }

            time_simulating = time_per_move - sim_time;
        } else
        {
            for (int i = 0; i < num_sims; i++)
            {
                search(board);
                explored_nodes++;
            }
        }

        //Choose best move.
        return (*best_move_strat)(move_data[board.hash]);
    }

    //Performs a simulation/rollout.
    inline void MCTS::search(Board board)
    {
        std::vector<std::pair<move_vector<Move>&, Move&>> state_stack;
        double v = 0.0;

        //Find a leaf or terminal node.
        while (true)
        {
            long long state = board.hash;

            if (!move_data.contains(state))
            {
                double es = board.end_score(player);
                if (es != 0.0)
                {
                    //Terminal node.

                    //Draws are not desired, but still worth if no better option exists.
                    if (es > 0.0 && es < 0.5)
                    {
                        v = -es;
                        break;
                    }

                    v = 1.0;
                    break;
                }

                //Leaf node.
                move_data[state] = board.legal_moves();
                move_vector<Move>& moves = move_data[state];
                moves.end_score = es;

                //Predict policy and value with nnet.
                auto [policy, value] = nnet.predict(board);
                if (eval.eval_types)
                    value = static_cast<float>(1.0 - eval_factor) * value + static_cast<float>(eval_factor * eval.eval(board));

                //Normalize and store policy of moves.
                double sum_policy = 0.0;
                for (Move& move : moves)
                {
                    double p = static_cast<double>(policy[move.hash()]);
                    sum_policy += p;
                    move.policy = p;
                }

                for (Move& move : moves)
                    move.policy /= sum_policy;

                //Enhance policy with additional strategies.
                for (auto policy_strat : policy_strats)
                    (*policy_strat)(this, board, moves);

                v = static_cast<double>(-value);
                break;
            }

            //Node was already visited. Choose move to expand.
            move_vector<Move>& moves = move_data[state];
            Move& move = (moves.size() == 1) ? moves.front() : (*expansion_strat)(moves);

            //Remember move choice in current state for backpropagation.
            state_stack.push_back(std::pair<move_vector<Move>&, Move&>(moves, move));

            //Expand move and descend into the next state.
            board.push(move);
        }

        //Search is done. Now backpropagate and update Q values back to the root.
        while (state_stack.size() > 0)
        {
            auto [moves, move] = state_stack.back();
            if (move.n_visits)
                move.Q_value = (*backprop_strat)(move, v);
            else
                move.Q_value = v;

            move.n_visits++;
            moves.n_visits++;

            v = -v;
            state_stack.pop_back();
        }
    }

    //////////////////////////////////////////////////////////////////////////////////
    //////////////////////////// EVALUATOR CLASS MEMBERS /////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////

    //Converts a given Q-value to Centipawn.
    inline int Evaluator::q_to_cp(double q) { return static_cast<int>(111.714640912 * tan(1.5620688421 * q)); }

    //Converts a given Centipawn value to a Q-value.
    inline double Evaluator::cp_to_q(double cp) { return 0.64018 * atan(0.00895 * cp); }

    //Adds the given feature to the evaluation function.
    inline void Evaluator::add_eval(EvalMask mask) { eval_types |= mask; }

    //Removes the given feature from the evaluation function.
    inline void Evaluator::remove_eval(EvalMask mask) { eval_types &= ~mask; }

    //Returns true, if the given feature is already part of the evaluation function.
    inline bool Evaluator::has_eval(EvalMask mask) { return eval_types & mask; }

    //Re-computes the values of WT and BT internal structures.
    inline void Evaluator::update_tables(Board& board)
    {
        // clear tables
        long* PWB = (long*)WT;
        long* PBB = (long*)BT;

        PWB[0] = PWB[1] = PWB[2] = PWB[3] = PWB[4] = PWB[5] = PWB[6] = PWB[7] = PWB[8] = PWB[9] = PWB[10] = PWB[11] = PWB[12] = PWB[13] = PWB[14] = PWB[15]
            = PWB[16] = PWB[17] = PWB[18] = PWB[19] = PWB[20] = PWB[21] = PWB[22] = PWB[23] = PWB[24] = PWB[25] = PWB[26] = PWB[27] = PWB[28] = PWB[29] = PWB[30] = PWB[31] = 0;
        PBB[0] = PBB[1] = PBB[2] = PBB[3] = PBB[4] = PBB[5] = PBB[6] = PBB[7] = PBB[8] = PBB[9] = PBB[10] = PBB[11] = PBB[12] = PBB[13] = PBB[14] = PBB[15]
            = PBB[16] = PBB[17] = PBB[18] = PBB[19] = PBB[20] = PBB[21] = PBB[22] = PBB[23] = PBB[24] = PBB[25] = PBB[26] = PBB[27] = PBB[28] = PBB[29] = PBB[30] = PBB[31] = 0;

        // recompute info about attacks and drops for each piece
        Bitboard all = board.p.all_pieces<WHITE>() | board.p.all_pieces<BLACK>();
        Bitboard empty = ~all;
        Bitboard attack_squares, piece_squares;

        for (int piece = 0; piece < NPIECE_TYPES; piece++)
        {
            // first attacks ...
            // white side
            piece_squares = board.p.bitboard_of(make_piece(WHITE, static_cast<PieceType>(piece)));
            while (piece_squares)
            {
                Square piece_s = pop_lsb(&piece_squares);
                attack_squares = (piece == PAWN) ? pawn_attacks<WHITE>(piece_s) : attacks(static_cast<PieceType>(piece), piece_s, all);

                while (attack_squares)
                {
                    Square attack_s = pop_lsb(&attack_squares);
                    add_attack_info(WT[attack_s], static_cast<PieceType>(piece));
                }
            }

            // black side
            piece_squares = board.p.bitboard_of(make_piece(BLACK, static_cast<PieceType>(piece)));
            while (piece_squares)
            {
                Square piece_s = pop_lsb(&piece_squares);
                attack_squares = (piece == PAWN) ? pawn_attacks<BLACK>(piece_s) : attacks(static_cast<PieceType>(piece), piece_s, all);

                while (attack_squares)
                {
                    Square attack_s = pop_lsb(&attack_squares);
                    add_attack_info(BT[attack_s], static_cast<PieceType>(piece));
                }
            }

            // ... then drops
            // white side
            int pocket_c = board.p.pocket_count(WHITE, static_cast<PieceType>(piece));
            if (pocket_c)
            {
                while (empty)
                {
                    Square drop_s = pop_lsb(&empty);
                    add_drop_info(WT[drop_s], static_cast<PieceType>(piece));
                }
            }
            empty = ~all;

            // black side
            pocket_c = board.p.pocket_count(BLACK, static_cast<PieceType>(piece));
            if (pocket_c)
            {
                while (empty)
                {
                    Square drop_s = pop_lsb(&empty);
                    add_drop_info(BT[drop_s], static_cast<PieceType>(piece));
                }
            }
            empty = ~all;
        }
    }

    //Returns the Centipawn score computed using materal advantage.
    inline double Evaluator::material(Board& board)
    {
        double eval = 0.0;

        // eval pieces on board ...
        eval += material_value[PAWN] * (static_cast<double>(pop_count(board.p.bitboard_of(WHITE_PAWN)))
            - static_cast<double>(pop_count(board.p.bitboard_of(BLACK_PAWN))))
            + material_value[KNIGHT] * (static_cast<double>(pop_count(board.p.bitboard_of(WHITE_KNIGHT)))
                - static_cast<double>(pop_count(board.p.bitboard_of(BLACK_KNIGHT))))
            + material_value[BISHOP] * (static_cast<double>(pop_count(board.p.bitboard_of(WHITE_BISHOP)))
                - static_cast<double>(pop_count(board.p.bitboard_of(BLACK_BISHOP))))
            + material_value[ROOK] * (static_cast<double>(pop_count(board.p.bitboard_of(WHITE_ROOK)))
                - static_cast<double>(pop_count(board.p.bitboard_of(BLACK_ROOK))))
            + material_value[QUEEN] * (static_cast<double>(pop_count(board.p.bitboard_of(WHITE_QUEEN)))
                - static_cast<double>(pop_count(board.p.bitboard_of(BLACK_QUEEN))));
        // ... and in the pocket
        eval += material_value_hand[PAWN] * (static_cast<double>(board.p.pocket_count(WHITE, PAWN))
            - static_cast<double>(board.p.pocket_count(BLACK, PAWN)))
            + material_value_hand[KNIGHT] * (static_cast<double>(board.p.pocket_count(WHITE, KNIGHT))
                - static_cast<double>(board.p.pocket_count(BLACK, KNIGHT)))
            + material_value_hand[BISHOP] * (static_cast<double>(board.p.pocket_count(WHITE, BISHOP))
                - static_cast<double>(board.p.pocket_count(BLACK, BISHOP)))
            + material_value_hand[ROOK] * (static_cast<double>(board.p.pocket_count(WHITE, ROOK))
                - static_cast<double>(board.p.pocket_count(BLACK, ROOK)))
            + material_value_hand[QUEEN] * (static_cast<double>(board.p.pocket_count(WHITE, QUEEN))
                - static_cast<double>(board.p.pocket_count(BLACK, QUEEN)));

        // add bonuses

        // a bonus for each pair of bishops on opposite colored spaces
        int light_w = pop_count(board.p.bitboard_of(WHITE_BISHOP) & light_squares);
        int dark_w = pop_count(board.p.bitboard_of(WHITE_BISHOP) & dark_squares);
        int light_b = pop_count(board.p.bitboard_of(BLACK_BISHOP) & light_squares);
        int dark_b = pop_count(board.p.bitboard_of(BLACK_BISHOP) & dark_squares);
        eval += bishop_pair_bonus * (static_cast<double>(light_w < dark_w ? light_w : dark_w)
            - static_cast<double>(light_b < dark_b ? light_b : dark_b));

        // a bonus for each knight if a friendly queen exists
        eval += knight_queen_bonus * (static_cast<double>(board.p.bitboard_of(WHITE_QUEEN) ? pop_count(board.p.bitboard_of(WHITE_KNIGHT)) : 0)
            - static_cast<double>(board.p.bitboard_of(BLACK_QUEEN) ? pop_count(board.p.bitboard_of(BLACK_KNIGHT)) : 0));

        // a bonus for each pair of bishops and rooks
        int bishop_w = pop_count(board.p.bitboard_of(WHITE_BISHOP));
        int rook_w = pop_count(board.p.bitboard_of(WHITE_ROOK));
        int bishop_b = pop_count(board.p.bitboard_of(BLACK_BISHOP));
        int rook_b = pop_count(board.p.bitboard_of(BLACK_ROOK));
        eval += bishop_pair_bonus * (static_cast<double>(bishop_w < rook_w ? bishop_w : rook_w)
            - static_cast<double>(bishop_b < rook_b ? bishop_b : rook_b));

        // a bonus for each knight for each pawn
        eval += knight_pawn_bonus * (static_cast<double>(pop_count(board.p.bitboard_of(WHITE_KNIGHT))) * static_cast<double>(pop_count(board.p.bitboard_of(WHITE_PAWN)))
            - static_cast<double>(pop_count(board.p.bitboard_of(BLACK_KNIGHT))) * static_cast<double>(pop_count(board.p.bitboard_of(BLACK_PAWN))));

        return (board.p.turn() == WHITE) ? eval : -eval;
    }

    //Returns the Centipawn score computed using pawn structure.
    inline double Evaluator::pawn_structure(Board& board)
    {
        double eval = 0.0;

        Bitboard pawns_w = board.p.bitboard_of(WHITE_PAWN);
        Bitboard pawns_b = board.p.bitboard_of(BLACK_PAWN);

        // doubled pawns
        for (int file = 0; file < 8; file++)
        {
            // white side
            int pawns_on_file = pop_count(pawns_w & MASK_FILE[file]);
            if (pawns_on_file >= 2)
                eval += 0.5 * doubled_pawn_pen[file];

            // black side
            pawns_on_file = pop_count(pawns_b & MASK_FILE[file]);
            if (pawns_on_file >= 2)
                eval -= 0.5 * doubled_pawn_pen[file];
        }

        // passed pawns
        Bitboard pawns_f, front_s;
        Square pawn_s, last_s;
        for (int file = 0; file < 8; file++)
        {
            // white side
            pawns_f = (pawns_w & MASK_FILE[file]);
            if (pawns_f)
            {
                // for the white side we need the pawn that is on the highest rank
                while (pawns_f)
                    pawn_s = pop_lsb(&pawns_f);
                last_s = create_square(static_cast<File>(file), RANK8);
                front_s = SQUARES_BETWEEN_BB[pawn_s][last_s] | SQUARE_BB[last_s];

                // check if the path ahead is blocked or attacked by enemy pawns
                if (front_s & pawns_b)
                    continue;

                Bitboard attacking_p = 0ULL;
                Square s;
                while (front_s)
                {
                    s = pop_lsb(&front_s);
                    attacking_p |= pawn_attacks<WHITE>(s);
                }

                if (attacking_p & pawns_b)
                    continue;

                // we have a passed pawn, which is ...
                int rank = rank_of(pawn_s) - 1;
                double s_diff;
                if (can_attack(WT[pawn_s], PAWN) && can_attack(WT[pawn_s + NORTH], PAWN))
                {
                    // ... supported
                    s_diff = (passed_pawn_hi_supp[rank] - passed_pawn_lo_supp[rank]) / 8.0;
                    eval += passed_pawn_hi_supp[rank] - s_diff * pop_count(pawns_b);
                } else
                {
                    // ... not supported
                    s_diff = (passed_pawn_hi_nsupp[rank] - passed_pawn_lo_nsupp[rank]) / 8.0;
                    eval += passed_pawn_hi_nsupp[rank] - s_diff * pop_count(pawns_b);
                }
            }

            // black side
            pawns_f = (pawns_b & MASK_FILE[file]);
            if (pawns_f)
            {
                // for the black side we need the pawn that is on the lowest rank
                pawn_s = pop_lsb(&pawns_f);
                last_s = create_square(static_cast<File>(file), RANK1);
                front_s = SQUARES_BETWEEN_BB[pawn_s][last_s] | SQUARE_BB[last_s];

                // check if the path ahead is blocked or attacked by enemy pawns
                if (front_s & pawns_w)
                    continue;

                Bitboard attacking_p = 0ULL;
                Square s;
                while (front_s)
                {
                    s = pop_lsb(&front_s);
                    attacking_p |= pawn_attacks<BLACK>(s);
                }

                if (attacking_p & pawns_w)
                    continue;

                // we have a passed pawn, which is ...
                int rank = relative_rank<BLACK>(rank_of(pawn_s)) - 1;
                double s_diff;
                if (can_attack(BT[pawn_s], PAWN) && can_attack(BT[pawn_s + SOUTH], PAWN))
                {
                    // ... supported
                    s_diff = (passed_pawn_hi_supp[rank] - passed_pawn_lo_supp[rank]) / 8.0;
                    eval -= passed_pawn_hi_supp[rank] - s_diff * pop_count(pawns_w);
                } else
                {
                    // ... not supported
                    s_diff = (passed_pawn_hi_nsupp[rank] - passed_pawn_lo_nsupp[rank]) / 8.0;
                    eval -= passed_pawn_hi_nsupp[rank] - s_diff * pop_count(pawns_w);
                }
            }
        }

        // isolated pawns
        // white side
        pawns_f = pawns_w;
        while (pawns_f)
        {
            pawn_s = pop_lsb(&pawns_f);

            if (file_of(pawn_s) == AFILE)
                front_s = SQUARE_BB[pawn_s + EAST] | SQUARE_BB[pawn_s + SOUTH_EAST];
            else if (file_of(pawn_s) == HFILE)
                front_s = SQUARE_BB[pawn_s + WEST] | SQUARE_BB[pawn_s + SOUTH_WEST];
            else
                front_s = SQUARE_BB[pawn_s + EAST] | SQUARE_BB[pawn_s + SOUTH_EAST] | SQUARE_BB[pawn_s + WEST] | SQUARE_BB[pawn_s + SOUTH_WEST];

            int supporting_pawns = pop_count(front_s & pawns_w);
            if (supporting_pawns >= 2)
                continue;

            // we have an isolated pawn
            int is_stopped = (SQUARE_BB[pawn_s + NORTH] & pawns_b) ? 1 : 0;
            last_s = create_square(file_of(pawn_s), RANK8);
            int half_open_file = ((SQUARES_BETWEEN_BB[pawn_s][last_s] | SQUARE_BB[last_s]) & pawns_b) ? 0 : 1;

            eval += isolated_pawn_pen[supporting_pawns][is_stopped][half_open_file];
        }

        // black side
        pawns_f = pawns_b;
        while (pawns_f)
        {
            pawn_s = pop_lsb(&pawns_f);

            if (file_of(pawn_s) == AFILE)
                front_s = SQUARE_BB[pawn_s + EAST] | SQUARE_BB[pawn_s + NORTH_EAST];
            else if (file_of(pawn_s) == HFILE)
                front_s = SQUARE_BB[pawn_s + WEST] | SQUARE_BB[pawn_s + NORTH_WEST];
            else
                front_s = SQUARE_BB[pawn_s + EAST] | SQUARE_BB[pawn_s + NORTH_EAST] | SQUARE_BB[pawn_s + WEST] | SQUARE_BB[pawn_s + NORTH_WEST];

            int supporting_pawns = pop_count(front_s & pawns_b);
            if (supporting_pawns >= 2)
                continue;

            // we have an isolated pawn
            int is_stopped = (SQUARE_BB[pawn_s + SOUTH] & pawns_w) ? 1 : 0;
            last_s = create_square(file_of(pawn_s), RANK1);
            int half_open_file = ((SQUARES_BETWEEN_BB[pawn_s][last_s] | SQUARE_BB[last_s]) & pawns_w) ? 0 : 1;

            eval -= isolated_pawn_pen[supporting_pawns][is_stopped][half_open_file];
        }

        return (board.p.turn() == WHITE) ? eval : -eval;
    }

    //Returns the Centipawn score computed using king safety.
    inline double Evaluator::king_safety(Board& board)
    {
        double eval = 0.0;

        Square w_king = bsf(board.p.bitboard_of(WHITE_KING));
        Square b_king = bsf(board.p.bitboard_of(BLACK_KING));

        // king location
        // - the king is vulnerable if he moves further away from his side of the board

        // white side
        eval -= king_square_vuln_w[w_king];

        // black side
        eval += king_square_vuln_b[b_king];

        // pawn shelter
        // - king's file + left file + right file are evaluated to see how advanced are the friendly and the enemy pawns

        Bitboard file, file_w, file_b;
        File k_file;
        int s_w, s_b;

        // white side
        k_file = file_of(w_king);
        file = MASK_FILE[k_file];
        file_w = file & board.p.bitboard_of(WHITE_PAWN);
        file_b = file & board.p.bitboard_of(BLACK_PAWN);
        s_w = (file_w) ? fmin(static_cast<int>(RANK4), static_cast<int>(rank_of(bsf(file_w)))) : 0;
        if (!file_b) s_b = 0;
        while (file_b)
            s_b = RANK8 - fmax(static_cast<int>(RANK5), static_cast<int>(rank_of(pop_lsb(&file_b))));
        eval -= 2.0 * king_struct_vuln[s_b][s_w];

        if (k_file != AFILE)
        {
            file = MASK_FILE[k_file - 1];
            file_w = file & board.p.bitboard_of(WHITE_PAWN);
            file_b = file & board.p.bitboard_of(BLACK_PAWN);
            s_w = (file_w) ? fmin(static_cast<int>(RANK4), static_cast<int>(rank_of(bsf(file_w)))) : 0;
            if (!file_b) s_b = 0;
            while (file_b)
                s_b = RANK8 - fmax(static_cast<int>(RANK5), static_cast<int>(rank_of(pop_lsb(&file_b))));
            eval -= king_struct_vuln[s_b][s_w];
        }

        if (k_file != HFILE)
        {
            file = MASK_FILE[k_file + 1];
            file_w = file & board.p.bitboard_of(WHITE_PAWN);
            file_b = file & board.p.bitboard_of(BLACK_PAWN);
            s_w = (file_w) ? fmin(static_cast<int>(RANK4), static_cast<int>(rank_of(bsf(file_w)))) : 0;
            if (!file_b) s_b = 0;
            while (file_b)
                s_b = RANK8 - fmax(static_cast<int>(RANK5), static_cast<int>(rank_of(pop_lsb(&file_b))));
            eval -= king_struct_vuln[s_b][s_w];
        }

        // black side
        k_file = file_of(b_king);
        file = MASK_FILE[k_file];
        file_w = file & board.p.bitboard_of(WHITE_PAWN);
        file_b = file & board.p.bitboard_of(BLACK_PAWN);
        s_w = (file_w) ? fmin(static_cast<int>(RANK4), static_cast<int>(rank_of(bsf(file_w)))) : 0;
        if (!file_b) s_b = 0;
        while (file_b)
            s_b = RANK8 - fmax(static_cast<int>(RANK5), static_cast<int>(rank_of(pop_lsb(&file_b))));
        eval += 2.0 * king_struct_vuln[s_w][s_b];

        if (k_file != AFILE)
        {
            file = MASK_FILE[k_file - 1];
            file_w = file & board.p.bitboard_of(WHITE_PAWN);
            file_b = file & board.p.bitboard_of(BLACK_PAWN);
            s_w = (file_w) ? fmin(static_cast<int>(RANK4), static_cast<int>(rank_of(bsf(file_w)))) : 0;
            if (!file_b) s_b = 0;
            while (file_b)
                s_b = RANK8 - fmax(static_cast<int>(RANK5), static_cast<int>(rank_of(pop_lsb(&file_b))));
            eval += king_struct_vuln[s_w][s_b];
        }

        if (k_file != HFILE)
        {
            file = MASK_FILE[k_file + 1];
            file_w = file & board.p.bitboard_of(WHITE_PAWN);
            file_b = file & board.p.bitboard_of(BLACK_PAWN);
            s_w = (file_w) ? fmin(static_cast<int>(RANK4), static_cast<int>(rank_of(bsf(file_w)))) : 0;
            if (!file_b) s_b = 0;
            while (file_b)
                s_b = RANK8 - fmax(static_cast<int>(RANK5), static_cast<int>(rank_of(pop_lsb(&file_b))));
            eval += king_struct_vuln[s_w][s_b];
        }

        // king region attacks
        // - inspect the 8 squares around the king
        //      - penalty for empty squares
        //      - penalty for enemy attacks
        // - penalty if king is in check

        // white side
        Bitboard defense = attacks<KING>(w_king, 0ULL);
        Bitboard all = board.p.all_pieces<WHITE>();

        eval -= empty_square_pen * pop_count(defense & !all);

        Square s_d;
        while (defense)
        {
            s_d = pop_lsb(&defense);

            int num_attacks = fmax(0, attacks_num(BT[s_d]) - attacks_num(WT[s_d]));
            if (num_attacks)
            {
                double attack_pen = 0.0;
                for (int piece = PAWN; piece < KING; piece++)
                {
                    if (can_attack(BT[s_d], static_cast<PieceType>(piece)))
                        attack_pen += material_value[piece];
                }
                eval -= num_attacks * attack_pen;
            }
        }

        if (BT[w_king])
            eval -= check_pen;

        // black side
        defense = attacks<KING>(b_king, 0ULL);
        all = board.p.all_pieces<WHITE>();

        eval += empty_square_pen * pop_count(defense & !all);

        while (defense)
        {
            s_d = pop_lsb(&defense);

            int num_attacks = fmax(0, attacks_num(WT[s_d]) - attacks_num(BT[s_d]));
            if (num_attacks)
            {
                double attack_pen = 0.0;
                for (int piece = PAWN; piece < KING; piece++)
                {
                    if (can_attack(WT[s_d], static_cast<PieceType>(piece)))
                        attack_pen += material_value[piece];
                }
                eval += num_attacks * attack_pen;
            }
        }

        if (WT[b_king])
            eval += check_pen;

        // castling rights
        // - bonus for not loosing castling rights

        // white side
        bool ks = board.p.has_kingside_castling_rights(WHITE);
        bool qs = board.p.has_queenside_castling_rights(WHITE);
        if (ks && qs)
            eval += full_castling_bonus;
        else if (ks)
            eval += ks_castling_bonus;
        else if (qs)
            eval += qs_castling_bonus;

        // black side
        ks = board.p.has_kingside_castling_rights(BLACK);
        qs = board.p.has_queenside_castling_rights(BLACK);
        if (ks && qs)
            eval -= full_castling_bonus;
        else if (ks)
            eval -= ks_castling_bonus;
        else if (qs)
            eval -= qs_castling_bonus;

        return (board.p.turn() == WHITE) ? eval : -eval;
    }

    //Returns the Centipawn score computed using piece placement.
    inline double Evaluator::piece_placement(Board& board)
    {
        double eval = 0.0;

        Bitboard pieces;
        Square sq;
        int w_king_zone = king_zone_w[bsf(board.p.bitboard_of(WHITE_KING))];
        int b_king_zone = king_zone_b[bsf(board.p.bitboard_of(BLACK_KING))];

        // pawn
        // 1. square score

        // white side
        pieces = board.p.bitboard_of(WHITE_PAWN);
        while (pieces)
        {
            sq = pop_lsb(&pieces);
            eval += pawn_square_score_w[sq];
        }

        // black side
        pieces = board.p.bitboard_of(BLACK_PAWN);
        while (pieces)
        {
            sq = pop_lsb(&pieces);
            eval -= pawn_square_score_b[sq];
        }

        // knight
        // 1. square score
        // 2. king distance score
        // 3. bonus if on strong square, with higher bonus is the strong square is in the center

        // white side
        pieces = board.p.bitboard_of(WHITE_KNIGHT);
        while (pieces)
        {
            sq = pop_lsb(&pieces);
            eval += knight_square_score_w[sq] + knight_distance_bonus[diamond_distance_b[b_king_zone][sq]];
            eval += (SQUARE_BB[sq] & black_side && !can_attack(BT[sq], PAWN)) ? ((SQUARE_BB[sq] & center_squares) ? strong_cent_sq_bonus : strong_sq_bonus) : 0.0;
        }

        // black side
        pieces = board.p.bitboard_of(BLACK_KNIGHT);
        while (pieces)
        {
            sq = pop_lsb(&pieces);
            eval -= knight_square_score_b[sq] + knight_distance_bonus[diamond_distance_w[w_king_zone][sq]];
            eval -= (SQUARE_BB[sq] & white_side && !can_attack(WT[sq], PAWN)) ? ((SQUARE_BB[sq] & center_squares) ? strong_cent_sq_bonus : strong_sq_bonus) : 0.0;
        }

        // bishop
        // 1. square score
        // 2. penalty for being on the same diagonal as friendly blocked pawn
        // 3. bonus for being on the same diagonal as a weak enemy pawn
        // 4. same bonus for strong squares as knight

        Bitboard attacked, friendly, enemy;

        // white side
        pieces = board.p.bitboard_of(WHITE_BISHOP);
        while (pieces)
        {
            sq = pop_lsb(&pieces);
            eval += bishop_square_score_w[sq];
            eval += (SQUARE_BB[sq] & black_side && !can_attack(BT[sq], PAWN)) ? ((SQUARE_BB[sq] & center_squares) ? strong_cent_sq_bonus : strong_sq_bonus) : 0.0;

            attacked = attacks<BISHOP>(sq, board.p.all_pieces<WHITE>() | board.p.all_pieces<BLACK>());
            friendly = attacked & board.p.bitboard_of(WHITE_PAWN);
            enemy = attacked & board.p.bitboard_of(BLACK_PAWN);

            while (friendly)
            {
                sq = pop_lsb(&friendly);
                if (SQUARE_BB[sq + NORTH] & board.p.all_pieces<WHITE>() | board.p.all_pieces<BLACK>())
                    eval += bishop_diag_penalty;
            }

            while (enemy)
            {
                sq = pop_lsb(&enemy);
                if (attacks_num(BT[sq]) == 0)
                    eval += bishop_diag_bonus;
            }
        }

        // black side
        pieces = board.p.bitboard_of(BLACK_BISHOP);
        while (pieces)
        {
            sq = pop_lsb(&pieces);
            eval -= bishop_square_score_b[sq];
            eval -= (SQUARE_BB[sq] & white_side && !can_attack(WT[sq], PAWN)) ? ((SQUARE_BB[sq] & center_squares) ? strong_cent_sq_bonus : strong_sq_bonus) : 0.0;

            attacked = attacks<BISHOP>(sq, board.p.all_pieces<WHITE>() | board.p.all_pieces<BLACK>());
            friendly = attacked & board.p.bitboard_of(BLACK_PAWN);
            enemy = attacked & board.p.bitboard_of(WHITE_PAWN);

            while (friendly)
            {
                sq = pop_lsb(&friendly);
                if (SQUARE_BB[sq + SOUTH] & board.p.all_pieces<WHITE>() | board.p.all_pieces<BLACK>())
                    eval -= bishop_diag_penalty;
            }

            while (enemy)
            {
                sq = pop_lsb(&enemy);
                if (attacks_num(WT[sq]) == 0)
                    eval -= bishop_diag_bonus;
            }
        }

        // rook
        // 1. square score
        // 2. bonus for open or half-open files
        // 3. bonus for files with weak enemy pawns
        // 4. king distance score
        // 5. bonus for strong squares, same as for knights

        // white side
        pieces = board.p.bitboard_of(WHITE_ROOK);
        while (pieces)
        {
            sq = pop_lsb(&pieces);
            eval += rook_square_score_w[sq] + rook_distance_bonus[cross_distance_b[b_king_zone][sq]];
            eval += (SQUARE_BB[sq] & black_side && !can_attack(BT[sq], PAWN)) ? ((SQUARE_BB[sq] & center_squares) ? strong_cent_sq_bonus : strong_sq_bonus) : 0.0;

            enemy = MASK_FILE[file_of(sq)] & board.p.bitboard_of(BLACK_PAWN);
            friendly = MASK_FILE[file_of(sq)] & board.p.bitboard_of(WHITE_PAWN);

            if (enemy && friendly)
            {
                // closed file + check if enemy pawn is weak
                if (attacks_num(BT[bsf(enemy)]) == 0)
                    eval += rook_weak_pawn_bonus;
            } else if (enemy)
            {
                // half-open file + check if enemy pawn is weak
                eval += rook_half_file_bonus;
                if (attacks_num(BT[bsf(enemy)]) == 0)
                    eval += rook_weak_pawn_bonus;
            } else if (friendly)
            {
                // half-open file
                eval += rook_half_file_bonus;
            } else
            {
                // open file
                eval += rook_open_file_bonus;
            }
        }

        // black side
        pieces = board.p.bitboard_of(BLACK_ROOK);
        while (pieces)
        {
            sq = pop_lsb(&pieces);
            eval -= rook_square_score_b[sq] + rook_distance_bonus[cross_distance_w[w_king_zone][sq]];
            eval -= (SQUARE_BB[sq] & white_side && !can_attack(WT[sq], PAWN)) ? ((SQUARE_BB[sq] & center_squares) ? strong_cent_sq_bonus : strong_sq_bonus) : 0.0;

            enemy = MASK_FILE[file_of(sq)] & board.p.bitboard_of(WHITE_PAWN);
            friendly = MASK_FILE[file_of(sq)] & board.p.bitboard_of(BLACK_PAWN);

            if (enemy && friendly)
            {
                // closed file + check if enemy pawn is weak
                if (attacks_num(WT[bsf(enemy)]) == 0)
                    eval -= rook_weak_pawn_bonus;
            } else if (enemy)
            {
                // half-open file + check if enemy pawn is weak
                eval -= rook_half_file_bonus;
                if (attacks_num(WT[bsf(enemy)]) == 0)
                    eval -= rook_weak_pawn_bonus;
            } else if (friendly)
            {
                // half-open file
                eval -= rook_half_file_bonus;
            } else
            {
                // open file
                eval -= rook_open_file_bonus;
            }
        }

        // queen
        // 1. square score
        // 2. king distance score
        // 3. strong square bonus, same as others

        // white side
        pieces = board.p.bitboard_of(WHITE_QUEEN);
        while (pieces)
        {
            sq = pop_lsb(&pieces);
            eval += queen_square_score_w[sq] + queen_distance_bonus[diamond_distance_b[b_king_zone][sq]];
            eval += (SQUARE_BB[sq] & black_side && !can_attack(BT[sq], PAWN)) ? ((SQUARE_BB[sq] & center_squares) ? strong_cent_sq_bonus : strong_sq_bonus) : 0.0;
        }

        // black side
        pieces = board.p.bitboard_of(BLACK_QUEEN);
        while (pieces)
        {
            sq = pop_lsb(&pieces);
            eval -= queen_square_score_b[sq] + queen_distance_bonus[diamond_distance_w[w_king_zone][sq]];
            eval -= (SQUARE_BB[sq] & white_side && !can_attack(WT[sq], PAWN)) ? ((SQUARE_BB[sq] & center_squares) ? strong_cent_sq_bonus : strong_sq_bonus) : 0.0;
        }

        // king
        // 1. square score

        // white side
        eval += king_square_score_w[bsf(board.p.bitboard_of(WHITE_KING))];

        // black side
        eval -= king_square_score_b[bsf(board.p.bitboard_of(BLACK_KING))];

        return (board.p.turn() == WHITE) ? eval : -eval;
    }

    //Returns the Centipawn score computed using board control.
    inline double Evaluator::board_control(Board& board)
    {
        double eval = 0.0;

        // weight controled squares according to the area
        Bitboard all_w = board.p.all_pieces<WHITE>();
        Bitboard all_b = board.p.all_pieces<BLACK>();
        Square s;

        // white side
        while (all_w)
        {
            s = pop_lsb(&all_w);
            eval += control_bonus_w[s];
        }

        // black side
        while (all_b)
        {
            s = pop_lsb(&all_b);
            eval -= control_bonus_b[s];
        }

        // estimate mobility by analysing the attack/move pressure each side has on the board
        int num;
        for (int i = 0; i < NSQUARES; i++)
        {
            if (num = attacks_num(WT[i]))
                eval += static_cast<double>(num) / 100.0;
            if (num = attacks_num(BT[i]))
                eval -= static_cast<double>(num) / 100.0;
        }

        return (board.p.turn() == WHITE) ? eval : -eval;
    }

    //Returns the Centipawn score computed by the configured evaluation function.
    inline double Evaluator::eval(Board& board)
    {
        update_tables(board);
        double cp = 0.0;

        if (has_eval(material_mask))
            cp += material(board);
        if (has_eval(pawn_structure_mask))
            cp += pawn_structure(board);
        if (has_eval(king_safety_mask))
            cp += king_safety(board);
        if (has_eval(piece_placement_mask))
            cp += piece_placement(board);
        if (has_eval(board_control_mask))
            cp += board_control(board);

        return cp_to_q(cp);
    }

    //////////////////////////////////////////////////////////////////////////////////
    ///////////////////////// MCTS STRATEGIES IMPLEMENTATION /////////////////////////
    //////////////////////////////////////////////////////////////////////////////////

    //Strategy to choose the best move to make.

    //Chooses move with the highest number of visits.
    inline Move& best_move_nvisits(move_vector<Move>& moves)
    {
        long most_visits = 0;
        std::vector<int> best_moves;
        int index = 0;
        for (Move& move : moves)
        {
            if (move.n_visits > most_visits)
            {
                most_visits = move.n_visits;
                best_moves.clear();
                best_moves.push_back(index);
            } 
            else if (most_visits == move.n_visits)
            {
                best_moves.push_back(index);
            }
            index++;
        }

        if (best_moves.size() == 1)
        {
            return moves[best_moves.front()];
        } 
        else
        {
            // if multiple moves share same max value, pick a random move
            srand(time(NULL));
            int index = rand() % best_moves.size();
            return moves[best_moves[index]];
        }
    }

    //Chooses move by also taking Q-values into account.
    inline Move& best_move_qvalue(move_vector<Move>& moves)
    {
        //Find the most visited move of the current state.
        long visit_thresh = 0;
        for (Move& move : moves)
        {
            if (move.n_visits > visit_thresh)
                visit_thresh = move.n_visits;
        }

        //Calculate Q-value threshold.
        double Q_thresh = Q_thresh_max - exp(static_cast<double>(-moves.n_visits) / static_cast<double>(Q_thresh_base)) * (Q_thresh_max - Q_thresh_init);
        visit_thresh = static_cast<long>(visit_thresh * Q_thresh);

        // calculate the best move
        std::vector<int> best_moves;
        int index = 0;
        double best_Q = 0.0;
        for (Move& move : moves)
        {
            // scale Q to [0, 1]
            double q = (move.Q_value + 1.0) / 2.0;

            // set Q values with Nsm < Q_thresh * max(Nsm) to 0
            if (move.n_visits < visit_thresh)
                q = 0.0;

            // combine Nsm and Q
            double move_eval = (1.0 - Q_factor) * (static_cast<double>(move.n_visits) / static_cast<double>(moves.n_visits)) + Q_factor * q;

            if (move_eval > best_Q)
            {
                best_Q = move_eval;
                best_moves.clear();
                best_moves.push_back(index);
            } 
            else if (move_eval == best_Q)
            {
                best_moves.push_back(index);
            }
            index++;
        }

        if (best_moves.size() == 1)
        {
            return moves[best_moves.front()];
        } else
        {
            // if multiple moves share same max value, pick a random move
            srand(time(NULL));
            int index = rand() % best_moves.size();
            return moves[best_moves[index]];
        }
    }



    //Strategy to choose the next move to expand during a MCTS simulation.

    //Chooses move according to the PUCT algorithm.
    inline Move& move_to_expand_default(move_vector<Move>& moves)
    {
        // calculate U-values and select best move
        int best_move = 0;
        double best_U = -std::numeric_limits<double>::infinity();
        double cpuct = log(static_cast<double>(moves.n_visits + cpuct_base + 1L) / static_cast<double>(cpuct_base)) + cpuct_init;

        int index = 0;
        for (Move& move : moves)
        {
            double u;
            if (move.n_visits)
            {
                // move was already explored
                u = move.Q_value + cpuct * move.policy * sqrt(static_cast<double>(moves.n_visits)) / (1.0 + static_cast<double>(move.n_visits));
            } else
            {
                // move was not yet explored
                u = Q_init + cpuct * move.policy * sqrt(static_cast<double>(moves.n_visits) + EPS);
            }

            if (u > best_U)
            {
                best_U = u;
                best_move = index;
            }
            index++;
        }
        return moves[best_move];
    }

    //Chooses move as proposed in CrazyAra. Encourages exploration. 
    inline Move& move_to_expand_inc(move_vector<Move>& moves)
    {
        // calculate U-values and select best move
        int best_move = 0;
        double best_U = -std::numeric_limits<double>::infinity();
        double cpuct = log(static_cast<double>(moves.n_visits + cpuct_base + 1) / static_cast<double>(cpuct_base)) + cpuct_init;
        double u_divisor = u_min - exp(static_cast<double>(-moves.n_visits) / static_cast<double>(u_base)) * (u_min - u_init);

        int index = 0;
        for (const Move& move : moves)
        {
            double u;
            if (move.n_visits)
            {
                // move was already explored
                u = move.Q_value + cpuct * move.policy * sqrt(static_cast<double>(moves.n_visits)) / (u_divisor + static_cast<double>(move.n_visits));
            } else
            {
                // move was not yet explored
                u = Q_init + cpuct * move.policy * sqrt(static_cast<double>(moves.n_visits)) / (u_divisor + static_cast<double>(move.n_visits));
            }

            if (u > best_U)
            {
                best_U = u;
                best_move = index;
            }
            index++;
        }
        return moves[best_move];
    }



    //Strategy to calculate Q-values during backpropagation.

    //Calculates Q-values according to the PUCT algorithm.
    inline double backprop_nvisits_qvalue(const Move& move, const double& v)
    {
        return (move.n_visits * move.Q_value + v) / (move.n_visits + 1.0);
    }

    //Calculates Q-values as a Simple Moving Average.
    inline double backprop_sma(const Move& move, const double& v)
    {
        return (move.Q_value + v) / 2.0;
    }



    //Strategy for enhancing the prior probability of legal moves returned by the neural network.

    //Add values from a Dirichlet distribution.
    inline void enhance_policy_dirichlet(MCTS* mcts, Board& board, move_vector<Move>& moves)
    {
        std::vector<double> noise = mcts->dirichlet.get_noise();
        double sum_policy = 0.0;
        for (Move& move : moves)
        {
            // apply dirichlet
            move.policy += dirichlet_factor * noise[move.hash()];

            if (move.policy < 0.0)
                move.policy = 0.0;

            // calculate sum for normalization
            sum_policy += move.policy;
        }

        // renormalize
        for (Move& move : moves)
            move.policy /= sum_policy;
    }

    //Improves probabilities for moves that result in a check.
    inline void enhance_policy_checking_moves(MCTS* mcts, Board& board, move_vector<Move>& moves)
    {
        double max_policy = 0.0;
        for (const Move& move : moves)
        {
            // find maximum policy value
            if (move.policy > max_policy)
                max_policy = move.policy;
        }

        // enhance the probability for all checking moves where P(s, a) < check_thresh
        bool enhanced = false;
        double sum_policy = 0.0;
        for (Move& move : moves)
        {
            if (move.policy < check_thresh && board.gives_check(move))
            {
                move.policy += max_policy * check_factor;
                enhanced = true;
            }
            sum_policy += move.policy;

        }

        // renormalize if needed
        if (enhanced)
        {
            for (Move& move : moves)
                move.policy /= sum_policy;
        }
    }

    //Improves probabilities for moves that result in a fork.
    inline void enhance_policy_forking_moves(MCTS* mcts, Board& board, move_vector<Move>& moves)
    {
        double max_policy = 0.0;
        for (const Move& move : moves)
        {
            // find maximum policy value
            if (move.policy > max_policy)
                max_policy = move.policy;
        }

        // enhance the probability for all forking moves where P(s, a) < check_thresh
        bool enhanced = false;
        double sum_policy = 0.0;
        for (Move& move : moves)
        {
            if (move.policy < check_thresh && board.gives_fork(move))
            {
                move.policy += max_policy * check_factor;
                enhanced = true;
            }
            sum_policy += move.policy;
        }

        // renormalize if needed
        if (enhanced)
        {
            for (Move& move : moves)
                move.policy /= sum_policy;
        }
    }

    //Improves probabilities for dropping moves that benefit the player.
    inline void enhance_policy_dropping_moves(MCTS* mcts, Board& board, move_vector<Move>& moves)
    {
        double max_policy = 0.0;
        for (const Move& move : moves)
        {
            // find maximum policy value
            if (move.policy > max_policy)
                max_policy = move.policy;
        }

        // enhance the probability for all dropping moves where P(s, a) < check_thresh
        bool enhanced = false;
        double sum_policy = 0.0;
        for (Move& move : moves)
        {
            if (move.policy < check_thresh && move.flags() >= DROP_PAWN && move.flags() <= DROP_QUEEN)
            {
                move.policy += max_policy * board.eval_drop(move);
                enhanced = true;
            }
            sum_policy += move.policy;
        }

        // renormalize if needed
        if (enhanced)
        {
            for (Move& move : moves)
                move.policy /= sum_policy;
        }
    }

    //Improves probabilities for moves that result in a capture.
    inline void enhance_policy_capturing_moves(MCTS* mcts, Board& board, move_vector<Move>& moves)
    {
        double max_policy = 0.0;
        for (const Move& move : moves)
        {
            // find maximum policy value
            if (move.policy > max_policy)
                max_policy = move.policy;
        }

        // enhance the probability for all checking moves where P(s, a) < check_thresh
        bool enhanced = false;
        double sum_policy = 0.0;
        for (Move& move : moves)
        {
            if (move.policy < check_thresh && move.flags() == CAPTURE)
            {
                move.policy += max_policy * check_factor;
                enhanced = true;
            }
            sum_policy += move.policy;

        }

        // renormalize if needed
        if (enhanced)
        {
            for (Move& move : moves)
                move.policy /= sum_policy;
        }
    }
}

#endif

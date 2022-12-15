#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <chrono>
#include <ctime>
#include "utils.h"
#include "crazyzero.h"
#include "cppflow/cppflow.h"
#include "uci/uci.h"

#define SELF_PLAY_LOG_NAME "./results/log"
#define SELF_PLAY_CONFIGS_FILE_PATH "./self_play_configs.txt"
#define GAMES_TO_PLAY 100
#define MAX_SIMULATIONS 1600
#define MIN_SIMULATIONS 100
#define SIMULATIONS_INCREASE 200

using namespace crazyzero;

void play_games(Board& board, MCTS& player1, MCTS& player2, int num_games, const Color p1_start, const Color p2_start, const int init_p1_wins, const int init_p2_wins, const int init_draws, PGN_writer& pgn_log, std::ofstream& result_log)
{
	Color p1 = p1_start;
	Color p2 = p2_start;
	Color turn = WHITE;

	int p1_wins = init_p1_wins;
	int p2_wins = init_p2_wins;
	int draws = init_draws;

	std::ostringstream p1_name;
	p1_name << player1.config << " " << player1.num_sims;

	std::ostringstream p2_name;
	p2_name << player2.config << " " << player2.num_sims;

	int starting_round = 0;
	if (num_games < GAMES_TO_PLAY)
	{
		starting_round = GAMES_TO_PLAY - num_games;
		num_games = GAMES_TO_PLAY;
	}

	for (int i = starting_round; i < num_games; i++)
	{
		player1.player = p1;
		player2.player = p2;

		if (p1 == WHITE)
			pgn_log.new_game("Self-play", i + 1, p1_name.str(), p2_name.str());
		else
			pgn_log.new_game("Self-play", i + 1, p2_name.str(), p1_name.str());

		int move_num = 0;
		std::cout << "P1 wins: " << p1_wins << ", P2 wins: " << p2_wins << ", draws: " << draws << " | playing game " << i + 1 << " of " << num_games << ", move " << move_num << "       \r";
		double end_score = 0.0;
		while (end_score == 0.0)
		{
			Move move;
			if (turn == player1.player)
				move = player1.best_move(board);
			else
				move = player2.best_move(board);

			pgn_log.add_move(board.san(move));
			board.push(move);
			move_num++;
			std::cout << "P1 wins: " << p1_wins << ", P2 wins: " << p2_wins << ", draws: " << draws << " | playing game " << i + 1 << " of " << num_games << ", move " << move_num << "       \r";

			turn = ~turn;
			end_score = board.end_score(turn);
		}

		if (end_score < 0.0)
		{
			if (turn == WHITE)
			{
				pgn_log.flush(BLACK);
				if (p1 == BLACK)
					p1_wins++;
				else
					p2_wins++;
			} else
			{
				pgn_log.flush(WHITE);
				if (p1 == WHITE)
					p1_wins++;
				else
					p2_wins++;
			}
		} 
		else
		{
			pgn_log.flush(NO_COLOR);
			draws++;
		}

		//std::cout << "\n";

		turn = WHITE;
		p1 = ~p1;
		p2 = ~p2;
		board.reset();
		player1.reset();
		player2.reset();
	}

	std::cout << "Final results -> P1 wins: " << p1_wins << ", P2 wins: " << p2_wins << ", draws: " << draws << "                                 \n";

	crazyzero::Elo elo(p1_wins, p2_wins, draws);

	std::cout << "ELO difference: " << elo.diff() << " +/- " << elo.error_margin() << "\n";

	result_log << "Self-play session: P1 [" << p1_name.str() << " simulations] vs. P2 [" << p2_name.str() << " simulations]\n";
	result_log << "P1 wins: " << p1_wins << "\n";
	result_log << "P2 wins: " << p2_wins << "\n";
	result_log << "Draws: " << draws << "\n";
	result_log << "ELO difference: " << elo.diff() << " +/- " << elo.error_margin() << "\n";
	result_log.flush();
}

void init_self_play(std::vector<SelfPlayConfig>& configs)
{
	std::ifstream file(SELF_PLAY_CONFIGS_FILE_PATH);

	if (file.is_open())
	{
		std::string line;
		std::getline(file, line);

		//First line defines the range of simulations amounts to test.
		// min_simulations-max_simulations-increment
		size_t split_point = line.find('-');
		int min_sims = std::stoi(line.substr(0, split_point));
		line = line.substr(split_point + 1);
		split_point = line.find('-');
		int max_sims = std::stoi(line.substr(0, split_point));
		int inc = std::stoi(line.substr(split_point + 1));

		//All other lines define the configurations to be tested.
		std::getline(file, line);
		while (file)
		{
			for (int p2_sims = min_sims; p2_sims < max_sims; p2_sims += inc)
			{
				//Check if results for this configuration already exist and how many there are.
				int p1_sims = p2_sims + inc;
				std::ostringstream file_name;
				file_name << SELF_PLAY_LOG_NAME << "_" << line << "_" << p2_sims << "-" << p1_sims << ".pgn";

				SelfPlayConfig config;
				config.config = parse_mod_mask(line);
				config.p1_sims = p1_sims;
				config.p2_sims = p2_sims;

				PGN_reader pgn_file(file_name.str());
				std::string player1, player2;
				int p1_wins = 0, p2_wins = 0, draws = 0;
				bool init_players = true;
				while (pgn_file.read_game())
				{
					if (init_players)
					{
						init_players = false;
						player1 = pgn_file.white;
						player2 = pgn_file.black;
					}

					if (pgn_file.result == WHITE)
					{
						if (player1 == pgn_file.white)
							p1_wins++;
						else
							p2_wins++;
					} else if (pgn_file.result == BLACK)
					{
						if (player1 == pgn_file.black)
							p1_wins++;
						else
							p2_wins++;
					} else
					{
						draws++;
					}
				}

				int all_games = p1_wins + p2_wins + draws;
				if (all_games >= GAMES_TO_PLAY)
					continue;

				config.num_games = GAMES_TO_PLAY - all_games;
				config.p1_wins = p1_wins;
				config.p2_wins = p2_wins;
				config.draws = draws;

				if (all_games % 2)
				{
					config.p1_start = BLACK;
					config.p2_start = WHITE;
				}
				else
				{
					config.p1_start = WHITE;
					config.p2_start = BLACK;
				}

				configs.push_back(config);
			}
			std::getline(file, line);
		}
	}
}

void self_play()
{
	std::cout << "[SELF-PLAY MODE]\n";

	std::vector<SelfPlayConfig> configs;
	init_self_play(configs);

	Board board;
	MCTS player1, player2;
	player1.init(board);
	player2.init(player1.nnet.model);

	player1.time_control = false;
	player2.time_control = false;
	player1.use_openings = false;
	player2.use_openings = false;
	player1.use_mate_search = false;
	player2.use_mate_search = false;

	for (const SelfPlayConfig& config : configs)
	{
		std::cout << "[CONFIGURATION " << config.config << " | P1: " << config.p1_sims << " simulations, P2: " << config.p2_sims << " simulations]\n";

		player1.set_config(config.config);
		player2.set_config(config.config);
		player1.num_sims = config.p1_sims;
		player2.num_sims = config.p2_sims;

		std::ostringstream pgn_log_name;
		pgn_log_name << SELF_PLAY_LOG_NAME << "_" << config.config << "_" << config.p2_sims << "-" << config.p1_sims << ".pgn";

		std::ostringstream result_log_name;
		result_log_name << SELF_PLAY_LOG_NAME << "_" << config.config << "_" << config.p2_sims << "-" << config.p1_sims << ".txt";

		PGN_writer pgn_log(pgn_log_name.str());
		std::ofstream result_log(result_log_name.str());

		play_games(board, player1, player2, config.num_games, config.p1_start, config.p2_start, config.p1_wins, config.p2_wins, config.draws, pgn_log, result_log);
	}

	player2.nnet.model = nullptr;
}

int main(int argc, char* argv[]) 
{
	_putenv("TF_CPP_MIN_LOG_LEVEL=3");

	initialise_all_databases();
	zobrist::initialise_zobrist_keys();
	initialise_eval_tables();

	std::cout << "CrazyZero 2.2 by Anei Makovec\n";

	/*
	//Testing MCTS simulations.
	Board b;
	MCTS mcts;
	mcts.time_control = false;
	mcts.num_sims = 1000;
	mcts.player = WHITE;
	mcts.init(b);

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	const Move& move = mcts.best_move(b);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cout << "Best move: " << move << "\n";
	std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0 << " seconds\n";
	*/

	//self_play();

	//return 0;

	if (argc > 1 && std::string(argv[1]) == "--self-play") {
		// perform self-play test
		self_play();
	} else {
		Board board;
		MCTS mcts;
		uci uci;
		bool debug_mode = true;

		// register callbacks to the messages from the UI and respond appropriately.
		uci.receive_uci.connect([&]() {
			uci.send_id("CrazyZero 2.1", "Anei Makovec");
			uci.send_option_combo_box("UCI_Variant", "crazyhouse", { "crazyhouse" });
			uci.send_option_combo_box("TimeControl", "Default", { "Default", "Variable", "None" });
			uci.send_option_spin_wheel("Simulations/Move", 100, 1, 10000);
			uci.send_option_combo_box("BestMoveStrategy", "Default", { "Default", "Q-value" });
			uci.send_option_combo_box("NodeExpansionStrategy", "Default", { "Default", "Exploration" });
			uci.send_option_combo_box("BackpropStrategy", "Default", { "Default", "SMA" });
			uci.send_option_check_box("PE_Dirichlet", true);
			uci.send_option_check_box("PE_CheckingMoves", false);
			uci.send_option_check_box("PE_ForkingMoves", false);
			uci.send_option_check_box("PE_DroppingMoves", false);
			uci.send_option_check_box("PE_CapturingMoves", false);
			uci.send_option_check_box("Eval_Material", false);
			uci.send_option_check_box("Eval_PawnStructure", false);
			uci.send_option_check_box("Eval_KingSafety", false);
			uci.send_option_check_box("Eval_PiecePlacement", false);
			uci.send_option_check_box("Eval_BoardControl", false);
			uci.send_option_check_box("TimeSavingConfig", false);
			uci.send_option_check_box("TS_PE_Dirichlet", true);
			uci.send_option_check_box("TS_PE_CheckingMoves", false);
			uci.send_option_check_box("TS_PE_ForkingMoves", false);
			uci.send_option_check_box("TS_PE_DroppingMoves", false);
			uci.send_option_check_box("TS_PE_CapturingMoves", false);
			uci.send_option_check_box("TS_Eval_Material", false);
			uci.send_option_check_box("TS_Eval_PawnStructure", false);
			uci.send_option_check_box("TS_Eval_KingSafety", false);
			uci.send_option_check_box("TS_Eval_PiecePlacement", false);
			uci.send_option_check_box("TS_Eval_BoardControl", false);
			uci.send_uci_ok();
			});
		uci.receive_set_option.connect([&](const std::string& name, const std::string& value) {
			/*
			if (name == "UCI_Variant") {
				// pass
				return;
			} else if (name == "TimeControl") {
				if (value == "Default") {
					mcts.config.time_control = true;
					mcts.config.variable_time_control = false;
				} else if (value == "Variable") {
					mcts.config.time_control = false;
					mcts.config.variable_time_control = true;
				} else if (value == "None") {
					mcts.config.time_control = false;
					mcts.config.variable_time_control = false;
				}
			} else if (name == "Simulations/Move") {
				int sims = stoi(value);
				if (sims >= 1 && sims <= 10000)
					mcts.config.num_sims = sims;
			} else if (name == "BestMoveStrategy") {
				if (value == "Default")
					mcts.config.best_move_strategy = BestMoveStrat::Default;
				else if (value == "Q-value")
					mcts.config.best_move_strategy = BestMoveStrat::Q_value;
			} else if (name == "NodeExpansionStrategy") {
				if (value == "Default")
					mcts.config.node_expansion_strategy = NodeExpansionStrat::Default;
				else if (value == "Exploration")
					mcts.config.node_expansion_strategy = NodeExpansionStrat::Exploration;
			} else if (name == "BackpropStrategy") {
				if (value == "Default")
					mcts.config.backprop_strategy = BackpropStrat::Default;
				else if (value == "SMA")
					mcts.config.backprop_strategy = BackpropStrat::SMA;
			} else if (name == "PE_Dirichlet") {
				if (value == "true")
					mcts.config.policy_strategies.insert(PolicyEnhancementStrat::Dirichlet);
				else
					mcts.config.policy_strategies.erase(PolicyEnhancementStrat::Dirichlet);
			} else if (name == "PE_CheckingMoves") {
				if (value == "true")
					mcts.config.policy_strategies.insert(PolicyEnhancementStrat::CheckingMoves);
				else
					mcts.config.policy_strategies.erase(PolicyEnhancementStrat::CheckingMoves);
			} else if (name == "PE_ForkingMoves") {
				if (value == "true")
					mcts.config.policy_strategies.insert(PolicyEnhancementStrat::ForkingMoves);
				else
					mcts.config.policy_strategies.erase(PolicyEnhancementStrat::ForkingMoves);
			} else if (name == "PE_DroppingMoves") {
				if (value == "true")
					mcts.config.policy_strategies.insert(PolicyEnhancementStrat::DroppingMoves);
				else
					mcts.config.policy_strategies.erase(PolicyEnhancementStrat::DroppingMoves);
			} else if (name == "PE_CapturingMoves") {
				if (value == "true")
					mcts.config.policy_strategies.insert(PolicyEnhancementStrat::CapturingMoves);
				else
					mcts.config.policy_strategies.erase(PolicyEnhancementStrat::CapturingMoves);
			} else if (name == "Eval_Material") {
				if (value == "true")
					mcts.config.eval_types |= material_mask;
				else
					mcts.config.eval_types &= ~material_mask;
			} else if (name == "Eval_PawnStructure") {
				if (value == "true")
					mcts.config.eval_types |= pawn_structure_mask;
				else
					mcts.config.eval_types &= ~pawn_structure_mask;
			} else if (name == "Eval_KingSafety") {
				if (value == "true")
					mcts.config.eval_types |= king_safety_mask;
				else
					mcts.config.eval_types &= ~king_safety_mask;
			} else if (name == "Eval_PiecePlacement") {
				if (value == "true")
					mcts.config.eval_types |= piece_placement_mask;
				else
					mcts.config.eval_types &= ~piece_placement_mask;
			} else if (name == "Eval_BoardControl") {
				if (value == "true")
					mcts.config.eval_types |= board_control_mask;
				else
					mcts.config.eval_types &= ~board_control_mask;
			} else if (name == "TimeSavingConfig") {
				if (value == "true")
					mcts.config.time_saving_config = true;
				else
					mcts.config.time_saving_config = false;
			} else if (name == "TS_PE_Dirichlet") {
				if (value == "true")
					mcts.config.ts_policy_strategies.insert(PolicyEnhancementStrat::Dirichlet);
				else
					mcts.config.ts_policy_strategies.erase(PolicyEnhancementStrat::Dirichlet);
			} else if (name == "TS_PE_CheckingMoves") {
				if (value == "true")
					mcts.config.ts_policy_strategies.insert(PolicyEnhancementStrat::CheckingMoves);
				else
					mcts.config.ts_policy_strategies.erase(PolicyEnhancementStrat::CheckingMoves);
			} else if (name == "TS_PE_ForkingMoves") {
				if (value == "true")
					mcts.config.ts_policy_strategies.insert(PolicyEnhancementStrat::ForkingMoves);
				else
					mcts.config.ts_policy_strategies.erase(PolicyEnhancementStrat::ForkingMoves);
			} else if (name == "TS_PE_DroppingMoves") {
				if (value == "true")
					mcts.config.ts_policy_strategies.insert(PolicyEnhancementStrat::DroppingMoves);
				else
					mcts.config.ts_policy_strategies.erase(PolicyEnhancementStrat::DroppingMoves);
			} else if (name == "TS_PE_CapturingMoves") {
				if (value == "true")
					mcts.config.ts_policy_strategies.insert(PolicyEnhancementStrat::CapturingMoves);
				else
					mcts.config.ts_policy_strategies.erase(PolicyEnhancementStrat::CapturingMoves);
			} else if (name == "TS_Eval_Material") {
				if (value == "true")
					mcts.config.ts_eval_types |= material_mask;
				else
					mcts.config.ts_eval_types &= ~material_mask;
			} else if (name == "TS_Eval_PawnStructure") {
				if (value == "true")
					mcts.config.ts_eval_types |= pawn_structure_mask;
				else
					mcts.config.ts_eval_types &= ~pawn_structure_mask;
			} else if (name == "TS_Eval_KingSafety") {
				if (value == "true")
					mcts.config.ts_eval_types |= king_safety_mask;
				else
					mcts.config.ts_eval_types &= ~king_safety_mask;
			} else if (name == "TS_Eval_PiecePlacement") {
				if (value == "true")
					mcts.config.ts_eval_types |= piece_placement_mask;
				else
					mcts.config.ts_eval_types &= ~piece_placement_mask;
			} else if (name == "TS_Eval_BoardControl") {
				if (value == "true")
					mcts.config.ts_eval_types |= board_control_mask;
				else
					mcts.config.ts_eval_types &= ~board_control_mask;
			} else {
				std::cout << "UCI ERROR: option " << name << " could not be set to value " << value << ".\n";
			}
			mcts.config.changed = true;
			*/
			});
		uci.receive_debug.connect([&](bool on) {
			debug_mode = on;
			});
		uci.receive_is_ready.connect([&]() {
			mcts.init(board);
			uci.send_ready_ok();
			});
		uci.receive_uci_new_game.connect([&]() {
			board.reset();
			mcts.reset();

			/*log << "--------------------------------------------------------------------\n";
			log << "--------------------------------------------------------------------\n";
			log << "--------------------------- NEW GAME -------------------------------\n";
			log << "--------------------------------------------------------------------\n";
			log << "--------------------------------------------------------------------\n";
			log << board.p << "\n";
			log.flush();*/
			});
		uci.receive_position.connect([&](const std::string& fen, const std::vector<std::string>& moves) {
			if (moves.size()) {
				if (moves.size() % 2 == 0)
					mcts.player = WHITE;
				else
					mcts.player = BLACK;

				Move prev_move(moves.back());
				board.push_encoded(prev_move.hash());

				/*log << "------------------------- MOVE HISTORY -----------------------------\n";
				for (std::string uci_move : moves)
					log << uci_move << " ";
				log << "\n";
				log << "------------------------- MOVE HISTORY -----------------------------\n\n";

				log << "OPPONENT'S MOVE: " << prev_move << "\n";
				log << board.p << "\n";
				log.flush();*/
			} else {
				mcts.player = WHITE;
			}
			});
		uci.receive_go.connect([&](const std::map<uci::command, std::string>& parameters) {
			if (parameters.contains(uci::command::white_time)) {
				if (mcts.time_per_move == -1LL) {
					switch (mcts.player) {
					case WHITE:
						mcts.init_time(std::stoi(parameters.at(uci::command::white_time)), std::stoi(parameters.at(uci::command::white_increment)));
						break;
					case BLACK:
						mcts.init_time(std::stoi(parameters.at(uci::command::black_time)), std::stoi(parameters.at(uci::command::black_increment)));
						break;
					}
				} else {
					switch (mcts.player) {
					case WHITE:
						mcts.update_time(std::stoi(parameters.at(uci::command::white_time)));
						break;
					case BLACK:
						mcts.update_time(std::stoi(parameters.at(uci::command::black_time)));
						break;
					}
				}
			} else if (parameters.contains(uci::command::move_time)) {
				mcts.time_per_move = std::stoll(parameters.at(uci::command::move_time)) - 500LL;
			}

			Move best_move = mcts.best_move(board);
			board.push(best_move);
			mcts.executed_moves++;

			/*log << "MY MOVE: " << best_move << "\n";
			log << board.p << "\n";
			log.flush();*/

			if (debug_mode)
				std::cout << "info depth " << mcts.explored_nodes << " score cp " << mcts.best_move_cp << " nodes " << mcts.explored_nodes << " time " << mcts.time_simulating << " nps " << static_cast<long long>(static_cast<double>(mcts.explored_nodes) / (static_cast<double>(mcts.time_simulating) / 1000.0)) << "\n";
			
			std::cout << "bestmove " << best_move << "\n";
			});

		// start communication with the UI through console
		uci.launch();
	}

	return 0;
}
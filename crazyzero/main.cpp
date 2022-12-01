#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <format>
#include "utils.h"
#include "crazyzero.h"
#include "cppflow/cppflow.h"
#include "uci/uci.hpp"



void play(std::ofstream& log, const int num_games, const int player1_nsims, const crazyzero::TestMask player1_config, const int player2_nsims, const crazyzero::TestMask player2_config) {
	crazyzero::Board board;
	crazyzero::MCTS player1;
	crazyzero::MCTS player2;
	player1.init(board);
	player2.nnet.model = player1.nnet.model;
	Color turn = WHITE, p1 = WHITE, p2 = BLACK;

	auto const time = std::chrono::current_zone()
		->to_local(std::chrono::system_clock::now());
	std::string date = std::format("{:%Y-%m-%d}", time);

	crazyzero::PGN pgn_log("./pgn_log.pgn");

	std::string p1_name = " -----|----", p2_name = "800 NORM";

	// configure player1
	player1.player = p1;
	player1.time_control = false;
	player1.num_sims = player1_nsims;
	player1.add_policy_enhancement_strategy(crazyzero::PolicyEnhancementStrat::Dirichlet);

	std::cout << "[PLAYER 1]: " << player1_nsims << " sims/move\n";

	if (player1_config & crazyzero::material_mask) {
		player1.eval.add_eval(crazyzero::material_mask);
		p1_name[1] = '1';
		std::cout << "    - material\n";
	}

	if (player1_config & crazyzero::pawn_structure_mask) {
		player1.eval.add_eval(crazyzero::pawn_structure_mask);
		p1_name[2] = '2';
		std::cout << "    - pawn structure\n";
	}

	if (player1_config & crazyzero::king_safety_mask) {
		player1.eval.add_eval(crazyzero::king_safety_mask);
		p1_name[3] = '3';
		std::cout << "    - king safety\n";
	}

	if (player1_config & crazyzero::piece_placement_mask) {
		player1.eval.add_eval(crazyzero::piece_placement_mask);
		p1_name[4] = '4';
		std::cout << "    - piece placement\n";
	}

	if (player1_config & crazyzero::board_control_mask) {
		player1.eval.add_eval(crazyzero::board_control_mask);
		p1_name[5] = '5';
		std::cout << "    - board control\n";
	}

	if (player1_config & crazyzero::checking_moves_mask) {
		player1.add_policy_enhancement_strategy(crazyzero::PolicyEnhancementStrat::CheckingMoves);
		p1_name[7] = '6';
		std::cout << "    - checking moves\n";
	}

	if (player1_config & crazyzero::forking_moves_mask) {
		player1.add_policy_enhancement_strategy(crazyzero::PolicyEnhancementStrat::ForkingMoves);
		p1_name[8] = '7';
		std::cout << "    - forking moves\n";
	}

	if (player1_config & crazyzero::dropping_moves_mask) {
		player1.add_policy_enhancement_strategy(crazyzero::PolicyEnhancementStrat::DroppingMoves);
		p1_name[9] = '8';
		std::cout << "    - dropping moves\n";
	}

	if (player1_config & crazyzero::capturing_moves_mask) {
		player1.add_policy_enhancement_strategy(crazyzero::PolicyEnhancementStrat::CapturingMoves);
		p1_name[10] = '9';
		std::cout << "    - capturing moves\n";
	}

	p1_name = std::to_string(player1_nsims) + p1_name;

	// configure player2
	player2.player = p2;
	player2.time_control = false;
	player2.num_sims = player2_nsims;
	player2.add_policy_enhancement_strategy(crazyzero::PolicyEnhancementStrat::Dirichlet);

	std::cout << "[PLAYER 2]: " << player2_nsims << " sims/move\n";

	if (player2_config & crazyzero::material_mask) {
		player2.eval.add_eval(crazyzero::material_mask);
		std::cout << "    - material\n";
	}

	if (player2_config & crazyzero::pawn_structure_mask) {
		player2.eval.add_eval(crazyzero::pawn_structure_mask);
		std::cout << "    - pawn structure\n";
	}

	if (player2_config & crazyzero::king_safety_mask) {
		player2.eval.add_eval(crazyzero::king_safety_mask);
		std::cout << "    - king safety\n";
	}

	if (player2_config & crazyzero::piece_placement_mask) {
		player2.eval.add_eval(crazyzero::piece_placement_mask);
		std::cout << "    - piece placement\n";
	}

	if (player2_config & crazyzero::board_control_mask) {
		player2.eval.add_eval(crazyzero::board_control_mask);
		std::cout << "    - board control\n";
	}

	if (player2_config & crazyzero::checking_moves_mask) {
		player2.add_policy_enhancement_strategy(crazyzero::PolicyEnhancementStrat::CheckingMoves);
		std::cout << "    - checking moves\n";
	}

	if (player2_config & crazyzero::forking_moves_mask) {
		player2.add_policy_enhancement_strategy(crazyzero::PolicyEnhancementStrat::ForkingMoves);
		std::cout << "    - forking moves\n";
	}

	if (player2_config & crazyzero::dropping_moves_mask) {
		player2.add_policy_enhancement_strategy(crazyzero::PolicyEnhancementStrat::DroppingMoves);
		std::cout << "    - dropping moves\n";
	}

	if (player2_config & crazyzero::dropping_moves_mask) {
		player2.add_policy_enhancement_strategy(crazyzero::PolicyEnhancementStrat::CapturingMoves);
		std::cout << "    - capturing moves\n";
	}

	int p1_wins = 0;
	int p2_wins = 0;
	int draws = 0;

	std::cout << "[STARTED PLAYING]\n";

	for (int i = 0; i < num_games; i++) {
		if (p1 == WHITE)
			pgn_log.new_game("Testing", i + 1, date, p1_name, p2_name);
		else
			pgn_log.new_game("Testing", i + 1, date, p2_name, p1_name);

		int move_num = 0;
		std::cout << "Game " << i + 1 << " of " << num_games << ", move " << move_num << ": ..\r";
		double end_score = 0.0;
		while (end_score == 0.0) {
			Move move;
			if (turn == player1.player) {
				move = player1.best_move(board);
				pgn_log.add_move(board.san(move));
				board.push(move);
				std::cout << "Game " << i + 1 << " of " << num_games << ", move " << ++move_num << ": P1\r";
			} else if (turn == player2.player) {
				move = player2.best_move(board);
				pgn_log.add_move(board.san(move));
				board.push(move);
				std::cout << "Game " << i + 1 << " of " << num_games << ", move " << ++move_num << ": P2\r";
			}

			turn = ~turn;
			end_score = board.end_score(turn);

			if (move_num >= 200) {
				// reset game
				turn = WHITE;
				board.reset();
				player1.soft_reset();
				player1.player = p1;
				player2.soft_reset();
				player2.player = p2;
				end_score = 0.0;
				move_num = 0;

				if (p1 == WHITE)
					pgn_log.new_game("Testing", i + 1, date, p1_name, p2_name);
				else
					pgn_log.new_game("Testing", i + 1, date, p2_name, p1_name);
			}
		}

		if (end_score < 0) {
			if (turn == WHITE) {
				std::cout << "Game " << i + 1 << " of " << num_games << ", move " << move_num << ": BLACK\n";
				pgn_log.flush(BLACK);
				if (p1 == BLACK)
					p1_wins++;
				else
					p2_wins++;
			} else {
				std::cout << "Game " << i + 1 << " of " << num_games << ", move " << move_num << ": WHITE\n";
				pgn_log.flush(WHITE);
				if (p1 == WHITE)
					p1_wins++;
				else
					p2_wins++;
			}
		} else {
			std::cout << "Game " << i + 1 << " of " << num_games << ", move " << move_num << ": DRAW\n";
			pgn_log.flush(NO_COLOR);
			draws++;
		}

		turn = WHITE;
		p1 = ~p1;
		p2 = ~p2;
		board.reset();
		player1.soft_reset();
		player1.player = p1;
		player2.soft_reset();
		player2.player = p2;
	}

	crazyzero::Elo elo(p1_wins, p2_wins, draws);

	std::cout << "ELO difference: " << elo.diff() << " +/- " << elo.error_margin() << "\n";

	log << "Testing session: " << p1_name << " vs. " << p2_name << "\n";
	log << "P1 wins: " << p1_wins << "\n";
	log << "P2 wins: " << p2_wins << "\n";
	log << "Draws: " << draws << "\n";
	log << "ELO difference: " << elo.diff() << " +/- " << elo.error_margin() << "\n";
	log.flush();

	pgn_log.close();

	//delete player1.nnet.model;
	//player1.nnet.model = NULL;
	player2.nnet.model = NULL;
}

int get_num_simulations() {
	int num_sims = 0;
	while (num_sims == 0) {
		std::string num_s;
		std::cout << "Select number of simulations per move:\n";
		std::cout << "    1 - 100\n";
		std::cout << "    2 - 200\n";
		std::cout << "    3 - 300\n";
		std::cout << "    4 - 400\n";
		std::cout << "    5 - 800\n";
		std::cout << "    6 - 1600\n";
		std::cout << "    7 - 2400\n";
		std::cin >> num_s;
		
		if (num_s.length() > 0) {
			switch (num_s[0]) {
			case '1':
				num_sims = 100;
				break;
			case '2':
				num_sims = 200;
				break;
			case '3':
				num_sims = 300;
				break;
			case '4':
				num_sims = 400;
				break;
			case '5':
				num_sims = 800;
				break;
			case '6':
				num_sims = 1600;
				break;
			case '7':
				num_sims = 2400;
				break;
			default:
				std::cout << "ERROR: invalid selection\n";
				break;
			}
		} else {
			std::cout << "ERROR: no option selected\n";
		}
	}
	return num_sims;
}

std::string get_command() {
	std::string command;
	while (true) {
		std::cout << "Select modifications to be tested:\n";
		std::cout << "    1 - material\n";
		std::cout << "    2 - pawn structure\n";
		std::cout << "    3 - king safety\n";
		std::cout << "    4 - piece placement\n";
		std::cout << "    5 - board control\n";
		std::cout << "    6 - checking moves\n";
		std::cout << "    7 - forking moves\n";
		std::cout << "    8 - dropping moves\n";
		std::cout << "    9 - capturing moves\n";

		std::cin >> command;
		if (command.length())
			break;
		else
			std::cout << "ERROR: empty command\n";
	}
	return command;
}

void start_testing(std::ofstream& log) {
	std::cout << "[TEST MODE]\n";

	int num_sims_1 = get_num_simulations();
	int num_sims_2 = get_num_simulations();

	crazyzero::TestMask eval_config = 0;
	bool no_command = false;

	while (!no_command && eval_config == 0) {
		std::string command = get_command();
		for (char mod : command) {
			switch (mod) {
			case '1':
				eval_config |= crazyzero::material_mask;
				break;
			case '2':
				eval_config |= crazyzero::pawn_structure_mask;
				break;
			case '3':
				eval_config |= crazyzero::king_safety_mask;
				break;
			case '4':
				eval_config |= crazyzero::piece_placement_mask;
				break;
			case '5':
				eval_config |= crazyzero::board_control_mask;
				break;
			case '6':
				eval_config |= crazyzero::checking_moves_mask;
				break;
			case '7':
				eval_config |= crazyzero::forking_moves_mask;
				break;
			case '8':
				eval_config |= crazyzero::dropping_moves_mask;
				break;
			case '9':
				eval_config |= crazyzero::capturing_moves_mask;
				break;
			case '0':
				eval_config = 0;
				no_command = true;
				break;
			default:
				std::cout << "ERROR: invalid command\n";
				eval_config = 0;
				break;
			}
		}
	}

	std::cout << "[STARTED TESTING]\n";
	play(log, 100, num_sims_1, eval_config, num_sims_2, 0);
}

int main(int argc, char* argv[]) 
{
	_putenv("TF_CPP_MIN_LOG_LEVEL=3");
	using namespace crazyzero;

	initialise_all_databases();
	zobrist::initialise_zobrist_keys();
	initialise_eval_tables();

	std::cout << "CrazyZero 2.1 by Anei Makovec\n";

	/*
	Board b;
	MateSearch m;

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	Move move = m.mate_move(b);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	if (move.from() != NO_SQUARE)
		std::cout << "CHECKMATE\n";
	else
		std::cout << "NO CHECKMATE\n";

	std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " milliseconds\n";
	*/

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

	return 0;

	if (argc > 1 && std::string(argv[1]) == "-test_mode") {
		// enter testing mode
		std::ofstream log("./log.txt", std::ios_base::out | std::ios_base::app);
		start_testing(log);
		log.close();
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
			});
		uci.receive_debug.connect([&](bool on) {
			debug_mode = on;
			});
		uci.receive_is_ready.connect([&]() {
			mcts.init(board);
			if (mcts.config.changed) {
				mcts.update_config();
			}
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
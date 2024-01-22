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

#include <iostream>
#include <stdlib.h>
#include <sstream>
#include "utils.h"
#include "crazyrabbit.h"
#include "cppflow/cppflow.h"
#include "uci/uci.h"

using namespace crazyrabbit;

int main(int argc, char* argv[]) 
{
	_putenv("TF_CPP_MIN_LOG_LEVEL=3");

	initialise_all_databases();
	zobrist::initialise_zobrist_keys();
	initialise_eval_tables();

	std::cout << "CrazyRabbit 2.2 by Anei Makovec\n";

	Board board;
	MCTS mcts;
	uci uci;
	bool debug_mode = true;

	// register callbacks to the messages from the UI and respond appropriately.
	uci.receive_uci.connect([&]() {
		uci.send_id("CrazyRabbit 2.2", "Anei Makovec");
		uci.send_option_combo_box("UCI_Variant", "crazyhouse", { "crazyhouse" });
		uci.send_option_combo_box("TimeControl", "Default", { "Default", "None" });
		uci.send_option_spin_wheel("Simulations/Move", 100, 1, 100000);
		uci.send_option_combo_box("BestMoveStrategy", "Default", { "Default", "Q-value" });
		uci.send_option_combo_box("NodeExpansionStrategy", "Default", { "Default", "Exploration" });
		uci.send_option_combo_box("BackpropStrategy", "Default", { "Default", "SMA" });
		uci.send_option_check_box("UseOpenings", false);
		uci.send_option_check_box("UseMateSearch", false);
		uci.send_option_spin_wheel("MateSearchMaxDepth", 3, 1, 10000);
		uci.send_option_check_box("MoveFiltering", false);
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
		uci.send_uci_ok();
	});

	uci.receive_set_option.connect([&](const std::string& name, const std::string& value) {
		if (name == "UCI_Variant") 
		{
			// pass
			return;
		} 
		else if (name == "TimeControl") 
		{
			if (value == "Default") {
				mcts.time_control = true;
			} else if (value == "None") {
				mcts.time_control = false;
			}
		} 
		else if (name == "Simulations/Move") 
		{
			int sims = stoi(value);
			if (sims >= 1 && sims <= 100000)
				mcts.num_sims = sims;
		} 
		else if (name == "BestMoveStrategy") 
		{
			if (value == "Default")
				mcts.set_best_move_strategy(BestMoveStrat::Default);
			else if (value == "Q-value")
				mcts.set_best_move_strategy(BestMoveStrat::Q_value);
		} 
		else if (name == "NodeExpansionStrategy") 
		{
			if (value == "Default")
				mcts.set_node_expansion_strategy(NodeExpansionStrat::Default);
			else if (value == "Exploration")
				mcts.set_node_expansion_strategy(NodeExpansionStrat::Exploration);
		} 
		else if (name == "BackpropStrategy") 
		{
			if (value == "Default")
				mcts.set_backprop_strategy(BackpropStrat::Default);
			else if (value == "SMA")
				mcts.set_backprop_strategy(BackpropStrat::SMA);
		} 
		else if (name == "UseOpenings")
		{
			if (value == "true")
				mcts.use_openings = true;
			else
				mcts.use_openings = false;
		}
		else if (name == "UseMateSearch")
		{
			if (value == "true")
				mcts.use_mate_search = true;
			else
				mcts.use_mate_search = false;
		} 
		else if (name == "MateSearchMaxDepth")
		{
			int depth = stoi(value);
			if (depth >= 1 && depth <= 10000)
				mcts.mate_search.max_depth = depth;
		} 
		else if (name == "MoveFiltering")
		{
			if (value == "true")
				mcts.filter_moves = true;
			else
				mcts.filter_moves = false;
		}
		else if (name == "PE_Dirichlet") 
		{
			if (value == "true")
				mcts.config.use_dirichlet = true;
			else
				mcts.config.use_dirichlet = false;
			mcts.set_config(mcts.config);
		} 
		else if (name == "PE_CheckingMoves") 
		{
			if (value == "true")
				mcts.config.policy_mask |= checking_moves_mask;
			else
				mcts.config.policy_mask &= ~checking_moves_mask;
			mcts.set_config(mcts.config);
		} 
		else if (name == "PE_ForkingMoves") 
		{
			if (value == "true")
				mcts.config.policy_mask |= forking_moves_mask;
			else
				mcts.config.policy_mask &= ~forking_moves_mask;
			mcts.set_config(mcts.config);
		} 
		else if (name == "PE_DroppingMoves") 
		{
			if (value == "true")
				mcts.config.policy_mask |= dropping_moves_mask;
			else
				mcts.config.policy_mask &= ~dropping_moves_mask;
			mcts.set_config(mcts.config);
		} 
		else if (name == "PE_CapturingMoves") 
		{
			if (value == "true")
				mcts.config.policy_mask |= capturing_moves_mask;
			else
				mcts.config.policy_mask &= ~capturing_moves_mask;
			mcts.set_config(mcts.config);
		} 
		else if (name == "Eval_Material") 
		{
			if (value == "true")
				mcts.config.eval_mask |= material_mask;
			else
				mcts.config.eval_mask &= ~material_mask;
			mcts.set_config(mcts.config);
		} 
		else if (name == "Eval_PawnStructure") 
		{
			if (value == "true")
				mcts.config.eval_mask |= pawn_structure_mask;
			else
				mcts.config.eval_mask &= ~pawn_structure_mask;
			mcts.set_config(mcts.config);
		} 
		else if (name == "Eval_KingSafety") 
		{
			if (value == "true")
				mcts.config.eval_mask |= king_safety_mask;
			else
				mcts.config.eval_mask &= ~king_safety_mask;
			mcts.set_config(mcts.config);
		} 
		else if (name == "Eval_PiecePlacement") 
		{
			if (value == "true")
				mcts.config.eval_mask |= piece_placement_mask;
			else
				mcts.config.eval_mask &= ~piece_placement_mask;
			mcts.set_config(mcts.config);
		} 
		else if (name == "Eval_BoardControl") 
		{
			if (value == "true")
				mcts.config.eval_mask |= board_control_mask;
			else
				mcts.config.eval_mask &= ~board_control_mask;
			mcts.set_config(mcts.config);
		} 
		else 
		{
			std::cout << "UCI ERROR: option " << name << " could not be set to value " << value << ".\n";
		}
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
	});

	uci.receive_position.connect([&](const std::string& fen, const std::vector<std::string>& moves) {
		if (moves.size()) 
		{
			if (moves.size() % 2 == 0)
				mcts.player = WHITE;
			else
				mcts.player = BLACK;

			Move prev_move(moves.back());
			board.push_encoded(prev_move.hash());
		} 
		else 
		{
			mcts.player = WHITE;
		}
	});

	uci.receive_go.connect([&](const std::map<uci::command, std::string>& parameters) {
		if (parameters.contains(uci::command::white_time)) 
		{
			if (mcts.time_per_move == -1LL) 
			{
				switch (mcts.player) 
				{
				case WHITE:
					mcts.init_time(std::stoi(parameters.at(uci::command::white_time)), std::stoi(parameters.at(uci::command::white_increment)));
					break;
				case BLACK:
					mcts.init_time(std::stoi(parameters.at(uci::command::black_time)), std::stoi(parameters.at(uci::command::black_increment)));
					break;
				}
			} 
			else 
			{
				switch (mcts.player) 
				{
				case WHITE:
					mcts.update_time(std::stoi(parameters.at(uci::command::white_time)));
					break;
				case BLACK:
					mcts.update_time(std::stoi(parameters.at(uci::command::black_time)));
					break;
				}
			}
		} 
		else if (parameters.contains(uci::command::move_time)) 
		{
			mcts.time_per_move = std::stoll(parameters.at(uci::command::move_time)) - 500LL;
		}

		Move best_move = mcts.best_move(board);
		board.push(best_move);

		if (debug_mode)
			std::cout << "info depth " << mcts.explored_nodes << " score cp " << mcts.best_move_cp << " nodes " << mcts.explored_nodes << " time " << mcts.time_simulating << " nps " << static_cast<long long>(static_cast<double>(mcts.explored_nodes) / (static_cast<double>(mcts.time_simulating) / 1000.0)) << "\n";
			
		std::cout << "bestmove " << best_move << "\n";
	});

	// start communication with the UI through console
	uci.launch();

	return 0;
}
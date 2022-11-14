import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

import chess
import chess.pgn

from NNet import NNet
from GameBoard import GameBoard

log = logging.getLogger(__name__)

ACTION_SIZE = 5184
LOAD_GAMES = 100

class Trainer():

    def train_with_games(self, games_file_path, num_train_games):
        # read all the games from the file
        pgn_file = open(games_file_path)
        game_info = chess.pgn.read_game(pgn_file)

        nnet = NNet()
        all_games = 0
        num_games = 0
        train_set = list()

        display_text = True

        while game_info:
            if num_games < LOAD_GAMES and all_games < num_train_games:
                # replay each game and store the input representations of every game state
                if display_text:
                    log.info("Loading train data...")
                    display_text = False

                board = GameBoard()

                res = 0.0
                if game_info.headers["Result"] == "1-0":
                    res = 1.0
                elif game_info.headers["Result"] == "0-1":
                    res = -1.0

                player = True

                for move in game_info.mainline_moves():
                    if board.turn:
                        v = res
                    else:
                        v = -res

                    action = board.encodeAction(move)
                    pi = np.zeros(ACTION_SIZE)
                    pi[action] = 1.0
                    input_rep = board.inputRepresentation()

                    train_set.append((input_rep, pi, v))

                    board.push(move)

                num_games += 1
                all_games += 1
                game_info = chess.pgn.read_game(pgn_file)
            else:
                # train the network
                log.info("Started training...")

                num_games = 0
                display_text = True

                shuffle(train_set)
                nnet.train(train_set)

                train_set.clear()

                log.info("Trained on " + str(all_games) + " in total.")

                if all_games >= num_train_games:
                    break

        # save the current neural network
        filename = "human_data_" + str(num_train_games)
        nnet.save_checkpoint(filename=filename)
        log.info("Done!")

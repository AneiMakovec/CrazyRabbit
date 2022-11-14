import os
import logging
import coloredlogs
from Trainer import Trainer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

def main():
    num_games = input("Insert number of games to train with: ")

    try:
        num_games = int(num_games)
    except:
        print("Please insert only the number of games.")
        exit(0)

    file_name = input("Insert path to .pgn file with games: ")

    log.info("Starting %s...", Trainer.__name__)

    t = Trainer()

    t.train_with_games(file_name, num_games)


if __name__ == "__main__":
    main()

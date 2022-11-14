import copy
from chess import *
from chess.variant import *
import typing

Up = 0
Down = 1
Left = 2
Right = 3
UpLeft = 4
UpRight = 5
DownLeft = 6
DownRight = 7

knight_directions = [15, 17, -17, -15, 6, -10, 10, -6]

class GameBoard(chess.variant.CrazyhouseBoard):

    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"

    def __init__(self, fen: Optional[str] = starting_fen, chess960: bool = False) -> None:
        super().__init__(fen, chess960=chess960)
        self.state_repetitions = {}
        state, _ = super().fen().split(None, 1)
        self.state_repetitions[state] = 0

    def push(self, move: chess.Move) -> None:
        # make move
        super().push(move)
        state, _ = self.fen().split(None, 1)
        if state not in self.state_repetitions:
            self.state_repetitions[state] = 0
        else:
            self.state_repetitions[state] += 1

    def copy(self: CrazyhouseBoardT, stack: Union[bool, int] = True) -> CrazyhouseBoardT:
        board = super().copy(stack=stack)
        board.state_repetitions = copy.deepcopy(self.state_repetitions)
        return board

    def mirror(self: CrazyhouseBoardT) -> CrazyhouseBoardT:
        board = super().mirror()
        board.state_repetitions = copy.deepcopy(self.state_repetitions)
        self.mirrored = True
        return board

    def squareMovement(self, source_square, target_square):
        source_rank = chess.square_rank(source_square)
        source_file = chess.square_file(source_square)
        target_rank = chess.square_rank(target_square)
        target_file = chess.square_file(target_square)

        if source_file == target_file:
            # up
            if target_rank > source_rank:
                return Up, chess.square_distance(source_square, target_square)
            # down
            else:
                return Down, chess.square_distance(source_square, target_square)
        elif source_rank == target_rank:
            # right
            if target_file > source_file:
                return Right, chess.square_distance(source_square, target_square)
            # left
            else:
                return Left, chess.square_distance(source_square, target_square)
        elif target_rank > source_rank:
            # up right
            if target_file > source_file:
                return UpRight, chess.square_distance(source_square, target_square)
            # up left
            else:
                return UpLeft, chess.square_distance(source_square, target_square)
        else:
            # down right
            if target_file > source_file:
                return DownRight, chess.square_distance(source_square, target_square)
            # down left
            else:
                return DownLeft, chess.square_distance(source_square, target_square)

    def isKnightMove(self, source_square, target_square):
        for direction in range(8):
            if source_square + knight_directions[direction] == target_square:
                return direction

        return -1

    def encodeAction(self, action):
        source_square = action.from_square
        target_square = action.to_square
        promoted_piece = action.promotion
        dropped_piece = action.drop

        encoded_action = 0
        if dropped_piece:
            # encode drops
            encoded_action = 76 + (dropped_piece - 1)
        elif promoted_piece:
            # encode promotons
            encoded_action = 64 + (abs(target_square - source_square) - 7) + ((promoted_piece - 2) * 3)
        elif (direction := self.isKnightMove(source_square, target_square)) >= 0:
            # encode knight moves
            encoded_action = 56 + direction
        else:
            # encode all other (normal) moves
            direction, num_steps = self.squareMovement(source_square, target_square)
            encoded_action = direction * 7 + (num_steps - 1)

        # now return the full encoded action
        return source_square * 81 + encoded_action

    def decodeAction(self, action, player):
        source_square = action // 81
        move_index = action % 81

        if move_index <= 55:
            # it's a move in one of 8 possible directions
            move_direction = move_index // 7
            move_steps = (move_index % 7) + 1

            steps_to_move = (0, 0)
            if (move_direction == Up):
                steps_to_move = (0, move_steps)
            elif (move_direction == Down):
                steps_to_move = (0, -move_steps)
            elif (move_direction == Left):
                steps_to_move = (-move_steps, 0)
            elif (move_direction == Right):
                steps_to_move = (move_steps, 0)
            elif (move_direction == UpLeft):
                steps_to_move = (-move_steps, move_steps)
            elif (move_direction == UpRight):
                steps_to_move = (move_steps, move_steps)
            elif (move_direction == DownLeft):
                steps_to_move = (-move_steps, -move_steps)
            elif (move_direction == DownRight):
                steps_to_move = (move_steps, -move_steps)

            source_rank = source_square // 8
            source_file = source_square % 8

            target_square = chess.square(source_file + steps_to_move[0], source_rank + steps_to_move[1])
            return chess.Move(source_square, target_square)
        elif move_index <= 63:
            # it's a knight move
            target_square = source_square + knight_directions[move_index - 56]

            return chess.Move(source_square, target_square)
        elif move_index <= 75:
            # it's an underpromotion
            move_index -= 64

            promoted_piece = (move_index // 3) + 2
            move_direction = move_index % 3

            steps_to_move = 0
            if move_direction == 0:
                steps_to_move = 7
            elif move_direction == 1:
                steps_to_move = 8
            else:
                steps_to_move = 9

            # if the player to play is the second player then invert the move direction
            steps_to_move = player * steps_to_move

            target_square = source_square + steps_to_move

            return chess.Move(source_square, target_square, promoted_piece)
        else:
            # it's a drop
            dropped_piece = move_index - 76 + 1

            return chess.Move(source_square, source_square, None, dropped_piece)


    def inputRepresentation(self):
        input_rep = np.zeros((34, 64))

        # pieces positions for each player
        for player in [True, False]:
            for piece in range(1, 7):
                input_rep[piece - 1 if player else 6 + (piece - 1), list(self.pieces(piece, player))] = 1

        # how often the board position has occured
        s, _ = self.fen().split(None, 1)
        if s not in self.state_repetitions:
            s, _ = self.mirror().fen().split(None, 1)

        input_rep[12, list(range(64))] = self.state_repetitions[s] / 500
        input_rep[13, list(range(64))] = self.state_repetitions[s] / 500

        # pocket counts
        for player in [True, False]:
            for piece in range(1, 6):
                input_rep[14 + (piece - 1) if player else 19 + (piece - 1), list(range(64))] = self.pockets[player].count(piece) / 32

        # promoted pieces (pawns)
        promoted_white = []
        promoted_black = []
        for square in range(64):
            if self.promoted & chess.BB_SQUARES[square] != 0:
                piece = self.piece_at(square)
                if piece:
                    if piece.color == chess.WHITE:
                        promoted_white.append(square)
                    else:
                        promoted_black.append(square)

        for player in [True, False]:
            input_rep[24 if player else 25, promoted_white if player else promoted_black] = 1

        # en-passant square
        if (self.ep_square):
            input_rep[26][self.ep_square] = 1;

        # color
        input_rep[27, list(range(64))] = 1 if self.turn else 0

        # total move count
        input_rep[28, list(range(64))] = self.fullmove_number / 500

        # castling rights
        for player in [True, False]:
            input_rep[29 if player else 31, list(range(64))] = 1 if self.has_kingside_castling_rights(player) else 0
            input_rep[30 if player else 32, list(range(64))] = 1 if self.has_queenside_castling_rights(player) else 0

        # no-progress count (halfmove count)
        input_rep[33, list(range(64))] = self.halfmove_clock / 40

        return input_rep

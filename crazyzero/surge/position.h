#ifndef SURGE_POSITION_HPP
#define SURGE_POSITION_HPP

#include <ostream>
#include <string>
#include <utility>
#include <sstream>
#include <list>
#include <unordered_map>
#include "types.h"
#include "tables.h"

//A psuedorandom number generator
//Source: Stockfish
class PRNG {
	uint64_t s;

	uint64_t rand64() {
		s ^= s >> 12, s ^= s << 25, s ^= s >> 27;
		return s * 2685821657736338717LL;
	}

public:
	PRNG(uint64_t seed) : s(seed) {}

	//Generate psuedorandom number
	template<typename T> T rand() { return T(rand64()); }

	//Generate psuedorandom number with only a few set bits
	template<typename T> 
	T sparse_rand() {
		return T(rand64() & rand64() & rand64());
	}
};


namespace zobrist {
	extern uint64_t zobrist_table[NPIECES][NSQUARES];
	extern void initialise_zobrist_keys();
}

//Stores position information which cannot be recovered on undo-ing a move
struct UndoInfo {
	//The bitboard of squares on which pieces have either moved from, or have been moved to. Used for castling
	//legality checks
	Bitboard entry;
	
	//The piece that was captured on the last move
	Piece captured;

	//If the captured piece was a promoted pawn
	bool promoted;
	
	//The en passant square. This is the square which pawns can move to in order to en passant capture an enemy pawn that has 
	//double pushed on the previous move
	Square epsq;

	//The number of halfmoves since the last capture or pawn advance
	int halfmove_clock;

	//The number of the full move. It starts at 1 and is incremented after Black's move
	int fullmove_number;

	constexpr UndoInfo() : entry(0), captured(NO_PIECE), promoted(false), epsq(NO_SQUARE), halfmove_clock(0), fullmove_number(1) {}
	
	//This preserves the entry bitboard across moves
	UndoInfo(const UndoInfo& prev) : 
		entry(prev.entry), captured(NO_PIECE), promoted(false), epsq(NO_SQUARE), halfmove_clock(prev.halfmove_clock + 1), fullmove_number(prev.fullmove_number) {}
};

class Position {
private:
	//A bitboard of the locations of each piece
	Bitboard piece_bb[NPIECES];
	
	//A mailbox representation of the board. Stores the piece occupying each square on the board
	Piece board[NSQUARES];

	//The number of pieces of each type in either player's pocket.
	int pocket[2][NPIECE_TYPES - 1];
	
	//The side whose turn it is to play next
	Color side_to_play;
	
	//The current game ply (depth), incremented after each move 
	int game_ply;
	
	//The zobrist hash of the position, which can be incrementally updated and rolled back after each
	//make/unmake
	uint64_t hash;
public:
	//The history of non-recoverable information
	//UndoInfo history[256];
	std::list<UndoInfo> history;
	
	//The bitboard of enemy pieces that are currently attacking the king, updated whenever generate_moves()
	//is called
	Bitboard checkers;
	
	//The bitboard of pieces that are currently pinned to the king by enemy sliders, updated whenever 
	//generate_moves() is called
	Bitboard pinned;

	//The bitboard of pieces that have been promoted, updated whenever play() or undo() are called
	Bitboard promoted;

	//How many times a certain board position occured
	std::unordered_map<std::string, int> repetitions;
	
	
	Position() : piece_bb{ 0 }, side_to_play(WHITE), game_ply(0), board{}, pocket{ {}, {} },
		hash(0), pinned(0), checkers(0), promoted(0) {
		
		//Sets all squares on the board as empty
		for (int i = 0; i < 64; i++) board[i] = NO_PIECE;
		//history[0] = UndoInfo();
		history.emplace_back();

		//Set the number of all pieces in pockets to zero
		for (int i = 0; i < NPIECE_TYPES - 1; i++) {
			pocket[WHITE][i] = 0;
			pocket[BLACK][i] = 0;
		}
	}
	
	//Places a piece on a particular square and updates the hash. Placing a piece on a square that is 
	//already occupied is an error
	inline void put_piece(Piece pc, Square s) {
		board[s] = pc;
		piece_bb[pc] |= SQUARE_BB[s];
		hash ^= zobrist::zobrist_table[pc][s];
	}

	//Removes a piece from a particular square and updates the hash. 
	inline void remove_piece(Square s) {
		hash ^= zobrist::zobrist_table[board[s]][s];
		piece_bb[board[s]] &= ~SQUARE_BB[s];
		board[s] = NO_PIECE;
	}

	void move_piece(Square from, Square to);
	void move_piece_quiet(Square from, Square to);


	friend std::ostream& operator<<(std::ostream& os, const Position& p);
	static void set(const std::string& fen, Position& p);
	std::string fen() const;
	std::string fen_hash() const;
	std::string fen_board() const;

	//Position& operator=(const Position&) = delete;
	inline bool operator==(const Position& other) const { return hash == other.hash; }
	Position& operator=(const Position& other) {
		for (int i = 0; i < NPIECES; i++)
			piece_bb[i] = other.piece_bb[i];

		for (int i = 0; i < NSQUARES; i++)
			board[i] = other.board[i];

		for (int i = 0; i < NPIECE_TYPES - 1; i++) {
			pocket[WHITE][i] = other.pocket[WHITE][i];
			pocket[BLACK][i] = other.pocket[BLACK][i];
		}

		side_to_play = other.side_to_play;
		game_ply = other.game_ply;
		hash = other.hash;

		/*for (int i = 0; i <= game_ply; i++)
			history[i] = other.history[i];*/

		history.assign(other.history.cbegin(), other.history.cend());

		checkers = other.checkers;
		pinned = other.pinned;
		promoted = other.promoted;

		for (auto reps : other.repetitions)
			repetitions[reps.first] = reps.second;
		return *this;
	}

	inline Bitboard bitboard_of(Piece pc) const { return piece_bb[pc]; }
	inline Bitboard bitboard_of(Color c, PieceType pt) const { return piece_bb[make_piece(c, pt)]; }
	inline Piece at(Square sq) const { return board[sq]; }
	inline int pocket_count(Color c, PieceType pt) const { return pocket[c][pt]; }
	inline Color turn() const { return side_to_play; }
	inline Square en_passant() { return history.back().epsq; }
	inline int halfmove_clock() { return history.back().halfmove_clock; }
	inline int fullmove_number() { return history.back().fullmove_number; }
	inline bool has_kingside_castling_rights(Color c) { return (c == WHITE) ? !(history.back().entry & WHITE_OO_MASK) : !(history.back().entry & BLACK_OO_MASK); }
	inline bool has_queenside_castling_rights(Color c) { return (c == WHITE) ? !(history.back().entry & WHITE_OOO_MASK) : !(history.back().entry & BLACK_OOO_MASK); }
	inline int ply() const { return game_ply; }
	inline uint64_t get_hash() const { return hash; }

	template<Color C> inline Bitboard diagonal_sliders() const;
	template<Color C> inline Bitboard orthogonal_sliders() const;
	template<Color C> inline Bitboard all_pieces() const;
	template<Color C> inline Bitboard attackers_from(Square s, Bitboard occ) const;

	template<Color C> inline bool in_check() const {
		return attackers_from<~C>(bsf(bitboard_of(C, KING)), all_pieces<WHITE>() | all_pieces<BLACK>());
	}

	template<Color C> void play(Move m);
	template<Color C> void undo(Move m);

	template<Color Us>
	Move *generate_legals(Move* list);

	template<Color Us>
	move_vector<Move> generate_legals();

	inline EndType is_checkmate();
	inline bool is_insufficient_material();
	inline bool is_seventyfive_moves();
	inline bool is_fivefold_repetition();

	template<Color Us>
	inline double end_score();
};

//Returns the bitboard of all bishops and queens of a given color
template<Color C> 
inline Bitboard Position::diagonal_sliders() const {
	return C == WHITE ? piece_bb[WHITE_BISHOP] | piece_bb[WHITE_QUEEN] :
		piece_bb[BLACK_BISHOP] | piece_bb[BLACK_QUEEN];
}

//Returns the bitboard of all rooks and queens of a given color
template<Color C> 
inline Bitboard Position::orthogonal_sliders() const {
	return C == WHITE ? piece_bb[WHITE_ROOK] | piece_bb[WHITE_QUEEN] :
		piece_bb[BLACK_ROOK] | piece_bb[BLACK_QUEEN];
}

//Returns a bitboard containing all the pieces of a given color
template<Color C> 
inline Bitboard Position::all_pieces() const {
	return C == WHITE ? piece_bb[WHITE_PAWN] | piece_bb[WHITE_KNIGHT] | piece_bb[WHITE_BISHOP] |
		piece_bb[WHITE_ROOK] | piece_bb[WHITE_QUEEN] | piece_bb[WHITE_KING] :

		piece_bb[BLACK_PAWN] | piece_bb[BLACK_KNIGHT] | piece_bb[BLACK_BISHOP] |
		piece_bb[BLACK_ROOK] | piece_bb[BLACK_QUEEN] | piece_bb[BLACK_KING];
}

//Returns a bitboard containing all pieces of a given color attacking a particluar square
template<Color C> 
inline Bitboard Position::attackers_from(Square s, Bitboard occ) const {
	return C == WHITE ? (pawn_attacks<BLACK>(s) & piece_bb[WHITE_PAWN]) |
		(attacks<KNIGHT>(s, occ) & piece_bb[WHITE_KNIGHT]) |
		(attacks<BISHOP>(s, occ) & (piece_bb[WHITE_BISHOP] | piece_bb[WHITE_QUEEN])) |
		(attacks<ROOK>(s, occ) & (piece_bb[WHITE_ROOK] | piece_bb[WHITE_QUEEN])) :

		(pawn_attacks<WHITE>(s) & piece_bb[BLACK_PAWN]) |
		(attacks<KNIGHT>(s, occ) & piece_bb[BLACK_KNIGHT]) |
		(attacks<BISHOP>(s, occ) & (piece_bb[BLACK_BISHOP] | piece_bb[BLACK_QUEEN])) |
		(attacks<ROOK>(s, occ) & (piece_bb[BLACK_ROOK] | piece_bb[BLACK_QUEEN]));
}

//Plays a move in the position
template<Color C>
void Position::play(const Move m) {
	//++game_ply;
	//history[game_ply] = UndoInfo(history[game_ply - 1]);
	history.emplace_back(history.back());

	if (side_to_play == BLACK)
		history.back().fullmove_number++;
	side_to_play = ~side_to_play;

	MoveFlags type = m.flags();
	history.back().entry |= SQUARE_BB[m.to()] | SQUARE_BB[m.from()];

	switch (type) {
	case QUIET:
		if (type_of(board[m.from()]) == PAWN)
			history.back().halfmove_clock = 0;

		//The to square is guaranteed to be empty here
		move_piece_quiet(m.from(), m.to());

		if (promoted & SQUARE_BB[m.from()]) {
			promoted &= ~SQUARE_BB[m.from()];
			promoted |= SQUARE_BB[m.to()];
		}
		break;
	case DOUBLE_PUSH:
		history.back().halfmove_clock = 0;

		//The to square is guaranteed to be empty here
		move_piece_quiet(m.from(), m.to());
			
		//This is the square behind the pawn that was double-pushed
		history.back().epsq = m.from() + relative_dir<C>(NORTH);
		break;
	case OO:
		if (C == WHITE) {
			move_piece_quiet(e1, g1);
			move_piece_quiet(h1, f1);
		} else {
			move_piece_quiet(e8, g8);
			move_piece_quiet(h8, f8);
		}			
		break;
	case OOO:
		if (C == WHITE) {
			move_piece_quiet(e1, c1); 
			move_piece_quiet(a1, d1);
		} else {
			move_piece_quiet(e8, c8);
			move_piece_quiet(a8, d8);
		}
		break;
	case EN_PASSANT:
		history.back().halfmove_clock = 0;

		move_piece_quiet(m.from(), m.to());
		remove_piece(m.to() + relative_dir<C>(SOUTH));

		pocket[C][PAWN]++;
		history.back().promoted = false;

		if (promoted & SQUARE_BB[m.from()]) {
			promoted &= ~SQUARE_BB[m.from()];
			promoted |= SQUARE_BB[m.to()];
		}
		break;
	case PR_KNIGHT:
		remove_piece(m.from());
		put_piece(make_piece(C, KNIGHT), m.to());

		promoted |= SQUARE_BB[m.to()];
		break;
	case PR_BISHOP:
		remove_piece(m.from());
		put_piece(make_piece(C, BISHOP), m.to());

		promoted |= SQUARE_BB[m.to()];
		break;
	case PR_ROOK:
		remove_piece(m.from());
		put_piece(make_piece(C, ROOK), m.to());

		promoted |= SQUARE_BB[m.to()];
		break;
	case PR_QUEEN:
		remove_piece(m.from());
		put_piece(make_piece(C, QUEEN), m.to());

		promoted |= SQUARE_BB[m.to()];
		break;
	case PC_KNIGHT:
		history.back().halfmove_clock = 0;

		remove_piece(m.from());
		history.back().captured = board[m.to()];

		if (promoted & SQUARE_BB[m.to()]) {
			pocket[C][PAWN]++;
			history.back().promoted = true;
		} else {
			pocket[C][type_of(board[m.to()])]++;
			history.back().promoted = false;
		}
		
		remove_piece(m.to());
		
		put_piece(make_piece(C, KNIGHT), m.to());

		promoted |= SQUARE_BB[m.to()];
		break;
	case PC_BISHOP:
		history.back().halfmove_clock = 0;

		remove_piece(m.from());
		history.back().captured = board[m.to()];
		
		if (promoted & SQUARE_BB[m.to()]) {
			pocket[C][PAWN]++;
			history.back().promoted = true;
		} else {
			pocket[C][type_of(board[m.to()])]++;
			history.back().promoted = false;
		}

		remove_piece(m.to());

		put_piece(make_piece(C, BISHOP), m.to());

		promoted |= SQUARE_BB[m.to()];
		break;
	case PC_ROOK:
		history.back().halfmove_clock = 0;

		remove_piece(m.from());
		history.back().captured = board[m.to()];
		
		if (promoted & SQUARE_BB[m.to()]) {
			pocket[C][PAWN]++;
			history.back().promoted = true;
		} else {
			pocket[C][type_of(board[m.to()])]++;
			history.back().promoted = false;
		}

		remove_piece(m.to());

		put_piece(make_piece(C, ROOK), m.to());

		promoted |= SQUARE_BB[m.to()];
		break;
	case PC_QUEEN:
		history.back().halfmove_clock = 0;

		remove_piece(m.from());
		history.back().captured = board[m.to()];
		
		if (promoted & SQUARE_BB[m.to()]) {
			pocket[C][PAWN]++;
			history.back().promoted = true;
		} else {
			pocket[C][type_of(board[m.to()])]++;
			history.back().promoted = false;
		}

		remove_piece(m.to());

		put_piece(make_piece(C, QUEEN), m.to());

		promoted |= SQUARE_BB[m.to()];
		break;
	case CAPTURE:
		history.back().halfmove_clock = 0;
		history.back().captured = board[m.to()];
		
		if (promoted & SQUARE_BB[m.to()]) {
			pocket[C][PAWN]++;
			history.back().promoted = true;
			promoted &= ~SQUARE_BB[m.to()];
		} else {
			pocket[C][type_of(board[m.to()])]++;
			history.back().promoted = false;
		}

		move_piece(m.from(), m.to());
		
		if (promoted & SQUARE_BB[m.from()]) {
			promoted &= ~SQUARE_BB[m.from()];
			promoted |= SQUARE_BB[m.to()];
		}
		break;
	case DROP_PAWN:
		pocket[C][PAWN]--;
		put_piece(make_piece(C, PAWN), m.to());
		break;
	case DROP_KNIGHT:
		pocket[C][KNIGHT]--;
		put_piece(make_piece(C, KNIGHT), m.to());
		break;
	case DROP_BISHOP:
		pocket[C][BISHOP]--;
		put_piece(make_piece(C, BISHOP), m.to());
		break;
	case DROP_ROOK:
		pocket[C][ROOK]--;
		put_piece(make_piece(C, ROOK), m.to());
		break;
	case DROP_QUEEN:
		pocket[C][QUEEN]--;
		put_piece(make_piece(C, QUEEN), m.to());
		break;
	}

	//Update the repetitions
	std::string board_fen = fen_board();
	if (!repetitions.contains(board_fen))
		repetitions[board_fen] = 0;
	else
		repetitions[board_fen]++;
}

//Undos a move in the current position, rolling it back to the previous position
template<Color C>
void Position::undo(const Move m) {
	//Update the repetitions
	std::string board_fen = fen_board();
	repetitions[board_fen]--;
	if (repetitions[board_fen] == 0)
		repetitions.erase(board_fen);

	MoveFlags type = m.flags();
	switch (type) {
	case QUIET:
		move_piece_quiet(m.to(), m.from());

		if (promoted & SQUARE_BB[m.to()]) {
			promoted &= ~SQUARE_BB[m.to()];
			promoted |= SQUARE_BB[m.from()];
		}
		break;
	case DOUBLE_PUSH:
		move_piece_quiet(m.to(), m.from());
		break;
	case OO:
		if (C == WHITE) {
			move_piece_quiet(g1, e1);
			move_piece_quiet(f1, h1);
		} else {
			move_piece_quiet(g8, e8);
			move_piece_quiet(f8, h8);
		}
		break;
	case OOO:
		if (C == WHITE) {
			move_piece_quiet(c1, e1);
			move_piece_quiet(d1, a1);
		} else {
			move_piece_quiet(c8, e8);
			move_piece_quiet(d8, a8);
		}
		break;
	case EN_PASSANT:
		move_piece_quiet(m.to(), m.from());
		put_piece(make_piece(~C, PAWN), m.to() + relative_dir<C>(SOUTH));

		pocket[C][PAWN]--;

		if (promoted & SQUARE_BB[m.to()]) {
			promoted &= ~SQUARE_BB[m.to()];
			promoted |= SQUARE_BB[m.from()];
		}
		break;
	case PR_KNIGHT:
	case PR_BISHOP:
	case PR_ROOK:
	case PR_QUEEN:
		remove_piece(m.to());
		put_piece(make_piece(C, PAWN), m.from());

		promoted &= ~SQUARE_BB[m.to()];
		break;
	case PC_KNIGHT:
	case PC_BISHOP:
	case PC_ROOK:
	case PC_QUEEN:
		remove_piece(m.to());
		put_piece(make_piece(C, PAWN), m.from());
		put_piece(history.back().captured, m.to());

		if (history.back().promoted) {
			pocket[C][PAWN]--;
		} else {
			pocket[C][type_of(history.back().captured)]--;
			promoted &= ~SQUARE_BB[m.to()];
		}
		break;
	case CAPTURE:
		move_piece_quiet(m.to(), m.from());
		put_piece(history.back().captured, m.to());

		if (history.back().promoted) {
			pocket[C][PAWN]--;

			if (promoted & SQUARE_BB[m.to()])
				promoted |= SQUARE_BB[m.from()];
			else
				promoted |= SQUARE_BB[m.to()];
		} else {
			pocket[C][type_of(history.back().captured)]--;

			if (promoted & SQUARE_BB[m.to()]) {
				promoted &= ~SQUARE_BB[m.to()];
				promoted |= SQUARE_BB[m.from()];
			}
		}
		break;
	case DROP_PAWN:
	case DROP_KNIGHT:
	case DROP_BISHOP:
	case DROP_ROOK:
	case DROP_QUEEN:
		pocket[C][type_of(board[m.to()])]++;
		remove_piece(m.to());
		break;
	}

	side_to_play = ~side_to_play;
	history.pop_back();
	//--game_ply;
}


//Generates all legal moves in a position for the given side. Advances the move pointer and returns it.
template<Color Us>
Move* Position::generate_legals(Move* list) {
	constexpr Color Them = ~Us;

	const Bitboard us_bb = all_pieces<Us>();
	const Bitboard them_bb = all_pieces<Them>();
	const Bitboard all = us_bb | them_bb;

	const Square our_king = bsf(bitboard_of(Us, KING));
	const Square their_king = bsf(bitboard_of(Them, KING));

	const Bitboard our_diag_sliders = diagonal_sliders<Us>();
	const Bitboard their_diag_sliders = diagonal_sliders<Them>();
	const Bitboard our_orth_sliders = orthogonal_sliders<Us>();
	const Bitboard their_orth_sliders = orthogonal_sliders<Them>();

	//General purpose bitboards for attacks, masks, etc.
	Bitboard b1, b2, b3;
	
	//Squares that our king cannot move to
	Bitboard danger = 0;

	//For each enemy piece, add all of its attacks to the danger bitboard
	danger |= pawn_attacks<Them>(bitboard_of(Them, PAWN)) | attacks<KING>(their_king, all);
	
	b1 = bitboard_of(Them, KNIGHT); 
	while (b1) danger |= attacks<KNIGHT>(pop_lsb(&b1), all);
	
	b1 = their_diag_sliders;
	//all ^ SQUARE_BB[our_king] is written to prevent the king from moving to squares which are 'x-rayed'
	//by enemy bishops and queens
	while (b1) danger |= attacks<BISHOP>(pop_lsb(&b1), all ^ SQUARE_BB[our_king]);
	
	b1 = their_orth_sliders;
	//all ^ SQUARE_BB[our_king] is written to prevent the king from moving to squares which are 'x-rayed'
	//by enemy rooks and queens
	while (b1) danger |= attacks<ROOK>(pop_lsb(&b1), all ^ SQUARE_BB[our_king]);

	//The king can move to all of its surrounding squares, except ones that are attacked, and
	//ones that have our own pieces on them
	b1 = attacks<KING>(our_king, all) & ~(us_bb | danger);
	list = make<QUIET>(our_king, b1 & ~them_bb, list);
	list = make<CAPTURE>(our_king, b1 & them_bb, list);

	//The capture mask filters destination squares to those that contain an enemy piece that is checking the 
	//king and must be captured
	Bitboard capture_mask;
	
	//The quiet mask filter destination squares to those where pieces must be moved to block an incoming attack 
	//to the king
	Bitboard quiet_mask;

	//The squares where pieces can be dropped
	Bitboard drop_mask;
	
	//A general purpose square for storing destinations, etc.
	Square s;

	//Checkers of each piece type are identified by:
	//1. Projecting attacks FROM the king square
	//2. Intersecting this bitboard with the enemy bitboard of that piece type
	checkers = attacks<KNIGHT>(our_king, all) & bitboard_of(Them, KNIGHT)
		| pawn_attacks<Us>(our_king) & bitboard_of(Them, PAWN);
	
	//Here, we identify slider checkers and pinners simultaneously, and candidates for such pinners 
	//and checkers are represented by the bitboard <candidates>
	Bitboard candidates = attacks<ROOK>(our_king, them_bb) & their_orth_sliders
		| attacks<BISHOP>(our_king, them_bb) & their_diag_sliders;

	pinned = 0;
	while (candidates) {
		s = pop_lsb(&candidates);
		b1 = SQUARES_BETWEEN_BB[our_king][s] & us_bb;
		
		//Do the squares in between the enemy slider and our king contain any of our pieces?
		//If not, add the slider to the checker bitboard
		if (b1 == 0) checkers ^= SQUARE_BB[s];
		//If there is only one of our pieces between them, add our piece to the pinned bitboard 
		else if ((b1 & b1 - 1) == 0) pinned ^= b1;
	}

	//This makes it easier to mask pieces
	const Bitboard not_pinned = ~pinned;

	switch (sparse_pop_count(checkers)) {
	case 2:
		//If there is a double check, the only legal moves are king moves out of check
		return list;
	case 1: {
		//It's a single check!
		
		Square checker_square = bsf(checkers);

		switch (board[checker_square]) {
		case make_piece(Them, PAWN):
			//If the checker is a pawn, we must check for e.p. moves that can capture it
			//This evaluates to true if the checking piece is the one which just double pushed
			if (checkers == shift<relative_dir<Us>(SOUTH)>(SQUARE_BB[history.back().epsq])) {
				//b1 contains our pawns that can capture the checker e.p.
				b1 = pawn_attacks<Them>(history.back().epsq) & bitboard_of(Us, PAWN) & not_pinned;
				while (b1) *list++ = Move(pop_lsb(&b1), history.back().epsq, EN_PASSANT);
			}
			//FALL THROUGH INTENTIONAL
		case make_piece(Them, KNIGHT):
			//If the checker is either a pawn or a knight, the only legal moves are to capture
			//the checker. Only non-pinned pieces can capture it
			b1 = attackers_from<Us>(checker_square, all) & not_pinned;
			while (b1) *list++ = Move(pop_lsb(&b1), checker_square, CAPTURE);

			return list;
		default:
			//We must capture the checking piece
			capture_mask = checkers;
			
			//...or we can block it since it is guaranteed to be a slider
			//by moving a piece in between...
			quiet_mask = SQUARES_BETWEEN_BB[our_king][checker_square];

			//...or dropping a piece in between
			drop_mask = SQUARES_BETWEEN_BB[our_king][checker_square];
			break;
		}

		break;
	}

	default:
		//We can capture any enemy piece
		capture_mask = them_bb;
		
		//...and we can play a quiet move to any square which is not occupied
		quiet_mask = ~all;

		if (history.back().epsq != NO_SQUARE) {
			//b1 contains our pawns that can perform an e.p. capture
			b2 = pawn_attacks<Them>(history.back().epsq) & bitboard_of(Us, PAWN);
			b1 = b2 & not_pinned;
			while (b1) {
				s = pop_lsb(&b1);
				
				//This piece of evil bit-fiddling magic prevents the infamous 'pseudo-pinned' e.p. case,
				//where the pawn is not directly pinned, but on moving the pawn and capturing the enemy pawn
				//e.p., a rook or queen attack to the king is revealed
				
				/*
				.nbqkbnr
				ppp.pppp
				........
				r..pP..K
				........
				........
				PPPP.PPP
				RNBQ.BNR
				
				Here, if white plays exd5 e.p., the black rook on a5 attacks the white king on h5 
				*/
				
				if ((sliding_attacks(our_king, all ^ SQUARE_BB[s]
					^ shift<relative_dir<Us>(SOUTH)>(SQUARE_BB[history.back().epsq]),
					MASK_RANK[rank_of(our_king)]) &
					their_orth_sliders) == 0)
						*list++ = Move(s, history.back().epsq, EN_PASSANT);
			}
			
			//Pinned pawns can only capture e.p. if they are pinned diagonally and the e.p. square is in line with the king 
			b1 = b2 & pinned & LINE[history.back().epsq][our_king];
			if (b1) {
				*list++ = Move(bsf(b1), history.back().epsq, EN_PASSANT);
			}
		}

		//Only add castling if:
		//1. The king and the rook have both not moved
		//2. No piece is attacking between the the rook and the king
		//3. The king is not in check
		if (!((history.back().entry & oo_mask<Us>()) | ((all | danger) & oo_blockers_mask<Us>())))
			*list++ = Us == WHITE ? Move(e1, g1, OO) : Move(e8, g8, OO);
		if (!((history.back().entry & ooo_mask<Us>()) |
			((all | (danger & ~ignore_ooo_danger<Us>())) & ooo_blockers_mask<Us>())))
			*list++ = Us == WHITE ? Move(e1, c1, OOO) : Move(e8, c8, OOO);

		//For each pinned rook, bishop or queen...
		b1 = ~(not_pinned | bitboard_of(Us, KNIGHT));
		while (b1) {
			s = pop_lsb(&b1);
			
			//...only include attacks that are aligned with our king, since pinned pieces
			//are constrained to move in this direction only
			b2 = attacks(type_of(board[s]), s, all) & LINE[our_king][s];
			list = make<QUIET>(s, b2 & quiet_mask, list);
			list = make<CAPTURE>(s, b2 & capture_mask, list);
		}

		//For each pinned pawn...
		b1 = ~not_pinned & bitboard_of(Us, PAWN);
		while (b1) {
			s = pop_lsb(&b1);

			if (rank_of(s) == relative_rank<Us>(RANK7)) {
				//Quiet promotions are impossible since the square in front of the pawn will
				//either be occupied by the king or the pinner, or doing so would leave our king
				//in check
				b2 = pawn_attacks<Us>(s) & capture_mask & LINE[our_king][s];
				list = make<PROMOTION_CAPTURES>(s, b2, list);
			}
			else {
				b2 = pawn_attacks<Us>(s) & them_bb & LINE[s][our_king];
				list = make<CAPTURE>(s, b2, list);
				
				//Single pawn pushes
				b2 = shift<relative_dir<Us>(NORTH)>(SQUARE_BB[s]) & ~all & LINE[our_king][s];
				//Double pawn pushes (only pawns on rank 3/6 are eligible)
				b3 = shift<relative_dir<Us>(NORTH)>(b2 &
					MASK_RANK[relative_rank<Us>(RANK3)]) & ~all & LINE[our_king][s];
				list = make<QUIET>(s, b2, list);
				list = make<DOUBLE_PUSH>(s, b3, list);
			}
		}
		
		//Pinned knights cannot move anywhere, so we're done with pinned pieces!

		//Pieces in pocket can be dropped on all empty spaces
		drop_mask = ~(all_pieces<WHITE>() | all_pieces<BLACK>());

		break;
	}

	//Non-pinned knight moves
	b1 = bitboard_of(Us, KNIGHT) & not_pinned;
	while (b1) {
		s = pop_lsb(&b1);
		b2 = attacks<KNIGHT>(s, all);
		list = make<QUIET>(s, b2 & quiet_mask, list);
		list = make<CAPTURE>(s, b2 & capture_mask, list);
	}

	//Non-pinned bishops and queens
	b1 = our_diag_sliders & not_pinned;
	while (b1) {
		s = pop_lsb(&b1);
		b2 = attacks<BISHOP>(s, all);
		list = make<QUIET>(s, b2 & quiet_mask, list);
		list = make<CAPTURE>(s, b2 & capture_mask, list);
	}

	//Non-pinned rooks and queens
	b1 = our_orth_sliders & not_pinned;
	while (b1) {
		s = pop_lsb(&b1);
		b2 = attacks<ROOK>(s, all);
		list = make<QUIET>(s, b2 & quiet_mask, list);
		list = make<CAPTURE>(s, b2 & capture_mask, list);
	}

	//b1 contains non-pinned pawns which are not on the last rank
	b1 = bitboard_of(Us, PAWN) & not_pinned & ~MASK_RANK[relative_rank<Us>(RANK7)];
	
	//Single pawn pushes
	b2 = shift<relative_dir<Us>(NORTH)>(b1) & ~all;
	
	//Double pawn pushes (only pawns on rank 3/6 are eligible)
	b3 = shift<relative_dir<Us>(NORTH)>(b2 & MASK_RANK[relative_rank<Us>(RANK3)]) & quiet_mask;
	
	//We & this with the quiet mask only later, as a non-check-blocking single push does NOT mean that the 
	//corresponding double push is not blocking check either.
	b2 &= quiet_mask;

	while (b2) {
		s = pop_lsb(&b2);
		*list++ = Move(s - relative_dir<Us>(NORTH), s, QUIET);
	}

	while (b3) {
		s = pop_lsb(&b3);
		*list++ = Move(s - relative_dir<Us>(NORTH_NORTH), s, DOUBLE_PUSH);
	}

	//Pawn captures
	b2 = shift<relative_dir<Us>(NORTH_WEST)>(b1) & capture_mask;
	b3 = shift<relative_dir<Us>(NORTH_EAST)>(b1) & capture_mask;

	while (b2) {
		s = pop_lsb(&b2);
		*list++ = Move(s - relative_dir<Us>(NORTH_WEST), s, CAPTURE);
	}

	while (b3) {
		s = pop_lsb(&b3);
		*list++ = Move(s - relative_dir<Us>(NORTH_EAST), s, CAPTURE);
	}

	//b1 now contains non-pinned pawns which ARE on the last rank (about to promote)
	b1 = bitboard_of(Us, PAWN) & not_pinned & MASK_RANK[relative_rank<Us>(RANK7)];
	if (b1) {
		//Quiet promotions
		b2 = shift<relative_dir<Us>(NORTH)>(b1) & quiet_mask;
		while (b2) {
			s = pop_lsb(&b2);
			//One move is added for each promotion piece
			*list++ = Move(s - relative_dir<Us>(NORTH), s, PR_KNIGHT);
			*list++ = Move(s - relative_dir<Us>(NORTH), s, PR_BISHOP);
			*list++ = Move(s - relative_dir<Us>(NORTH), s, PR_ROOK);
			*list++ = Move(s - relative_dir<Us>(NORTH), s, PR_QUEEN);
		}

		//Promotion captures
		b2 = shift<relative_dir<Us>(NORTH_WEST)>(b1) & capture_mask;
		b3 = shift<relative_dir<Us>(NORTH_EAST)>(b1) & capture_mask;

		while (b2) {
			s = pop_lsb(&b2);
			//One move is added for each promotion piece
			*list++ = Move(s - relative_dir<Us>(NORTH_WEST), s, PC_KNIGHT);
			*list++ = Move(s - relative_dir<Us>(NORTH_WEST), s, PC_BISHOP);
			*list++ = Move(s - relative_dir<Us>(NORTH_WEST), s, PC_ROOK);
			*list++ = Move(s - relative_dir<Us>(NORTH_WEST), s, PC_QUEEN);
		}

		while (b3) {
			s = pop_lsb(&b3);
			//One move is added for each promotion piece
			*list++ = Move(s - relative_dir<Us>(NORTH_EAST), s, PC_KNIGHT);
			*list++ = Move(s - relative_dir<Us>(NORTH_EAST), s, PC_BISHOP);
			*list++ = Move(s - relative_dir<Us>(NORTH_EAST), s, PC_ROOK);
			*list++ = Move(s - relative_dir<Us>(NORTH_EAST), s, PC_QUEEN);
		}
	}

	//Dropping moves
	for (int piece = PAWN; piece <= QUEEN; piece++) {
		if (pocket[Us][piece]) {
			Bitboard to = drop_mask;
			MoveFlags drop;
			switch (piece) {
			case PAWN:
				to &= ~(MASK_RANK[RANK8] | MASK_RANK[RANK1]);
				drop = DROP_PAWN;
				break;
			case KNIGHT:
				drop = DROP_KNIGHT;
				break;
			case BISHOP:
				drop = DROP_BISHOP;
				break;
			case ROOK:
				drop = DROP_ROOK;
				break;
			case QUEEN:
				drop = DROP_QUEEN;
				break;
			}

			Square p;
			while (to) {
				p = pop_lsb(&to);
				*list++ = Move(p, p, drop);
			}
		}
	}

	return list;
}

//Generates all legal moves in a position for the given side. Advances the move pointer and returns it.
template<Color Us>
move_vector<Move> Position::generate_legals() {
	constexpr Color Them = ~Us;

	move_vector<Move> list;
	list.reserve(200);

	const Bitboard us_bb = all_pieces<Us>();
	const Bitboard them_bb = all_pieces<Them>();
	const Bitboard all = us_bb | them_bb;

	const Square our_king = bsf(bitboard_of(Us, KING));
	const Square their_king = bsf(bitboard_of(Them, KING));

	const Bitboard our_diag_sliders = diagonal_sliders<Us>();
	const Bitboard their_diag_sliders = diagonal_sliders<Them>();
	const Bitboard our_orth_sliders = orthogonal_sliders<Us>();
	const Bitboard their_orth_sliders = orthogonal_sliders<Them>();

	//General purpose bitboards for attacks, masks, etc.
	Bitboard b1, b2, b3;

	//Squares that our king cannot move to
	Bitboard danger = 0;

	//For each enemy piece, add all of its attacks to the danger bitboard
	danger |= pawn_attacks<Them>(bitboard_of(Them, PAWN)) | attacks<KING>(their_king, all);

	b1 = bitboard_of(Them, KNIGHT);
	while (b1) danger |= attacks<KNIGHT>(pop_lsb(&b1), all);

	b1 = their_diag_sliders;
	//all ^ SQUARE_BB[our_king] is written to prevent the king from moving to squares which are 'x-rayed'
	//by enemy bishops and queens
	while (b1) danger |= attacks<BISHOP>(pop_lsb(&b1), all ^ SQUARE_BB[our_king]);

	b1 = their_orth_sliders;
	//all ^ SQUARE_BB[our_king] is written to prevent the king from moving to squares which are 'x-rayed'
	//by enemy rooks and queens
	while (b1) danger |= attacks<ROOK>(pop_lsb(&b1), all ^ SQUARE_BB[our_king]);

	//The king can move to all of its surrounding squares, except ones that are attacked, and
	//ones that have our own pieces on them
	b1 = attacks<KING>(our_king, all) & ~(us_bb | danger);
	make<QUIET>(our_king, b1 & ~them_bb, list);
	make<CAPTURE>(our_king, b1 & them_bb, list);

	//The capture mask filters destination squares to those that contain an enemy piece that is checking the 
	//king and must be captured
	Bitboard capture_mask;

	//The quiet mask filter destination squares to those where pieces must be moved to block an incoming attack 
	//to the king
	Bitboard quiet_mask;

	//The squares where pieces can be dropped
	Bitboard drop_mask;

	//A general purpose square for storing destinations, etc.
	Square s;

	//Checkers of each piece type are identified by:
	//1. Projecting attacks FROM the king square
	//2. Intersecting this bitboard with the enemy bitboard of that piece type
	checkers = attacks<KNIGHT>(our_king, all) & bitboard_of(Them, KNIGHT)
		| pawn_attacks<Us>(our_king) & bitboard_of(Them, PAWN);

	//Here, we identify slider checkers and pinners simultaneously, and candidates for such pinners 
	//and checkers are represented by the bitboard <candidates>
	Bitboard candidates = attacks<ROOK>(our_king, them_bb) & their_orth_sliders
		| attacks<BISHOP>(our_king, them_bb) & their_diag_sliders;

	pinned = 0;
	while (candidates) {
		s = pop_lsb(&candidates);
		b1 = SQUARES_BETWEEN_BB[our_king][s] & us_bb;

		//Do the squares in between the enemy slider and our king contain any of our pieces?
		//If not, add the slider to the checker bitboard
		if (b1 == 0) checkers ^= SQUARE_BB[s];
		//If there is only one of our pieces between them, add our piece to the pinned bitboard 
		else if ((b1 & b1 - 1) == 0) pinned ^= b1;
	}

	//This makes it easier to mask pieces
	const Bitboard not_pinned = ~pinned;

	switch (sparse_pop_count(checkers)) {
	case 2:
		//If there is a double check, the only legal moves are king moves out of check
		return list;
	case 1:
	{
		//It's a single check!

		Square checker_square = bsf(checkers);

		switch (board[checker_square]) {
		case make_piece(Them, PAWN):
			//If the checker is a pawn, we must check for e.p. moves that can capture it
			//This evaluates to true if the checking piece is the one which just double pushed
			if (checkers == shift<relative_dir<Us>(SOUTH)>(SQUARE_BB[history.back().epsq])) {
				//b1 contains our pawns that can capture the checker e.p.
				b1 = pawn_attacks<Them>(history.back().epsq) & bitboard_of(Us, PAWN) & not_pinned;
				while (b1) list.emplace_back(pop_lsb(&b1), history.back().epsq, EN_PASSANT);
			}
			//FALL THROUGH INTENTIONAL
		case make_piece(Them, KNIGHT):
			//If the checker is either a pawn or a knight, the only legal moves are to capture
			//the checker. Only non-pinned pieces can capture it
			b1 = attackers_from<Us>(checker_square, all) & not_pinned;
			while (b1) list.emplace_back(pop_lsb(&b1), checker_square, CAPTURE);

			return list;
		default:
			//We must capture the checking piece
			capture_mask = checkers;

			//...or we can block it since it is guaranteed to be a slider
			//by moving a piece in between...
			quiet_mask = SQUARES_BETWEEN_BB[our_king][checker_square];

			//...or drop a piece in between
			drop_mask = SQUARES_BETWEEN_BB[our_king][checker_square];
			break;
		}

		break;
	}

	default:
		//We can capture any enemy piece
		capture_mask = them_bb;

		//... we can play a quiet move to any square which is not occupied
		quiet_mask = ~all;

		//... and pieces in pocket can be dropped on all empty spaces
		drop_mask = ~all;

		if (history.back().epsq != NO_SQUARE) {
			//b1 contains our pawns that can perform an e.p. capture
			b2 = pawn_attacks<Them>(history.back().epsq) & bitboard_of(Us, PAWN);
			b1 = b2 & not_pinned;
			while (b1) {
				s = pop_lsb(&b1);

				//This piece of evil bit-fiddling magic prevents the infamous 'pseudo-pinned' e.p. case,
				//where the pawn is not directly pinned, but on moving the pawn and capturing the enemy pawn
				//e.p., a rook or queen attack to the king is revealed

				/*
				.nbqkbnr
				ppp.pppp
				........
				r..pP..K
				........
				........
				PPPP.PPP
				RNBQ.BNR

				Here, if white plays exd5 e.p., the black rook on a5 attacks the white king on h5
				*/

				if ((sliding_attacks(our_king, all ^ SQUARE_BB[s]
					^ shift<relative_dir<Us>(SOUTH)>(SQUARE_BB[history.back().epsq]),
					MASK_RANK[rank_of(our_king)]) &
					their_orth_sliders) == 0)
					list.emplace_back(s, history.back().epsq, EN_PASSANT);
			}

			//Pinned pawns can only capture e.p. if they are pinned diagonally and the e.p. square is in line with the king 
			b1 = b2 & pinned & LINE[history.back().epsq][our_king];
			if (b1) {
				list.emplace_back(bsf(b1), history.back().epsq, EN_PASSANT);
			}
		}

		//Only add castling if:
		//1. The king and the rook have both not moved
		//2. No piece is attacking between the the rook and the king
		//3. The king is not in check
		if (!((history.back().entry & oo_mask<Us>()) | ((all | danger) & oo_blockers_mask<Us>())))
		{
			if (Us == WHITE)
				list.emplace_back(e1, g1, OO);
			else
				list.emplace_back(e8, g8, OO);
		}
		if (!((history.back().entry & ooo_mask<Us>()) |
			((all | (danger & ~ignore_ooo_danger<Us>())) & ooo_blockers_mask<Us>())))
		{
			if (Us == WHITE)
				list.emplace_back(e1, c1, OOO);
			else
				list.emplace_back(e8, c8, OOO);
		}

		//For each pinned rook, bishop or queen...
		b1 = ~(not_pinned | bitboard_of(Us, KNIGHT));
		while (b1) {
			s = pop_lsb(&b1);

			//...only include attacks that are aligned with our king, since pinned pieces
			//are constrained to move in this direction only
			b2 = attacks(type_of(board[s]), s, all) & LINE[our_king][s];
			make<QUIET>(s, b2 & quiet_mask, list);
			make<CAPTURE>(s, b2 & capture_mask, list);
		}

		//For each pinned pawn...
		b1 = ~not_pinned & bitboard_of(Us, PAWN);
		while (b1) {
			s = pop_lsb(&b1);

			if (rank_of(s) == relative_rank<Us>(RANK7)) {
				//Quiet promotions are impossible since the square in front of the pawn will
				//either be occupied by the king or the pinner, or doing so would leave our king
				//in check
				b2 = pawn_attacks<Us>(s) & capture_mask & LINE[our_king][s];
				make<PROMOTION_CAPTURES>(s, b2, list);
			} else {
				b2 = pawn_attacks<Us>(s) & them_bb & LINE[s][our_king];
				make<CAPTURE>(s, b2, list);

				//Single pawn pushes
				b2 = shift<relative_dir<Us>(NORTH)>(SQUARE_BB[s]) & ~all & LINE[our_king][s];
				//Double pawn pushes (only pawns on rank 3/6 are eligible)
				b3 = shift<relative_dir<Us>(NORTH)>(b2 &
					MASK_RANK[relative_rank<Us>(RANK3)]) & ~all & LINE[our_king][s];
				make<QUIET>(s, b2, list);
				make<DOUBLE_PUSH>(s, b3, list);
			}
		}

		//Pinned knights cannot move anywhere, so we're done with pinned pieces!
		break;
	}

	//Non-pinned knight moves
	b1 = bitboard_of(Us, KNIGHT) & not_pinned;
	while (b1) {
		s = pop_lsb(&b1);
		b2 = attacks<KNIGHT>(s, all);
		make<QUIET>(s, b2 & quiet_mask, list);
		make<CAPTURE>(s, b2 & capture_mask, list);
	}

	//Non-pinned bishops and queens
	b1 = our_diag_sliders & not_pinned;
	while (b1) {
		s = pop_lsb(&b1);
		b2 = attacks<BISHOP>(s, all);
		make<QUIET>(s, b2 & quiet_mask, list);
		make<CAPTURE>(s, b2 & capture_mask, list);
	}

	//Non-pinned rooks and queens
	b1 = our_orth_sliders & not_pinned;
	while (b1) {
		s = pop_lsb(&b1);
		b2 = attacks<ROOK>(s, all);
		make<QUIET>(s, b2 & quiet_mask, list);
		make<CAPTURE>(s, b2 & capture_mask, list);
	}

	//b1 contains non-pinned pawns which are not on the last rank
	b1 = bitboard_of(Us, PAWN) & not_pinned & ~MASK_RANK[relative_rank<Us>(RANK7)];

	//Single pawn pushes
	b2 = shift<relative_dir<Us>(NORTH)>(b1) & ~all;

	//Double pawn pushes (only pawns on rank 3/6 are eligible)
	b3 = shift<relative_dir<Us>(NORTH)>(b2 & MASK_RANK[relative_rank<Us>(RANK3)]) & quiet_mask;

	//We & this with the quiet mask only later, as a non-check-blocking single push does NOT mean that the 
	//corresponding double push is not blocking check either.
	b2 &= quiet_mask;

	while (b2) {
		s = pop_lsb(&b2);
		list.emplace_back(s - relative_dir<Us>(NORTH), s, QUIET);
	}

	while (b3) {
		s = pop_lsb(&b3);
		list.emplace_back(s - relative_dir<Us>(NORTH_NORTH), s, DOUBLE_PUSH);
	}

	//Pawn captures
	b2 = shift<relative_dir<Us>(NORTH_WEST)>(b1) & capture_mask;
	b3 = shift<relative_dir<Us>(NORTH_EAST)>(b1) & capture_mask;

	while (b2) {
		s = pop_lsb(&b2);
		list.emplace_back(s - relative_dir<Us>(NORTH_WEST), s, CAPTURE);
	}

	while (b3) {
		s = pop_lsb(&b3);
		list.emplace_back(s - relative_dir<Us>(NORTH_EAST), s, CAPTURE);
	}

	//b1 now contains non-pinned pawns which ARE on the last rank (about to promote)
	b1 = bitboard_of(Us, PAWN) & not_pinned & MASK_RANK[relative_rank<Us>(RANK7)];
	if (b1) {
		//Quiet promotions
		b2 = shift<relative_dir<Us>(NORTH)>(b1) & quiet_mask;
		while (b2) {
			s = pop_lsb(&b2);
			//One move is added for each promotion piece
			list.emplace_back(s - relative_dir<Us>(NORTH), s, PR_KNIGHT);
			list.emplace_back(s - relative_dir<Us>(NORTH), s, PR_BISHOP);
			list.emplace_back(s - relative_dir<Us>(NORTH), s, PR_ROOK);
			list.emplace_back(s - relative_dir<Us>(NORTH), s, PR_QUEEN);
		}

		//Promotion captures
		b2 = shift<relative_dir<Us>(NORTH_WEST)>(b1) & capture_mask;
		b3 = shift<relative_dir<Us>(NORTH_EAST)>(b1) & capture_mask;

		while (b2) {
			s = pop_lsb(&b2);
			//One move is added for each promotion piece
			list.emplace_back(s - relative_dir<Us>(NORTH_WEST), s, PC_KNIGHT);
			list.emplace_back(s - relative_dir<Us>(NORTH_WEST), s, PC_BISHOP);
			list.emplace_back(s - relative_dir<Us>(NORTH_WEST), s, PC_ROOK);
			list.emplace_back(s - relative_dir<Us>(NORTH_WEST), s, PC_QUEEN);
		}

		while (b3) {
			s = pop_lsb(&b3);
			//One move is added for each promotion piece
			list.emplace_back(s - relative_dir<Us>(NORTH_EAST), s, PC_KNIGHT);
			list.emplace_back(s - relative_dir<Us>(NORTH_EAST), s, PC_BISHOP);
			list.emplace_back(s - relative_dir<Us>(NORTH_EAST), s, PC_ROOK);
			list.emplace_back(s - relative_dir<Us>(NORTH_EAST), s, PC_QUEEN);
		}
	}

	//Dropping moves
	if (drop_mask) {
		for (int piece = PAWN; piece <= QUEEN; piece++) {
			if (pocket[Us][piece]) {
				Bitboard to = drop_mask;
				MoveFlags drop;
				switch (piece) {
				case PAWN:
					to &= ~(MASK_RANK[RANK8] | MASK_RANK[RANK1]);
					drop = DROP_PAWN;
					break;
				case KNIGHT:
					drop = DROP_KNIGHT;
					break;
				case BISHOP:
					drop = DROP_BISHOP;
					break;
				case ROOK:
					drop = DROP_ROOK;
					break;
				case QUEEN:
					drop = DROP_QUEEN;
					break;
				}

				Square p;
				while (to) {
					p = pop_lsb(&to);
					list.emplace_back(p, p, drop);
				}
			}
		}
	}

	return list;
}

//A convenience class for interfacing with legal moves, rather than using the low-level
//generate_legals() function directly. It can be iterated over.
template<Color Us>
class MoveList {
public:
	explicit MoveList(Position& p) : last(p.generate_legals<Us>(list)) {}

	const Move* begin() const { return list; }
	const Move* end() const { return last; }
	size_t size() const { return last - list; }
private:
	Move list[218];
	Move *last;
};

inline EndType Position::is_checkmate() {
	bool can_move;
	if (side_to_play == WHITE) {
		can_move = (generate_legals<WHITE>().size()) ? true : false;
	} else {
		can_move = (generate_legals<BLACK>().size()) ? true : false;
	}

	if (!can_move) {
		bool check;
		if (side_to_play == WHITE)
			check = in_check<WHITE>();
		else
			check = in_check<BLACK>();

		if (check)
			//Checkmate
			return CHECKMATE;
		else
			//Stalemate
			return STALEMATE;
	} else {
		//Not end game
		return NONE;
	}
}

inline bool Position::is_insufficient_material() {
	Bitboard mats = piece_bb[WHITE_PAWN] | piece_bb[WHITE_ROOK] | piece_bb[WHITE_QUEEN] | piece_bb[BLACK_PAWN] | piece_bb[BLACK_ROOK] | piece_bb[BLACK_QUEEN];
	if (all_pieces<WHITE>() & mats && all_pieces<BLACK>() & mats)
		return false;

	// Knights are only insufficient material if:
	// (1) We do not have any other pieces, including more than one knight.
	// (2) The opponent does not have pawns, knights, bishops or rooks.
	//     These would allow selfmate.
	if (all_pieces<WHITE>() & piece_bb[WHITE_KNIGHT]) {
		if (!(pop_count(all_pieces<WHITE>()) <= 2 && !(all_pieces<BLACK>() & ~piece_bb[BLACK_KING] & ~piece_bb[BLACK_QUEEN])))
			return false;
	}

	if (all_pieces<BLACK>() & piece_bb[BLACK_KNIGHT]) {
		if (!(pop_count(all_pieces<BLACK>()) <= 2 && !(all_pieces<WHITE>() & ~piece_bb[WHITE_KING] & ~piece_bb[WHITE_QUEEN])))
			return false;
	}

	// Bishops are only insufficient material if:
	// (1) We do not have any other pieces, including bishops of the
	//     opposite color.
	// (2) The opponent does not have bishops of the opposite color,
	//     pawns or knights.These would allow selfmate.
	if (all_pieces<WHITE>() & piece_bb[WHITE_BISHOP]) {
		bool same_color = !(piece_bb[WHITE_BISHOP] & dark_squares) || !(piece_bb[WHITE_BISHOP] & light_squares);
		if (same_color && !piece_bb[WHITE_PAWN] && !piece_bb[WHITE_KNIGHT])
			return false;
	}

	if (all_pieces<BLACK>() & piece_bb[BLACK_BISHOP]) {
		bool same_color = !(piece_bb[BLACK_BISHOP] & dark_squares) || !(piece_bb[BLACK_BISHOP] & light_squares);
		if (same_color && !piece_bb[BLACK_PAWN] && !piece_bb[BLACK_KNIGHT])
			return false;
	}

	return true;
}

inline bool Position::is_seventyfive_moves() {
	return history.back().halfmove_clock >= 150;
}

inline bool Position::is_fivefold_repetition() {
	std::string board_fen = fen_board();
	if (repetitions.contains(board_fen) && repetitions[board_fen] >= 4)
		return true;
	else
		return false;
}

//Check if reached end of game and returnes the score that the player got
template<Color Us>
inline double Position::end_score() {
	constexpr double not_ended = 0.0;
	constexpr double draw = 1e-4;
	double score;
	if (Us == side_to_play)
		score = 1.0;
	else
		score = -1.0;

	switch (is_checkmate()) {
	case CHECKMATE:
		return -score;
	case STALEMATE:
		return draw;
	}

	//if (is_insufficient_material())
	//	return draw;

	//if (is_seventyfive_moves())
	//	return draw;

	if (is_fivefold_repetition())
		return draw;

	return not_ended;
}




//Zobrist keys for each piece and each square
//Used to incrementally update the hash key of a position
uint64_t zobrist::zobrist_table[NPIECES][NSQUARES];

//Initializes the zobrist table with random 64-bit numbers
void zobrist::initialise_zobrist_keys() {
	PRNG rng(70026072);
	for (int i = 0; i < NPIECES; i++)
		for (int j = 0; j < NSQUARES; j++)
			zobrist::zobrist_table[i][j] = rng.rand<uint64_t>();
}

//Pretty-prints the position (including FEN and hash key)
std::ostream& operator<< (std::ostream& os, const Position& p) {
	const char* s = "   +---+---+---+---+---+---+---+---+\n";
	const char* t = "     A   B   C   D   E   F   G   H\n";
	const char* poc = "   +---+---+---+---+---+---+---+---+     +---+---+---+\n";
	os << t;
	for (int i = 56; i >= 0; i -= 8) {
		int rank = i / 8 + 1;
		if (rank >= 1 && rank <= 7)
			os << poc << " " << rank << " ";
		else
			os << s << " " << rank << " ";
		for (int j = 0; j < 8; j++)
			os << "| " << PIECE_STR[p.board[i + j]] << " ";
		if (rank >= 2 && rank <= 7) {
			os << "| " << rank << "   |";

			switch (rank) {
			case 7:
				os << "   | w | b |\n";
				break;
			case 6:
				os << " p |";
				if (p.pocket[WHITE][PAWN] >= 10)
					os << " " << p.pocket[WHITE][PAWN] << "|";
				else
					os << " " << p.pocket[WHITE][PAWN] << " |";
				if (p.pocket[BLACK][PAWN] >= 10)
					os << " " << p.pocket[BLACK][PAWN] << "|\n";
				else
					os << " " << p.pocket[BLACK][PAWN] << " |\n";
				break;
			case 5:
				os << " n |";
				os << " " << p.pocket[WHITE][KNIGHT] << " |";
				os << " " << p.pocket[BLACK][KNIGHT] << " |\n";
				break;
			case 4:
				os << " b |";
				os << " " << p.pocket[WHITE][BISHOP] << " |";
				os << " " << p.pocket[BLACK][BISHOP] << " |\n";
				break;
			case 3:
				os << " r |";
				os << " " << p.pocket[WHITE][ROOK] << " |";
				os << " " << p.pocket[BLACK][ROOK] << " |\n";
				break;
			case 2:
				os << " q |";
				os << " " << p.pocket[WHITE][QUEEN] << " |";
				os << " " << p.pocket[BLACK][QUEEN] << " |\n";
				break;
			default:
				os << "\n";
				break;
			}
		} else
			os << "| " << rank << "\n";
	}
	os << s;
	os << t << "\n";

	os << "FEN: " << p.fen() << "\n";
	os << "FEN hash: " << p.fen_hash() << "\n";
	os << "Hash: 0x" << std::hex << p.hash << std::dec << "\n";

	return os;
}

//Returns the FEN (Forsyth-Edwards Notation) representation of the position
std::string Position::fen() const {
	std::ostringstream fen;
	int empty;

	for (int i = 56; i >= 0; i -= 8) {
		empty = 0;
		for (int j = 0; j < 8; j++) {
			Piece p = board[i + j];
			if (p == NO_PIECE) empty++;
			else {
				fen << (empty == 0 ? "" : std::to_string(empty))
					<< PIECE_STR[p];
				empty = 0;
			}
		}

		if (empty != 0) fen << empty;
		if (i > 0) fen << '/';
	}

	//Add the pockets
	fen << "[";
	for (int color = 0; color <= 1; color++) {
		for (int piece = PAWN; piece <= QUEEN; piece++) {
			if (pocket[color][piece]) {
				for (int i = 0; i < pocket[color][piece]; i++) {
					if (color)
						fen << PIECE_STR[8 + piece];
					else
						fen << PIECE_STR[piece];
				}
			}
		}
	}
	fen << "]";

	fen << (side_to_play == WHITE ? " w " : " b ")
		<< (history.back().entry & WHITE_OO_MASK ? "" : "K")
		<< (history.back().entry & WHITE_OOO_MASK ? "" : "Q")
		<< (history.back().entry & BLACK_OO_MASK ? "" : "k")
		<< (history.back().entry & BLACK_OOO_MASK ? "" : "q")
		<< (history.back().entry & ALL_CASTLING_MASK ? "-" : "") << " "
		<< (history.back().epsq == NO_SQUARE ? "-" : SQSTR[history.back().epsq]);

	fen << " " << history.back().halfmove_clock << " " << history.back().fullmove_number;

	return fen.str();
}

//Updates a position according to an FEN string
void Position::set(const std::string& fen, Position& p) {
	int square = a8;
	for (char ch : fen.substr(0, fen.find('['))) {
		if (isdigit(ch))
			square += (ch - '0') * EAST;
		else if (ch == '/')
			square += 2 * SOUTH;
		else
			p.put_piece(Piece(PIECE_STR.find(ch)), Square(square++));
	}

	for (int piece = PAWN; piece <= QUEEN; piece++) {
		p.pocket[WHITE][piece] = 0;
		p.pocket[BLACK][piece] = 0;
	}
	if (fen.find('[') + 1 != fen.find(']')) {
		for (char ch : fen.substr(fen.find('[') + 1)) {
			if (ch == ']')
				break;

			switch (ch) {
			case 'P':
				p.pocket[WHITE][PAWN]++;
					break;
			case 'N':
				p.pocket[WHITE][KNIGHT]++;
					break;
			case 'B':
				p.pocket[WHITE][BISHOP]++;
					break;
			case 'R':
				p.pocket[WHITE][ROOK]++;
				break;
			case 'Q':
				p.pocket[WHITE][QUEEN]++;
				break;
			case 'p':
				p.pocket[BLACK][PAWN]++;
				break;
			case 'n':
				p.pocket[BLACK][KNIGHT]++;
				break;
			case 'b':
				p.pocket[BLACK][BISHOP]++;
				break;
			case 'r':
				p.pocket[BLACK][ROOK]++;
				break;
			case 'q':
				p.pocket[BLACK][QUEEN]++;
				break;
			default:
				break;
			}
		}
	}

	std::string info = fen.substr(fen.find(' ') + 1);
	char color = info[0];
	p.side_to_play = color == 'w' ? WHITE : BLACK;

	info = info.substr(info.find(' ') + 1);
	p.history.back().entry = ALL_CASTLING_MASK;
	for (int i = 0; i < info.size(); i++) {
		if (info[i] == '-' || info[i] == ' ')
			break;

		switch (info[i]) {
		case 'K':
			p.history.back().entry &= ~WHITE_OO_MASK;
			break;
		case 'Q':
			p.history.back().entry &= ~WHITE_OOO_MASK;
			break;
		case 'k':
			p.history.back().entry &= ~BLACK_OO_MASK;
			break;
		case 'q':
			p.history.back().entry &= ~BLACK_OOO_MASK;
			break;
		}
	}

	info = info.substr(info.find(' ') + 1);
	if (info[0] != '-') {
		std::string enpass = info.substr(0, 2);
		for (int square = a1; square <= h8; square++) {
			if (SQSTR[square] == enpass) {
				p.history.back().epsq = (Square)square;
				break;
			}
		}
	}

	info = info.substr(info.find(' ') + 1);
	std::ostringstream num;
	for (int i = 0; i < info.size(); i++) {
		if (info[i] == ' ')
			break;

		num << info[i];
	}
	p.history.back().halfmove_clock = std::stoi(num.str());

	info = info.substr(info.find(' ') + 1);
	std::ostringstream num2;
	for (int i = 0; i < info.size(); i++) {
		num2 << info[i];
	}
	p.history.back().fullmove_number = std::stoi(num2.str());

	p.repetitions.clear();
	p.repetitions[p.fen_board()] = 0;
}

//Returns the string representation that can be used as a hash
std::string Position::fen_hash() const {
	std::ostringstream fen;
	int empty;

	for (int i = 56; i >= 0; i -= 8) {
		empty = 0;
		for (int j = 0; j < 8; j++) {
			Piece p = board[i + j];
			if (p == NO_PIECE) empty++;
			else {
				fen << (empty == 0 ? "" : std::to_string(empty))
					<< PIECE_STR[p];
				empty = 0;
			}
		}

		if (empty != 0) fen << empty;
		if (i > 0) fen << '/';
	}

	//Add the pockets
	fen << "[";
	for (int color = 0; color <= 1; color++) {
		for (int piece = PAWN; piece <= QUEEN; piece++) {
			if (pocket[color][piece]) {
				for (int i = 0; i < pocket[color][piece]; i++) {
					if (color)
						fen << PIECE_STR[8 + piece];
					else
						fen << PIECE_STR[piece];
				}
			}
		}
	}
	fen << "]" << (side_to_play == WHITE ? " w " : " b ") << history.back().halfmove_clock << " " << history.back().fullmove_number;
	return fen.str();
}

std::string Position::fen_board() const {
	std::ostringstream fen;
	int empty;

	for (int i = 56; i >= 0; i -= 8) {
		empty = 0;
		for (int j = 0; j < 8; j++) {
			Piece p = board[i + j];
			if (p == NO_PIECE) empty++;
			else {
				fen << (empty == 0 ? "" : std::to_string(empty))
					<< PIECE_STR[p];
				empty = 0;
			}
		}

		if (empty != 0) fen << empty;
		if (i > 0) fen << '/';
	}

	//Add the pockets
	fen << "[";
	for (int color = 0; color <= 1; color++) {
		for (int piece = PAWN; piece <= QUEEN; piece++) {
			if (pocket[color][piece]) {
				for (int i = 0; i < pocket[color][piece]; i++) {
					if (color)
						fen << PIECE_STR[8 + piece];
					else
						fen << PIECE_STR[piece];
				}
			}
		}
	}
	fen << "]";
	return fen.str();
}


//Moves a piece to a (possibly empty) square on the board and updates the hash
void Position::move_piece(Square from, Square to) {
	hash ^= zobrist::zobrist_table[board[from]][from] ^ zobrist::zobrist_table[board[from]][to]
		^ zobrist::zobrist_table[board[to]][to];
	Bitboard mask = SQUARE_BB[from] | SQUARE_BB[to];
	piece_bb[board[from]] ^= mask;
	piece_bb[board[to]] &= ~mask;
	board[to] = board[from];
	board[from] = NO_PIECE;
}

//Moves a piece to an empty square. Note that it is an error if the <to> square contains a piece
void Position::move_piece_quiet(Square from, Square to) {
	hash ^= zobrist::zobrist_table[board[from]][from] ^ zobrist::zobrist_table[board[from]][to];
	piece_bb[board[from]] ^= (SQUARE_BB[from] | SQUARE_BB[to]);
	board[to] = board[from];
	board[from] = NO_PIECE;
}

#endif

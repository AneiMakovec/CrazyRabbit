#ifndef SURGE_TYPES_HPP
#define SURGE_TYPES_HPP
#pragma warning(disable : 26812)

#include <cstdint>
#include <string>
#include <ostream>
#include <iostream>
#include <sstream>
#include <vector>

const size_t NCOLORS = 2;
enum Color : int {
	WHITE, BLACK, NO_COLOR = -1
};

//Inverts the color (WHITE -> BLACK) and (BLACK -> WHITE)
constexpr Color operator~(Color c) {
	return Color(c ^ BLACK);
}

enum EndType : int {
	NONE,
	CHECKMATE,
	STALEMATE
};

const size_t NDIRS = 8;
enum Direction : int {
	NORTH = 8, NORTH_EAST = 9, EAST = 1, SOUTH_EAST = -7,
	SOUTH = -8, SOUTH_WEST = -9, WEST = -1, NORTH_WEST = 7,
	NORTH_NORTH = 16, SOUTH_SOUTH = -16
};

enum SquareDirection {
	UP,
	DOWN,
	LEFT,
	RIGHT,
	UP_LEFT,
	UP_RIGHT,
	DOWN_LEFT,
	DOWN_RIGHT
};

const int knight_move_offsets[8]{
	15, 17, -17, -15, 6, -10, 10, -6
};

constexpr uint16_t KNIGHT_MOVE_START = 56U;
constexpr uint16_t UNDERPROMOTION_MOVE_START = 64U;
constexpr uint16_t DROP_MOVE_START = 76U;
constexpr uint16_t MOVES_PER_SQUARE = 81U;

const size_t NPIECE_TYPES = 6;
enum PieceType : int {
	PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
};

//PIECE_STR[piece] is the algebraic chess representation of that piece
const std::string PIECE_STR = "PNBRQK~>pnbrqk.";

//The FEN of the starting position
const std::string DEFAULT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -";

//The Kiwipete position, used for perft debugging
const std::string KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -";


const size_t NPIECES = 15;
enum Piece : int {
	WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING,
	BLACK_PAWN = 8, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING,
	NO_PIECE
};

constexpr Piece make_piece(Color c, PieceType pt) {
	return Piece((c << 3) + pt);
}

constexpr PieceType type_of(Piece pc) {
	return PieceType(pc & 0b111);
}

constexpr Color color_of(Piece pc) {
	return Color((pc & 0b1000) >> 3);
}



typedef uint64_t Bitboard;

const size_t NSQUARES = 64;
enum Square : uint16_t {
	a1, b1, c1, d1, e1, f1, g1, h1,
	a2, b2, c2, d2, e2, f2, g2, h2,
	a3, b3, c3, d3, e3, f3, g3, h3,
	a4, b4, c4, d4, e4, f4, g4, h4,
	a5, b5, c5, d5, e5, f5, g5, h5,
	a6, b6, c6, d6, e6, f6, g6, h6,
	a7, b7, c7, d7, e7, f7, g7, h7,
	a8, b8, c8, d8, e8, f8, g8, h8,
	NO_SQUARE
};

inline Square& operator++(Square& s) { return s = Square(int(s) + 1); }
constexpr Square operator+(Square s, Direction d) { return Square(int(s) + int(d)); }
constexpr Square operator-(Square s, Direction d) { return Square(int(s) - int(d)); }
inline Square& operator+=(Square& s, Direction d) { return s = s + d; }
inline Square& operator-=(Square& s, Direction d) { return s = s - d; }

enum File : int {
	AFILE, BFILE, CFILE, DFILE, EFILE, FFILE, GFILE, HFILE
};

const std::string FILE_STR = "abcdefgh";

enum Rank : int {
	RANK1, RANK2, RANK3, RANK4, RANK5, RANK6, RANK7, RANK8
};

const std::string RANK_STR = "12345678";

extern const char* SQSTR[65];

extern const Bitboard MASK_FILE[8];
extern const Bitboard MASK_RANK[8];
extern const Bitboard MASK_DIAGONAL[15];
extern const Bitboard MASK_ANTI_DIAGONAL[15];
extern const Bitboard SQUARE_BB[65];

extern void print_bitboard(Bitboard b);

extern const Bitboard k1;
extern const Bitboard k2;
extern const Bitboard k4;
extern const Bitboard kf;

extern inline int pop_count(Bitboard x);
extern inline int sparse_pop_count(Bitboard x);
extern inline Square pop_lsb(Bitboard* b);

constexpr int DEBRUIJN64[64] = {
	0, 47,  1, 56, 48, 27,  2, 60,
   57, 49, 41, 37, 28, 16,  3, 61,
   54, 58, 35, 52, 50, 42, 21, 44,
   38, 32, 29, 23, 17, 11,  4, 62,
   46, 55, 26, 59, 40, 36, 15, 53,
   34, 51, 20, 43, 31, 22, 10, 45,
   25, 39, 14, 33, 19, 30,  9, 24,
   13, 18,  8, 12,  7,  6,  5, 63
};
extern const Bitboard MAGIC;
extern constexpr Square bsf(Bitboard b);

constexpr Rank rank_of(Square s) { return Rank(s >> 3); }
constexpr File file_of(Square s) { return File(s & 0b111); }
constexpr int diagonal_of(Square s) { return 7 + rank_of(s) - file_of(s); }
constexpr int anti_diagonal_of(Square s) { return rank_of(s) + file_of(s); }
constexpr Square create_square(File f, Rank r) { return Square(r << 3 | f); }

//Shifts a bitboard in a particular direction. There is no wrapping, so bits that are shifted of the edge are lost 
template<Direction D>
constexpr Bitboard shift(Bitboard b) {
	return D == NORTH ? b << 8 : D == SOUTH ? b >> 8
		: D == NORTH + NORTH ? b << 16 : D == SOUTH + SOUTH ? b >> 16
		: D == EAST ? (b & ~MASK_FILE[HFILE]) << 1 : D == WEST ? (b & ~MASK_FILE[AFILE]) >> 1
		: D == NORTH_EAST ? (b & ~MASK_FILE[HFILE]) << 9 
		: D == NORTH_WEST ? (b & ~MASK_FILE[AFILE]) << 7
		: D == SOUTH_EAST ? (b & ~MASK_FILE[HFILE]) >> 7 
		: D == SOUTH_WEST ? (b & ~MASK_FILE[AFILE]) >> 9
		: 0;	
}

//Returns the actual rank from a given side's perspective (e.g. rank 1 is rank 8 from Black's perspective)
template<Color C>
constexpr Rank relative_rank(Rank r) {
	return C == WHITE ? r : Rank(RANK8 - r);
}

//Returns the actual direction from a given side's perspective (e.g. North is South from Black's perspective)
template<Color C>
constexpr Direction relative_dir(Direction d) {
	return Direction(C == WHITE ? d : -d);
}

//Default data structure to hold generated legal moves.
template<class T>
class move_vector : public std::vector<T>
{
public:
	long n_visits = 0L;
	double end_score = 0.0;
};


//The type of the move
enum MoveFlags : uint16_t {
	QUIET, DOUBLE_PUSH,
	OO, OOO,
	CAPTURE,
	EN_PASSANT,
	PR_KNIGHT, PR_BISHOP, PR_ROOK, PR_QUEEN,
	PC_KNIGHT, PC_BISHOP, PC_ROOK, PC_QUEEN,
	DROP_PAWN, DROP_KNIGHT, DROP_BISHOP, DROP_ROOK, DROP_QUEEN,
	PROMOTIONS, PROMOTION_CAPTURES, DROPS
};

class Move {
private:
	//The internal representation of the move
	Square from_square;
	Square to_square;
	MoveFlags move_flags;
	uint16_t move_hash;
public:
	double policy;
	double Q_value;
	long n_visits;

	//Defaults to a null move (a1a1)
	inline Move() : from_square(NO_SQUARE), to_square(NO_SQUARE), move_flags(DROPS), move_hash(0U), policy(0.0), Q_value(0.0), n_visits(0L) {}

	inline Move(Square from, Square to, MoveFlags flags) : policy(0.0), Q_value(0.0), n_visits(0L)
	{
		from_square = from;
		to_square = to;
		move_flags = flags;
		move_hash = encode();
	}

	inline Move(const std::string uci) : policy(0.0), Q_value(0.0), n_visits(0L)
	{
		move_flags = DROPS;
		if (uci[1] == '@') {
			//Drop
			to_square = create_square(File(uci[2] - 'a'), Rank(uci[3] - '1'));
			from_square = to_square;

			switch (uci[0]) {
			case 'P':
				move_flags = DROP_PAWN;
				break;
			case 'N':
				move_flags = DROP_KNIGHT;
				break;
			case 'B':
				move_flags = DROP_BISHOP;
				break;
			case 'R':
				move_flags = DROP_ROOK;
				break;
			case 'Q':
				move_flags = DROP_QUEEN;
				break;
			}
		} else {
			from_square = create_square(File(uci[0] - 'a'), Rank(uci[1] - '1'));
			to_square = create_square(File(uci[2] - 'a'), Rank(uci[3] - '1'));

			if (uci.size() == 5) {
				//Promotion
				switch (uci[4]) {
				case 'n':
					move_flags = PR_KNIGHT;
					break;
				case 'b':
					move_flags = PR_BISHOP;
					break;
				case 'r':
					move_flags = PR_ROOK;
					break;
				case 'q':
					move_flags = PR_QUEEN;
					break;
				default:
					break;
				}
			}
		}

		bool is_encoded_string = false;
		if (uci.size() >= 5)
		{
			switch (uci[4])
			{
			case 'n':
			case 'b':
			case 'r':
			case 'q':
				break;
			default:
				is_encoded_string = true;
				break;
			}

			if (is_encoded_string)
				move_flags = static_cast<MoveFlags>(std::stoi(uci.substr(4)));
		}

		move_hash = encode();
	}

	inline Square to() const { return to_square; }
	inline Square from() const { return from_square; }
	inline MoveFlags flags() const { return move_flags; }
	inline uint16_t hash() const { return move_hash; }

	//void operator=(Move m) { move = m.move; }
	//bool operator==(Move a) const { return from_square == a.from() && to_square == a.to() && move_flags == a.flags(); }
	//bool operator!=(Move a) const { return from_square != a.from() || to_square != a.to() || move_flags != a.flags(); }
	bool operator==(Move a) const { return move_hash == a.hash() && move_flags == a.flags(); }
	bool operator!=(Move a) const { return move_hash != a.hash() || move_flags != a.flags(); }

	//Used for move encoding. Returns direction of knight movement if it is a knight move
	static inline int is_knight_move(const int from_square, const int to_square) {
		for (int direction = 0; direction < 8; direction++) {
			if (from_square + knight_move_offsets[direction] == to_square)
				return direction + 1;
		}
		return 0;
	}

	//Used for move encoding. Returns direction and distance between two squares 
	static inline std::pair<int, int> square_movement(const int from_square, const int to_square) {
		int from_rank = from_square >> 3;
		int from_file = from_square & 7;
		int to_rank = to_square >> 3;
		int to_file = to_square & 7;

		if (from_file == to_file) {
			// up
			if (to_rank > from_rank)
				return std::pair<int, int>(UP, to_rank - from_rank);
			// down
			else
				return std::pair<int, int>(DOWN, from_rank - to_rank);
		} else if (from_rank == to_rank) {
			// right
			if (to_file > from_file)
				return std::pair<int, int>(RIGHT, to_file - from_file);
			// left
			else
				return std::pair<int, int>(LEFT, from_file - to_file);
		} else if (to_rank > from_rank) {
			// up right
			if (to_file > from_file)
				return std::pair<int, int>(UP_RIGHT, std::max(to_rank - from_rank, to_file - from_file));
			// up left
			else
				return std::pair<int, int>(UP_LEFT, std::max(to_rank - from_rank, from_rank - to_rank));
		} else {
			// down right
			if (to_file > from_file)
				return std::pair<int, int>(DOWN_RIGHT, std::max(from_rank - to_rank, to_file - from_file));
			// down left
			else
				return std::pair<int, int>(DOWN_LEFT, std::max(from_rank - to_rank, from_file - to_file));
		}
	}

	inline uint16_t encode() {
		uint16_t encoded_move;
		uint16_t flag = move_flags;
		if (flag >= DROP_PAWN && flag <= DROP_QUEEN) {
			// encode drop
			encoded_move = DROP_MOVE_START + (flag - DROP_PAWN);
		} else if (flag >= PR_KNIGHT && flag <= PC_QUEEN) {
			// encode underpromotion
			if (flag < PC_KNIGHT)
				flag -= PR_KNIGHT;
			else
				flag -= PC_KNIGHT;
			encoded_move = UNDERPROMOTION_MOVE_START + (std::abs(to_square - from_square) - 7) + (flag * 3);
		} else if (int kn_direction = is_knight_move(from_square, to_square)) {
			// encode knight moves
			encoded_move = KNIGHT_MOVE_START + (kn_direction - 1);
		} else {
			std::pair<int, int> square_info = square_movement(from_square, to_square);

			encoded_move = square_info.first * 7 + (square_info.second - 1);
		}

		// now return the full encoded action
		return from_square * MOVES_PER_SQUARE + encoded_move;
	}

	inline std::string to_encoded_string();
};

//Adds, to the move pointer all moves of the form (from, s), where s is a square in the bitboard to
template<MoveFlags F = QUIET>
inline Move *make(Square from, Bitboard to, Move *list) {
	while (to) *list++ = Move(from, pop_lsb(&to), F);
	return list;
}

//Adds, to the move pointer all quiet promotion moves of the form (from, s), where s is a square in the bitboard to
template<>
inline Move *make<PROMOTIONS>(Square from, Bitboard to, Move *list) {
	Square p;
	while (to) {
		p = pop_lsb(&to);
		*list++ = Move(from, p, PR_KNIGHT);
		*list++ = Move(from, p, PR_BISHOP);
		*list++ = Move(from, p, PR_ROOK);
		*list++ = Move(from, p, PR_QUEEN);
	}
	return list;
}

//Adds, to the move pointer all capture promotion moves of the form (from, s), where s is a square in the bitboard to
template<>
inline Move* make<PROMOTION_CAPTURES>(Square from, Bitboard to, Move* list) {
	Square p;
	while (to) {
		p = pop_lsb(&to);
		*list++ = Move(from, p, PC_KNIGHT);
		*list++ = Move(from, p, PC_BISHOP);
		*list++ = Move(from, p, PC_ROOK);
		*list++ = Move(from, p, PC_QUEEN);
	}
	return list;
}

//Adds, to the move pointer all moves of the form (from, s), where s is a square in the bitboard to
template<MoveFlags F = QUIET>
inline void make(Square from, Bitboard to, std::vector<Move>& list) {
	while (to) list.emplace_back(from, pop_lsb(&to), F);
}

//Adds, to the move pointer all quiet promotion moves of the form (from, s), where s is a square in the bitboard to
template<>
inline void make<PROMOTIONS>(Square from, Bitboard to, std::vector<Move>& list) {
	Square p;
	while (to) {
		p = pop_lsb(&to);
		list.emplace_back(from, p, PR_KNIGHT);
		list.emplace_back(from, p, PR_BISHOP);
		list.emplace_back(from, p, PR_ROOK);
		list.emplace_back(from, p, PR_QUEEN);
	}
}

//Adds, to the move pointer all capture promotion moves of the form (from, s), where s is a square in the bitboard to
template<>
inline void make<PROMOTION_CAPTURES>(Square from, Bitboard to, std::vector<Move>& list) {
	Square p;
	while (to) {
		p = pop_lsb(&to);
		list.emplace_back(from, p, PC_KNIGHT);
		list.emplace_back(from, p, PC_BISHOP);
		list.emplace_back(from, p, PC_ROOK);
		list.emplace_back(from, p, PC_QUEEN);
	}
}

extern std::ostream& operator<<(std::ostream& os, const Move& m);

//The white king and kingside rook
const Bitboard WHITE_OO_MASK = 0x90;
//The white king and queenside rook
const Bitboard WHITE_OOO_MASK = 0x11;

//Squares between the white king and kingside rook
const Bitboard WHITE_OO_BLOCKERS_AND_ATTACKERS_MASK = 0x60;
//Squares between the white king and queenside rook
const Bitboard WHITE_OOO_BLOCKERS_AND_ATTACKERS_MASK = 0xe;

//The black king and kingside rook
const Bitboard BLACK_OO_MASK = 0x9000000000000000;
//The black king and queenside rook
const Bitboard BLACK_OOO_MASK = 0x1100000000000000;
//Squares between the black king and kingside rook
const Bitboard BLACK_OO_BLOCKERS_AND_ATTACKERS_MASK = 0x6000000000000000;
//Squares between the black king and queenside rook
const Bitboard BLACK_OOO_BLOCKERS_AND_ATTACKERS_MASK = 0xE00000000000000;

//The white king, white rooks, black king and black rooks
const Bitboard ALL_CASTLING_MASK = 0x9100000000000091;

template<Color C> constexpr Bitboard oo_mask() { return C == WHITE ? WHITE_OO_MASK : BLACK_OO_MASK; }
template<Color C> constexpr Bitboard ooo_mask() { return C == WHITE ? WHITE_OOO_MASK : BLACK_OOO_MASK; }

template<Color C>
constexpr Bitboard oo_blockers_mask() { 
	return C == WHITE ? WHITE_OO_BLOCKERS_AND_ATTACKERS_MASK :
		BLACK_OO_BLOCKERS_AND_ATTACKERS_MASK; 
}

template<Color C>
constexpr Bitboard ooo_blockers_mask() {
	return C == WHITE ? WHITE_OOO_BLOCKERS_AND_ATTACKERS_MASK :
		BLACK_OOO_BLOCKERS_AND_ATTACKERS_MASK;
}
	
template<Color C> constexpr Bitboard ignore_ooo_danger() { return C == WHITE ? 0x2 : 0x200000000000000; }

//Lookup tables of square names in algebraic chess notation
const char* SQSTR[65] = {
	"a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
	"a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
	"a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
	"a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
	"a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
	"a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
	"a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
	"a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
	"None"
};

//All masks have been generated from a Java program

//Precomputed file masks
const Bitboard MASK_FILE[8] = {
	0x101010101010101, 0x202020202020202, 0x404040404040404, 0x808080808080808,
	0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
};

//Precomputed rank masks
const Bitboard MASK_RANK[8] = {
	0xff, 0xff00, 0xff0000, 0xff000000,
	0xff00000000, 0xff0000000000, 0xff000000000000, 0xff00000000000000
};

//Precomputed diagonal masks
const Bitboard MASK_DIAGONAL[15] = {
	0x80, 0x8040, 0x804020,
	0x80402010, 0x8040201008, 0x804020100804,
	0x80402010080402, 0x8040201008040201, 0x4020100804020100,
	0x2010080402010000, 0x1008040201000000, 0x804020100000000,
	0x402010000000000, 0x201000000000000, 0x100000000000000,
};

//Precomputed anti-diagonal masks
const Bitboard MASK_ANTI_DIAGONAL[15] = {
	0x1, 0x102, 0x10204,
	0x1020408, 0x102040810, 0x10204081020,
	0x1020408102040, 0x102040810204080, 0x204081020408000,
	0x408102040800000, 0x810204080000000, 0x1020408000000000,
	0x2040800000000000, 0x4080000000000000, 0x8000000000000000,
};

//Precomputed square masks
const Bitboard SQUARE_BB[65] = {
	0x1, 0x2, 0x4, 0x8,
	0x10, 0x20, 0x40, 0x80,
	0x100, 0x200, 0x400, 0x800,
	0x1000, 0x2000, 0x4000, 0x8000,
	0x10000, 0x20000, 0x40000, 0x80000,
	0x100000, 0x200000, 0x400000, 0x800000,
	0x1000000, 0x2000000, 0x4000000, 0x8000000,
	0x10000000, 0x20000000, 0x40000000, 0x80000000,
	0x100000000, 0x200000000, 0x400000000, 0x800000000,
	0x1000000000, 0x2000000000, 0x4000000000, 0x8000000000,
	0x10000000000, 0x20000000000, 0x40000000000, 0x80000000000,
	0x100000000000, 0x200000000000, 0x400000000000, 0x800000000000,
	0x1000000000000, 0x2000000000000, 0x4000000000000, 0x8000000000000,
	0x10000000000000, 0x20000000000000, 0x40000000000000, 0x80000000000000,
	0x100000000000000, 0x200000000000000, 0x400000000000000, 0x800000000000000,
	0x1000000000000000, 0x2000000000000000, 0x4000000000000000, 0x8000000000000000,
	0x0
};

//Prints the bitboard, little-endian format
void print_bitboard(Bitboard b) {
	for (int i = 56; i >= 0; i -= 8) {
		for (int j = 0; j < 8; j++)
			std::cout << (char)(((b >> (i + j)) & 1) + '0') << " ";
		std::cout << "\n";
	}
	std::cout << "\n";
}

const Bitboard k1 = 0x5555555555555555;
const Bitboard k2 = 0x3333333333333333;
const Bitboard k4 = 0x0f0f0f0f0f0f0f0f;
const Bitboard kf = 0x0101010101010101;

const Bitboard light_squares = 0x55AA55AA55AA55AA;
const Bitboard dark_squares = 0xAA55AA55AA55AA55;

const Bitboard center_squares = 0x00003C3C3C3C0000;

const Bitboard white_side = 0x00000000FFFFFFFF;
const Bitboard black_side = 0xFFFFFFFF00000000;

//Returns number of set bits in the bitboard
inline int pop_count(Bitboard x) {
	x = x - ((x >> 1) & k1);
	x = (x & k2) + ((x >> 2) & k2);
	x = (x + (x >> 4)) & k4;
	x = (x * kf) >> 56;
	return int(x);
}

//Returns number of set bits in the bitboard. Faster than pop_count(x) when the bitboard has few set bits
inline int sparse_pop_count(Bitboard x) {
	int count = 0;
	while (x) {
		count++;
		x &= x - 1;
	}
	return count;
}

const Bitboard MAGIC = 0x03f79d71b4cb0a89;

//Returns the index of the least significant bit in the bitboard, and removes the bit from the bitboard
inline Square pop_lsb(Bitboard* b) {
	int lsb = bsf(*b);
	*b &= *b - 1;
	return Square(lsb);
}

//Returns the index of the least significant bit in the bitboard
constexpr Square bsf(Bitboard b) {
	return Square(DEBRUIJN64[MAGIC * (b ^ (b - 1)) >> 58]);
}

//Returns the representation of the move type in algebraic chess notation. (capture) is used for debugging
const char* MOVE_TYPESTR[19] = {
	"QUIET", "DOUBLE_PUSH", "OO", "OOO", "CAPTURE", "EN_PASSANT", "PR_KNIGHT", "PR_BISHOP", "PR_ROOK", "PR_QUEEN",
	"PC_KNIGHT", "PC_BISHOP", "PC_ROOK", "PC_QUEEN", "DROP_PAWN", "DROP_KNIGHT", "DROP_BISHOP", "DROP_ROOK", "DROP_QUEEN"
};

//Prints the move
//For example: e5d6 (capture); a7a8R; O-O
std::ostream& operator<<(std::ostream& os, const Move& m) {
	int flag = m.flags();
	if (flag >= DROP_PAWN && flag <= DROP_QUEEN) {
		// encode drop
		switch (flag - DROP_PAWN) {
		case PAWN:
			os << "P@" << SQSTR[m.from()];
			break;
		case KNIGHT:
			os << "N@" << SQSTR[m.from()];
			break;
		case BISHOP:
			os << "B@" << SQSTR[m.from()];
			break;
		case ROOK:
			os << "R@" << SQSTR[m.from()];
			break;
		case QUEEN:
			os << "Q@" << SQSTR[m.from()];
			break;
		default:
			os << "Unknown drop";
			break;
		}
	} else if (flag >= PR_KNIGHT && flag <= PC_QUEEN) {
		// encode underpromotion
		if (flag < PC_KNIGHT)
			flag -= PR_KNIGHT;
		else
			flag -= PC_KNIGHT;
		os << SQSTR[m.from()] << SQSTR[m.to()];

		switch (flag + 1) {
		case KNIGHT:
			os << "n";
			break;
		case BISHOP:
			os << "b";
			break;
		case ROOK:
			os << "r";
			break;
		case QUEEN:
			os << "q";
			break;
		default:
			os << "Unknown promotion";
			break;
		}
	} else {
		os << SQSTR[m.from()] << SQSTR[m.to()];
	}
	return os;
}

inline std::string Move::to_encoded_string()
{
	std::ostringstream enc_string;
	enc_string << *this << move_flags;
	return enc_string.str();
}

#endif

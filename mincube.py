"""
A barebones cube class for maximising data loading speed during training.
"""

from itertools import chain

FACES = "ULFRBD" # This matches the order they appear in the Cube object
MOVES = [face + magnitude for face in FACES for magnitude in ("", "2", "'")]
CHAR_TO_NUMBER = {"2" : 2, "'" : 3}
NUMBER_TO_CHAR = {1 : "", 2 : "2", 3 : "'"}

class Move:

    """
    HTM moves for 3x3 cubes only.
    """

    def __init__(self, move):
        if len(move) == 1:
            self.letter = move
            self.number = 1
        else:
            self.letter = move[0]
            self.number = CHAR_TO_NUMBER[move[1]]

    def invert(self):
        """
        Converts a move to its inverse. This changes the object in place rather
        than returning a new object.
        """
        self.number = 4 - self.number

    def inverse(self):
        """
        Returns a new Move object with the inverse value.
        """
        return Move(self.letter + NUMBER_TO_CHAR[self.number])

    def __str__(self):
        return self.letter + NUMBER_TO_CHAR[self.number]


class Algorithm:

    def __init__(self, alg):
        self.moves = [Move(m) for m in alg.split()]

    def invert(self):
        self.moves = self.moves[::-1]
        for move in self.moves:
            move.invert()

    def __str__(self):
        return " ".join([str(m) for m in self.moves])

MOVES_OBJS = [Move(m) for m in MOVES]

class Cube:

    def __init__(self, alg=None):
        self.cube = [[[colour for _ in range(3)]
                                for _ in range(3)]
                                  for colour in range(6)]
        if alg:
            if isinstance(alg, str):
                # alg is passed as a string
                self.apply_alg(Algorithm(alg))
            else:
                # alg is passed as an Algorithm object
                self.apply_alg(alg)

    def is_solved(self):
        # In the value network test, it wasn't always detecting if the cube
        # was solved, so maybe this isn't working...
        return all([self.cube[face][0] == self.cube[face][1] == self.cube[face][2]
                        for face in range(6)])

    def apply_alg(self, alg):
        for move in alg.moves:
            self.apply_move(move)

    def apply_move(self, move):
        if isinstance(move, str):
            move = Move(move)
        # Rotate the stickers on the face
        # First cycle is corners, second is edges
        f = FACES.index(move.letter)
        if move.number == 1:   # 90 degrees clockwise
            self._cycle_stickers((f, 0, 0), (f, 0, 2), (f, 2, 2), (f, 2, 0))
            self._cycle_stickers((f, 0, 1), (f, 1, 2), (f, 2, 1), (f, 1, 0))
        elif move.number == 2: # 180 degrees
            self._cycle_stickers((f, 0, 0), (f, 2, 2))
            self._cycle_stickers((f, 0, 2), (f, 2, 0))
            self._cycle_stickers((f, 0, 1), (f, 2, 1))
            self._cycle_stickers((f, 1, 2), (f, 1, 0))
        elif move.number == 3: # 90 degrees anti-clockwise
            self._cycle_stickers((f, 2, 0), (f, 2, 2), (f, 0, 2), (f, 0, 0))
            self._cycle_stickers((f, 1, 0), (f, 2, 1), (f, 1, 2), (f, 0, 1))

        # Move the stickers on the sides
        if move.letter == "U":
            if move.number == 1:
                self._cycle_rows((4, 0), (3, 0), (2, 0), (1, 0))
            elif move.number == 2:
                self._cycle_rows((1, 0), (3, 0))
                self._cycle_rows((2, 0), (4, 0))
            elif move.number == 3:
                self._cycle_rows((1, 0), (2, 0), (3, 0), (4, 0))

        elif move.letter == "D":
            if move.number == 1:
                self._cycle_rows((1, 2), (2, 2), (3, 2), (4, 2))
            elif move.number == 2:
                self._cycle_rows((1, 2), (3, 2))
                self._cycle_rows((2, 2), (4, 2))
            elif move.number == 3:
                self._cycle_rows((4, 2), (3, 2), (2, 2), (1, 2))

        elif move.letter == "L":
            if move.number == 1:
                self._cycle_stickers((0, 0, 0), (2, 0, 0), (5, 0, 0), (4, 2, 2))
                self._cycle_stickers((0, 1, 0), (2, 1, 0), (5, 1, 0), (4, 1, 2))
                self._cycle_stickers((0, 2, 0), (2, 2, 0), (5, 2, 0), (4, 0, 2))
            elif move.number == 2:
                self._cycle_stickers((0, 0, 0), (5, 0, 0))
                self._cycle_stickers((2, 0, 0), (4, 2, 2))
                self._cycle_stickers((0, 1, 0), (5, 1, 0))
                self._cycle_stickers((2, 1, 0), (4, 1, 2))
                self._cycle_stickers((0, 2, 0), (5, 2, 0))
                self._cycle_stickers((2, 2, 0), (4, 0, 2))
            elif move.number == 3:
                self._cycle_stickers((4, 2, 2), (5, 0, 0), (2, 0, 0), (0, 0, 0))
                self._cycle_stickers((4, 1, 2), (5, 1, 0), (2, 1, 0), (0, 1, 0))
                self._cycle_stickers((4, 0, 2), (5, 2, 0), (2, 2, 0), (0, 2, 0))

        elif move.letter == "F":
            if move.number == 1:
                self._cycle_stickers((0, 2, 0), (3, 0, 0), (5, 0, 2), (1, 2, 2))
                self._cycle_stickers((0, 2, 1), (3, 1, 0), (5, 0, 1), (1, 1, 2))
                self._cycle_stickers((0, 2, 2), (3, 2, 0), (5, 0, 0), (1, 0, 2))
            elif move.number == 2:
                self._cycle_stickers((0, 2, 0), (5, 0, 2))
                self._cycle_stickers((3, 0, 0), (1, 2, 2))
                self._cycle_stickers((0, 2, 1), (5, 0, 1))
                self._cycle_stickers((3, 1, 0), (1, 1, 2))
                self._cycle_stickers((0, 2, 2), (5, 0, 0))
                self._cycle_stickers((3, 2, 0), (1, 0, 2))
            elif move.number == 3:
                self._cycle_stickers((1, 2, 2), (5, 0, 2), (3, 0, 0), (0, 2, 0))
                self._cycle_stickers((1, 1, 2), (5, 0, 1), (3, 1, 0), (0, 2, 1))
                self._cycle_stickers((1, 0, 2), (5, 0, 0), (3, 2, 0), (0, 2, 2))

        elif move.letter == "R":
            if move.number == 1:
                self._cycle_stickers((0, 2, 2), (4, 0, 0), (5, 2, 2), (2, 2, 2))
                self._cycle_stickers((0, 1, 2), (4, 1, 0), (5, 1, 2), (2, 1, 2))
                self._cycle_stickers((0, 0, 2), (4, 2, 0), (5, 0, 2), (2, 0, 2))
            elif move.number == 2:
                self._cycle_stickers((0, 2, 2), (5, 2, 2))
                self._cycle_stickers((4, 0, 0), (2, 2, 2))
                self._cycle_stickers((0, 1, 2), (5, 1, 2))
                self._cycle_stickers((4, 1, 0), (2, 1, 2))
                self._cycle_stickers((0, 0, 2), (5, 0, 2))
                self._cycle_stickers((4, 2, 0), (2, 0, 2))
            elif move.number == 3:
                self._cycle_stickers((2, 2, 2), (5, 0, 2), (4, 2, 0), (0, 2, 2))
                self._cycle_stickers((2, 1, 2), (5, 1, 2), (4, 1, 0), (0, 1, 2))
                self._cycle_stickers((2, 0, 2), (5, 2, 2), (4, 0, 0), (0, 0, 2))

        elif move.letter == "B":
            if move.number == 1:
                self._cycle_stickers((0, 0, 2), (1, 0, 0), (5, 2, 0), (3, 2, 2))
                self._cycle_stickers((0, 0, 1), (1, 1, 0), (5, 2, 1), (3, 1, 2))
                self._cycle_stickers((0, 0, 0), (1, 2, 0), (5, 2, 2), (3, 0, 2))
            elif move.number == 2:
                self._cycle_stickers((0, 0, 2), (5, 2, 0))
                self._cycle_stickers((1, 0, 0), (3, 2, 2))
                self._cycle_stickers((0, 0, 1), (5, 2, 1))
                self._cycle_stickers((1, 1, 0), (3, 1, 2))
                self._cycle_stickers((0, 0, 0), (5, 2, 2))
                self._cycle_stickers((1, 2, 0), (3, 0, 2))
            elif move.number == 3:
                self._cycle_stickers((3, 2, 2), (5, 2, 0), (1, 0, 0), (0, 0, 2))
                self._cycle_stickers((3, 1, 2), (5, 2, 1), (1, 1, 0), (0, 0, 1))
                self._cycle_stickers((3, 0, 2), (5, 2, 2), (1, 2, 0), (0, 0, 0))

    def _cycle_stickers(self, *args):
        # Store the colour of the last sticker
        temp = self.cube[args[-1][0]][args[-1][1]][args[-1][2]]
        # Set stickers 2 to n to the colour of the next sticker in the cycle
        for i in range(len(args) - 1, 0, -1):
            self.cube[args[i][0]][args[i][1]][args[i][2]] = self.cube[args[i-1][0]][args[i-1][1]][args[i-1][2]]
        # Set the colour of the first sticker to the value stored from the first sticker
        self.cube[args[0][0]][args[0][1]][args[0][2]] = temp

    def _cycle_rows(self, *args):
        # Works like _cycle_stickers but moves whole rows rather than stickers
        temp = self.cube[args[-1][0]][args[-1][1]]
        for i in range(len(args) - 1, 0, -1):
            self.cube[args[i][0]][args[i][1]] = self.cube[args[i-1][0]][args[i-1][1]]
        self.cube[args[0][0]][args[0][1]] = temp

    def legal_moves(self):
        return MOVES_OBJS

    def hashable_repr(self):
        return "".join([str(sticker) for sticker in chain(*chain(*self.cube))])

    def __str__(self):
        indent = " " * (((len(str(self.cube[0][0][0])) + 1) * 3) + 1)
        face_one = "\n".join([indent + " ".join(
                        [str(self.cube[0][row][col]) for col in range(3)])
                            for row in range(3)])
        faces_two_to_five = "\n".join("  ".join(
                    " ".join([str(self.cube[face][row][col]) for col in range(3)])
                        for face in range(1, 5))
                            for row in range(3))
        face_six = "\n".join([indent + " ".join([
                        str(self.cube[5][row][col]) for col in range(3)])
                            for row in range(3)])
        s = "\n\n".join([face_one, faces_two_to_five, face_six])
        return s

if __name__ == "__main__":
    c = Cube()
    print(c.hashable_repr())
    print(c)

"""
Representation of an NxNxN cube.

Currently aiming to use separate move and algorithm classes. It may be
quicker to switch these to just directly interpreting strings, or even some sort
of numeric encoding if speed gains are really needed. Similarly, np arrays
could replace the lists used in the cube representation.

TODO: implement __eq__ and __ne__ so that == and != work for these classes.
"""

import re
import random

# This pattern is quite strict, and won't accept slice moves
# or using the 2' suffix.
VALID_MOVE = re.compile("""
                [2-9]?      # Number of layers, absence indicates one layer
                [BDFLRU]    # Face being turned
                ['2]?       # Magnitude, absence indicates 90 degress cw
                """,
                re.VERBOSE)


class Move():

    def __init__(self, move):
        if is_valid_move(move):
            self.move = move
        else:
            raise ValueError("Invalid move passed to constructor: {}".format(move))

    def inverse(self):
        """
        Returns the inverse of the move: that is, it returns the move that will
        undo the
        """
        if self.move.endswith("2"):
            return self.move
        elif self.move.endswith("'"):
            return self.move[:-1]
        else:
            return self.move + "'"

    def __str__(self):
        return str(self.move)


class Algorithm():

    def __init__(self, alg):
        # `alg` should be a space-separated string of moves
        moves = alg.split()
        self.moves = [Move(move) for move in moves]

    def inverse(self):
        return Algorithm(" ".join([move.inverse() for move in reversed(self.moves)]))

    def __repr__(self):
        return " ".join([str(move) for move in self.moves])


class Cube():

    FACES = "BDFLRU".split()
    MAGNITUDES = " '2".split()

    def __init__(self, size=3):
        """
        Initalizes a cube of side length `side`

        could keep a move history and scramble associated with a cube...
        probably not necessary though.
        """
        size = size // 1
        if size < 2:
            raise ValueError("size of cube cannot be less than 2")
        self.size = size
        # Cube state, as a 6xNxN array of sticker colours
        # Initialized to the solved state
        self.state = [[[colour for y in range(size)]
                            for x in range(size)]
                                for colour in range(6)]

    # TODO: Write a constructor that generates a cube in a given state


    def do_move(self, move):
        pass

    def apply_alg(self, alg):
        for move in alg:
            self.do_move(move)

    def scramble(self):
        self.apply_alg(get_scramble())

    def is_solved(self):
        for side in self.state:
            # Check if the first row is all the same colour, and then
            # check if the other two rows are the same as the first
            if len(set(side[0])) != 1 and len(set(side)) != 1:
                return False
        return True

    def _cycle_stickers(self, *args):
        # Using David Adams idea for how to apply moves
        """
        Takes a list of tuples specifying sticker positions and cycles them
        by one place.
        """
        # Get the last sticker from args
        temp = self.state[args[len(args)-1][0]]
                         [args[len(args)-1][1]]
                         [args[len(args)-1][2]]


    def __str__(self):
        # Apparently ''.join(list_comp) is the most efficient way to build strings
        # in python, but I still feel like this
        # could be flattened a bit more (fewer joins) to improve speed.

        # Performance currently seems to be a little over
        # 0.0001(n) seconds for cube size n
        # It doesn't grow linearly, but it's close enough for n < 10

        indent = " " * ((self.size * 2) + 1)

        face_one = "\n".join([indent + " ".join(
                        [str(self.state[0][row][col]) for col in range(self.size)])
                            for row in range(self.size)])

        faces_two_to_five = "\n".join("  ".join(
                    " ".join([str(self.state[face][row][col]) for col in range(self.size)])
                        for face in range(1, 5))
                            for row in range(self.size))

        face_six = "\n".join([indent + " ".join([
                        str(self.state[5][row][col]) for col in range(self.size)])
                            for row in range(self.size)])

        s = "\n\n".join([face_one, faces_two_to_five, face_six])
        return s


def is_valid_move(move):
    return VALID_MOVE.fullmatch(move)

def get_scramble(length=25):
    """
    For testing, this just produces a random move scramble.
    """
    scramble = [random.choice(Cube.FACES) + random.choice(Cube.MAGNITUDES)
                    for move in range(length)]
    return Algorithm(" ".join(scramble))



def _test():
    c = Cube()
    print(c)
    a = Algorithm("R U R' U'")
    print(a)
    # Inverse isn't printing with spaces in...
    # Likely to be something to do with the way str and repr are being called internally
    b = a.inverse()
    print(b)


if __name__ == "__main__":
    _test()

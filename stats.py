import numpy as np
import time

def show_move_distributions():
    
    path_base = "/Users/Zak/Desktop/MScCS/Project/DataGenerator/data/"

    counts = dict()
    from training import MOVE_ENCODING

    for scramble_len in range(1, 21):
        t = time.time()
        path = path_base + str(scramble_len) + "_move_scrambles.txt"

        counts[scramble_len] = [0 for _ in range(len(MOVE_ENCODING))]
        with open(path, "r") as f:
            for scramble in f:
                try:
                    i = 0
                    depth = 1
                    while i < len(scramble) - 1:
                        move = scramble[i:i+3]
                        index = MOVE_ENCODING.index(move.strip())
                        counts[depth][index] += 1
                        i += 3
                        depth += 1
                except ValueError:
                    print(scramble)
                    print("Error occurred at i = {}".format(i))
                    quit()
        print("Finished length {} in {}s".format(scramble_len, format(time.time() - t, ".2f")))

    with open(path_base + "mixed_length_scrambles.txt", "r") as f:
        t = time.time()
        counts[21] = [0 for _ in range(len(MOVE_ENCODING))]
        for scramble in f:
            i = 0
            depth = 1
            while i < len(scramble) - 1:
                move = scramble[i:i+2]
                index = MOVE_ENCODING.index(move.strip())
                counts[21][index] += 1
                i += 3
                depth += 1
        print("Finished mixed scrambles in {}s".format(format(time.time() - t, ".2f")))
        print(counts[21])


    print("Depth ", end="")
    for move in MOVE_ENCODING:
        print(str(move).ljust(6), end="")
    print()
    for depth in counts:
        print(str(depth).ljust(6), end="")
        total = sum(counts[depth])
        for move in counts[depth]:
            print(format((move / total), ".3f"), end=" ")
        print()

    all_scrambles = [0 for _ in range(len(MOVE_ENCODING))]
    for c in counts:
        for i in range(len(all_scrambles)):
            all_scrambles[i] += counts[c][i]
    print("All   ", end="")
    total = sum(all_scrambles)
    for move in all_scrambles:
        print(format((move / total), ".3f"), end=" ")
    print()

"""
Functions to turn cube objects into feature vectors.

These features are currently hard-coded for 3x3.
"""

import numpy as np

def _stickers_feature(cube):
    # Feature of shape (faces, rows, cols, colours)
    feature = np.zeros((6, 3, 3, 6), dtype=np.bool)
    for i, face in enumerate(cube.cube):
        for j, row in enumerate(face):
            for k, colour in enumerate(row):
                feature[i][j][k][colour] = 1
    return feature.flatten()

def _two_by_two_blocks_feature(cube):
    feature = np.zeros((6, 4), dtype=np.bool)
    for face in range(6):
        # x and y indicate the position of the block's upper left corner
        for block_no, (x, y) in enumerate(((x, y) for x in range(2)
                                                  for y in range(2))):
            f = cube.cube[face]
            feature[face][block_no] = (f[x][y]
                                        == f[x][y+1]
                                        == f[x+1][y]
                                        == f[x+1][y+1])
    return feature.flatten()

def _corner_edge_block_feature(cube):
    feature = np.zeros((6, 8), dtype=np.bool)
    for face in range(6):
        f = cube.cube[face]
        # UL corner + U edge
        feature[face][0] = f[0][0] == f[0][1]
        # UL corner + L edge
        feature[face][1] = f[0][0] == f[1][0]
        # UR corner + U edge
        feature[face][2] = f[0][2] == f[0][1]
        # UR corner + R edge
        feature[face][3] = f[0][2] == f[1][2]
        # DL corner + D edge
        feature[face][4] = f[2][0] == f[2][1]
        # DL corner + L edge
        feature[face][5] = f[2][0] == f[1][2]
        # DR corner + D edge
        feature[face][6] = f[2][2] == f[2][1]
        # DR corner + R edge
        feature[face][7] = f[2][2] == f[1][2]
    return feature.flatten()

def _centre_edge_block_feature(cube):
    # A block is formed when one of the numbered stickers below (its block_no)
    # matches the centre colour, C.
    # -------------
    # |   | 0 |   |
    # -------------
    # | 1 | C | 2 |
    # -------------
    # |   | 3 |   |
    # -------------

    feature = np.zeros((6, 4), dtype=np.bool)
    for face in range(6):
        f = cube.cube[face]
        centre = f[1][1]
        for block_no, edge in enumerate(((0,1), (1,0), (1,2), (2,1))):
            feature[face][block_no] = centre == f[edge[0]][edge[1]]
    return feature.flatten()

def get_features(cube):
    return np.concatenate([
                _stickers_feature(cube),
                _two_by_two_blocks_feature(cube),
                _corner_edge_block_feature(cube),
                _centre_edge_block_feature(cube)
                ])

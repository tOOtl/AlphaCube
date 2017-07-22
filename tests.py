"""
Tests for AlphaCube.
"""

import unittest
import cube

class TestMoveClass(unittest.TestCase):

    def test_face_moves(self):
        m = cube.Move("B")
        self.assertEqual(str(m), "B")
        m = cube.Move("D'")
        self.assertEqual(str(m), "D'")
        m = cube.Move("F2")
        self.assertEqual(str(m), "F2")
        m = cube.Move("L")
        self.assertEqual(str(m), "L")
        m = cube.Move("R'")
        self.assertEqual(str(m), "R'")
        m = cube.Move("U2")
        self.assertEqual(str(m), "U2")

    def test_wedge_moves(self):
        m = cube.Move("2R")
        self.assertEqual(str(m), "2R")
        m = cube.Move("4D'")
        self.assertEqual(str(m), "4D'")
        m = cube.Move("5U2")
        self.assertEqual(str(m), "5U2")

    def test_invalid_moves(self):
        with self.assertRaises(ValueError, msg="Invalid face"):
            m = cube.Move("A")
        with self.assertRaises(ValueError, msg="Invalid depth"):
            m = cube.Move("10R")
        with self.assertRaises(ValueError, msg="Invalid depth"):
            m = cube.Move("1R")
        with self.assertRaises(ValueError, msg="Invalid magnitude"):
            m = cube.Move("R3")
        with self.assertRaises(ValueError, msg="Two moves passed"):
            m = cube.Move("RR")
        with self.assertRaises(ValueError, msg="Magnitude of 2' used"):
            m = cube.Move("R2'")
        with self.assertRaises(ValueError, msg="Rw syntax used"):
            m = cube.Move("Rw")


class TestAlgorithmClass(unittest.TestCase):

    def test_construction(self):
        pass

    def test_methods(self):
        # Test inverse method
        sexy = cube.Algorithm("R U R' U'")
        sexy_inv = cube.Algorithm("U R U' R'")
        self.assertEqual(sexy.inverse(), sexy_inv)
        t_perm = cube.Algorithm("R U R' U' R' F R2 U' R' U' R U R' F'")
        t_perm_inv = cube.Algorithm("F R U' R' U R U R2 F' R U R U' R'")
        self.assertEqual(t_perm.inverse(), t_perm_inv)



if __name__ == "__main__":
    unittest.main()

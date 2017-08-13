"""
Tests for AlphaCube.
"""

import unittest
import mincube as rubiks

class TestMoveClass(unittest.TestCase):

    def test_face_moves(self):
        m = rubiks.Move("B")
        self.assertEqual(str(m), "B")
        m = rubiks.Move("D'")
        self.assertEqual(str(m), "D'")
        m = rubiks.Move("F2")
        self.assertEqual(str(m), "F2")
        m = rubiks.Move("L")
        self.assertEqual(str(m), "L")
        m = rubiks.Move("R'")
        self.assertEqual(str(m), "R'")
        m = rubiks.Move("U2")
        self.assertEqual(str(m), "U2")


class TestAlgorithmClass(unittest.TestCase):

    def test_construction(self):
        pass

    def test_methods(self):
        pass


class TestCubeClass(unittest.TestCase):

    def test_construction(self):
        # Basic construction
        c = rubiks.Cube()
        self.assertEqual(c.hashable_repr(),
                        "000000000"
                      + "111111111"
                      + "222222222"
                      + "333333333"
                      + "444444444"
                      + "555555555"
                        )
        # Alg passed as string
        c = rubiks.Cube(alg="U")
        self.assertEqual(c.hashable_repr(),
                        "000000000"
                      + "222111111"
                      + "333222222"
                      + "444333333"
                      + "111444444"
                      + "555555555"
                        )
        # Alg passed as an object
        c = rubiks.Cube(alg=rubiks.Algorithm("U"))
        self.assertEqual(c.hashable_repr(),
                        "000000000"
                      + "222111111"
                      + "333222222"
                      + "444333333"
                      + "111444444"
                      + "555555555"
                        )

    def test_U(self):
        c = rubiks.Cube(alg="U")
        self.assertEqual(c.hashable_repr(),
                        "000000000"
                      + "222111111"
                      + "333222222"
                      + "444333333"
                      + "111444444"
                      + "555555555")

    def test_U2(self):
        c = rubiks.Cube(alg="U2")
        self.assertEqual(c.hashable_repr(),
                        "000000000"
                      + "333111111"
                      + "444222222"
                      + "111333333"
                      + "222444444"
                      + "555555555")

    def test_U_prime(self):
        c = rubiks.Cube(alg="U'")
        self.assertEqual(c.hashable_repr(),
                        "000000000"
                      + "444111111"
                      + "111222222"
                      + "222333333"
                      + "333444444"
                      + "555555555")

    def test_L(self):
        c = rubiks.Cube(alg="L")
        self.assertEqual(c.hashable_repr(),
                        "400400400"
                      + "111111111"
                      + "022022022"
                      + "333333333"
                      + "445445445"
                      + "255255255")

    def test_L2(self):
        c = rubiks.Cube(alg="L2")
        self.assertEqual(c.hashable_repr(),
                        "500500500"
                      + "111111111"
                      + "422422422"
                      + "333333333"
                      + "442442442"
                      + "055055055")

    def test_L_prime(self):
        c = rubiks.Cube(alg="L'")
        self.assertEqual(c.hashable_repr(),
                        "200200200"
                      + "111111111"
                      + "522522522"
                      + "333333333"
                      + "440440440"
                      + "455455455")

    def test_F(self):
        c = rubiks.Cube(alg="F")
        self.assertEqual(c.hashable_repr(),
                        "000000111"
                      + "115115115"
                      + "222222222"
                      + "033033033"
                      + "444444444"
                      + "333555555")

    def test_F2(self):
        c = rubiks.Cube(alg="F2")
        self.assertEqual(c.hashable_repr(),
                        "000000555"
                      + "113113113"
                      + "222222222"
                      + "133133133"
                      + "444444444"
                      + "000555555")

    def test_F_prime(self):
        c = rubiks.Cube(alg="F'")
        self.assertEqual(c.hashable_repr(),
                        "000000333"
                      + "110110110"
                      + "222222222"
                      + "533533533"
                      + "444444444"
                      + "111555555")

    def test_R(self):
        c = rubiks.Cube(alg="R")
        self.assertEqual(c.hashable_repr(),
                        "002002002"
                      + "111111111"
                      + "225225225"
                      + "333333333"
                      + "044044044"
                      + "554554554")

    def test_R2(self):
        c = rubiks.Cube(alg="R2")
        self.assertEqual(c.hashable_repr(),
                        "005005005"
                      + "111111111"
                      + "224224224"
                      + "333333333"
                      + "244244244"
                      + "550550550")

    def test_R_prime(self):
        c = rubiks.Cube(alg="R'")
        self.assertEqual(c.hashable_repr(),
                        "004004004"
                      + "111111111"
                      + "220220220"
                      + "333333333"
                      + "544544544"
                      + "552552552")

    def test_B(self):
        c = rubiks.Cube(alg="B")
        self.assertEqual(c.hashable_repr(),
                        "333000000"
                      + "011011011"
                      + "222222222"
                      + "335335335"
                      + "444444444"
                      + "555555111")

    def test_B2(self):
        c = rubiks.Cube(alg="B2")
        self.assertEqual(c.hashable_repr(),
                        "555000000"
                      + "311311311"
                      + "222222222"
                      + "331331331"
                      + "444444444"
                      + "555555000")

    def test_B_prime(self):
        c = rubiks.Cube(alg="B'")
        self.assertEqual(c.hashable_repr(),
                        "111000000"
                      + "511511511"
                      + "222222222"
                      + "330330330"
                      + "444444444"
                      + "555555333")

    def test_D(self):
        c = rubiks.Cube(alg="D")
        self.assertEqual(c.hashable_repr(),
                        "000000000"
                      + "111111444"
                      + "222222111"
                      + "333333222"
                      + "444444333"
                      + "555555555")

    def test_D2(self):
        c = rubiks.Cube(alg="D2")
        self.assertEqual(c.hashable_repr(),
                        "000000000"
                      + "111111333"
                      + "222222444"
                      + "333333111"
                      + "444444222"
                      + "555555555")

    def test_D_prime(self):
        c = rubiks.Cube(alg="D'")
        self.assertEqual(c.hashable_repr(),
                        "000000000"
                      + "111111222"
                      + "222222333"
                      + "333333444"
                      + "444444111"
                      + "555555555")




if __name__ == "__main__":
    unittest.main()

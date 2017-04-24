# -*- coding: utf-8 -*-
"""
Finite Elements Coursework for Simulation and Modelling Assignment #2 2016

README:
There are two classes, FiniteElements and the assosciated FiniteElementsTest class.

The unit-testing is being run from within main() as is the code to generate
the required plots.

The code required to generate the mesh and associated functionality have been
copied from the assignment description.

@author: P.Bartram@soton.ac.uk
"""

import numpy as np
import unittest
import sympy
import FiniteElements as fe

class FiniteElementsTest(unittest.TestCase):
    ''' This class performs testing on the FiniteElements class. '''
    def setUp(self):
        self.UUT = fe.FiniteElements()

    def test_CalculateShapeFunctionsInputs(self):
        ''' Test the inputs to CalculateShapeFunctions are restricted. '''
        with self.assertRaises(AssertionError):
            self.UUT._CalculateShapeFunctions(1)
        with self.assertRaises(AssertionError):
            self.UUT._CalculateShapeFunctions(np.array([1]))

    def test_CalculateShapeFunctionsOutputs(self):
        ''' Test the outputs from CalculateShapeFunctions are correct. '''
        inputs = [1,2]
        outputs = self.UUT._CalculateShapeFunctions(inputs)
        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0], -2)
        self.assertEqual(outputs[1], 1)
        self.assertEqual(outputs[2], 2)

    def test_CalculateShapeFunctionDerivatives(self):
        ''' Test the outputs from CalculateShapeFunctionDerivatives are correct. '''
        outputs = self.UUT._CalculateShapeFunctionDerivatives()
        self.assertEqual(len(outputs), 2)
        self.assertEqual(len(outputs[0]), 3)
        self.assertEqual(outputs[0][0], -1)
        self.assertEqual(outputs[0][1], 1)
        self.assertEqual(outputs[0][2], 0)
        self.assertEqual(outputs[1][0], -1)
        self.assertEqual(outputs[1][1], 0)
        self.assertEqual(outputs[1][2], 1)

    def test_CalculateGlobalCoordinatesGivenLocalCoordinatesInputs(self):
        ''' Test the inputs to CalculateGlobalCoordinatesGivenLocalCoordinates are restricted. '''
        with self.assertRaises(AssertionError):
            localCoords = np.zeros([2])
            self.UUT._CalculateGlobalCoordinatesGivenLocalCoordinates(1, localCoords)
        with self.assertRaises(AssertionError):
            globalCoords = np.zeros([3,2])
            self.UUT._CalculateGlobalCoordinatesGivenLocalCoordinates(globalCoords, 1)
        with self.assertRaises(AssertionError):
            localCoords = np.zeros([3])
            globalCoords = np.zeros([3,2])
            self.UUT._CalculateGlobalCoordinatesGivenLocalCoordinates(globalCoords, localCoords)
        with self.assertRaises(AssertionError):
            localCoords = np.zeros([3])
            globalCoords = np.zeros([3,2])
            self.UUT._CalculateGlobalCoordinatesGivenLocalCoordinates(globalCoords, localCoords)
        with self.assertRaises(AssertionError):
            localCoords = np.zeros([2])
            globalCoords = np.zeros([3,1])
            self.UUT._CalculateGlobalCoordinatesGivenLocalCoordinates(globalCoords, localCoords)

    def test_CalculateGlobalCoordinatesGivenLocalCoordinates(self):
        ''' Test the outputs from CalculateGlobalCoordinatesGivenLocalCoordinates are correct. '''
        localCoords = np.zeros([2])
        globalCoords = np.zeros([3,2])
        globalCoords[0,0] = 0
        globalCoords[1,0] = 1
        globalCoords[2,0] = 0
        globalCoords[0,1] = 0
        globalCoords[1,1] = 0
        globalCoords[2,1] = 1
        localCoords[0] = 1/3
        localCoords[1] = 1/3

        globalOutput = self.UUT._CalculateGlobalCoordinatesGivenLocalCoordinates(globalCoords, localCoords)
        self.assertAlmostEqual(globalOutput[0], 1/3)
        self.assertAlmostEqual(globalOutput[1], 1/3)

    def test_CalculateJacobianGivenLocalCoordinatesInputs(self):
        ''' Test the inputs to CalculateJacobianGivenLocalCoordinatesInputs are restricted. '''
        with self.assertRaises(AssertionError):
            localCoords = np.zeros([2])
            self.UUT._CalculateJacobianGivenLocalCoordinates(1, localCoords)
        with self.assertRaises(AssertionError):
            globalCoords = np.zeros([3,2])
            self.UUT._CalculateJacobianGivenLocalCoordinates(globalCoords, 1)
        with self.assertRaises(AssertionError):
            localCoords = np.zeros([3])
            globalCoords = np.zeros([3,2])
            self.UUT._CalculateJacobianGivenLocalCoordinates(globalCoords, localCoords)
        with self.assertRaises(AssertionError):
            localCoords = np.zeros([3])
            globalCoords = np.zeros([3,2])
            self.UUT._CalculateJacobianGivenLocalCoordinates(globalCoords, localCoords)
        with self.assertRaises(AssertionError):
            localCoords = np.zeros([2])
            globalCoords = np.zeros([3,1])
            self.UUT._CalculateJacobianGivenLocalCoordinates(globalCoords, localCoords)

    def test_CalculateJacobianGivenLocalCoordinates(self):
        ''' Test the outputs from CalculateJacobianGivenLocalCoordinates are
         correct against a known set of inputs. '''
        localCoords = np.zeros([2])
        globalCoords = np.zeros([3,2])
        globalCoords[0,0] = 0
        globalCoords[1,0] = 1
        globalCoords[2,0] = 0
        globalCoords[0,1] = 0
        globalCoords[1,1] = 0
        globalCoords[2,1] = 1
        localCoords[0] = 1/3
        localCoords[1] = 1/3

        globalOutput = self.UUT._CalculateJacobianGivenLocalCoordinates(globalCoords, localCoords)
        self.assertAlmostEqual(globalOutput[0,0], 1)
        self.assertAlmostEqual(globalOutput[0,1], 0)
        self.assertAlmostEqual(globalOutput[1,0], 0)
        self.assertAlmostEqual(globalOutput[1,1], 1)

    def test_CalculateJacobianDeterminantGivenLocalCoordinatesInputs(self):
        ''' Test the inputs to _CalculateJacobianDeterminantGivenLocalCoordinates are restricted. '''
        with self.assertRaises(AssertionError):
            localCoords = np.zeros([2])
            self.UUT._CalculateJacobianDeterminantGivenLocalCoordinates(1, localCoords)
        with self.assertRaises(AssertionError):
            globalCoords = np.zeros([3,2])
            self.UUT._CalculateJacobianDeterminantGivenLocalCoordinates(globalCoords, 1)
        with self.assertRaises(AssertionError):
            localCoords = np.zeros([3])
            globalCoords = np.zeros([3,2])
            self.UUT._CalculateJacobianDeterminantGivenLocalCoordinates(globalCoords, localCoords)
        with self.assertRaises(AssertionError):
            localCoords = np.zeros([3])
            globalCoords = np.zeros([3,2])
            self.UUT._CalculateJacobianDeterminantGivenLocalCoordinates(globalCoords, localCoords)
        with self.assertRaises(AssertionError):
            localCoords = np.zeros([2])
            globalCoords = np.zeros([3,1])
            self.UUT._CalculateJacobianDeterminantGivenLocalCoordinates(globalCoords, localCoords)

    def test_CalculateJacobianDeterminantGivenLocalCoordinates(self):
        ''' Test the outputs from CalculateJacobianGivenLocalCoordinates are
         correct against a known set of inputs. '''
        localCoords = np.zeros([2])
        globalCoords = np.zeros([3,2])
        globalCoords[0,0] = 0
        globalCoords[1,0] = 1
        globalCoords[2,0] = 0
        globalCoords[0,1] = 0
        globalCoords[1,1] = 0
        globalCoords[2,1] = 1
        localCoords[0] = 1/3
        localCoords[1] = 1/3

        determinant = self.UUT._CalculateJacobianDeterminantGivenLocalCoordinates(globalCoords, localCoords)
        self.assertAlmostEqual(determinant, 1)

    def test_CalculateDerivativesOfShapeFunctionsInGlobalCoordinatesInputs(self):
        ''' Test the inputs to _CalculateDerivativesOfShapeFunctionsInGlobalCoordinates are restricted. '''
        with self.assertRaises(AssertionError):
            localCoords = np.zeros([2])
            self.UUT._CalculateDerivativesOfShapeFunctionsInGlobalCoordinates(1, localCoords)
        with self.assertRaises(AssertionError):
            globalCoords = np.zeros([3,2])
            self.UUT._CalculateDerivativesOfShapeFunctionsInGlobalCoordinates(globalCoords, 1)
        with self.assertRaises(AssertionError):
            localCoords = np.zeros([3])
            globalCoords = np.zeros([3,2])
            self.UUT._CalculateDerivativesOfShapeFunctionsInGlobalCoordinates(globalCoords, localCoords)
        with self.assertRaises(AssertionError):
            localCoords = np.zeros([3])
            globalCoords = np.zeros([3,2])
            self.UUT._CalculateDerivativesOfShapeFunctionsInGlobalCoordinates(globalCoords, localCoords)
        with self.assertRaises(AssertionError):
            localCoords = np.zeros([2])
            globalCoords = np.zeros([3,1])
            self.UUT._CalculateDerivativesOfShapeFunctionsInGlobalCoordinates(globalCoords, localCoords)

    def test_CalculateDerivativesOfShapeFunctionsInGlobalCoordinates(self):
        ''' Test the calculation of the derivative of the shape function at
        the location specified. '''
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        IEN = np.array([[0, 1, 2],
                           [1, 3, 2]])

        output = self.UUT._CalculateDerivativesOfShapeFunctionsInGlobalCoordinates(nodes[IEN[0,:],:], np.array([1/6,1/6]))
        self.assertEqual(output[0,0], -1)
        self.assertEqual(output[0,1], -1)
        self.assertEqual(output[1,0], 1)
        self.assertEqual(output[1,1], -0)
        self.assertEqual(output[2,0], 0)
        self.assertEqual(output[2,1], 1)

    def test_PerformQuadratureOnReferenceTriangleInputs(self):
        ''' Test that we can only pass a function into
            _PerformQuadratureOnReferenceTriangleInputs '''
        with self.assertRaises(AssertionError):
            self.UUT._PerformQuadratureOnReferenceTriangle(4)

    def test_PerformQuadratureOnElementInputs(self):
        ''' Perform testing of the input error checking for
        _PerformQuadratureOnElement'''
        globalCoords = np.zeros([3,2])
        globalCoords[0,0] = 0
        globalCoords[1,0] = 1
        globalCoords[2,0] = 0
        globalCoords[0,1] = 0
        globalCoords[1,1] = 0
        globalCoords[2,1] = 1
        with self.assertRaises(AssertionError):
            self.UUT._PerformQuadratureOnElement(4, globalCoords)
        with self.assertRaises(AssertionError):
            globalCoords.shape = (2,3)
            self.UUT._PerformQuadratureOnElement(lambda x,y,z: 1, globalCoords)

    def test_CalculateStiffnessInputs(self):
        ''' Perform testing of the input error checking for _CalculateStiffness '''
        globalCoords = np.zeros([3,2])
        globalCoords[0,0] = 0
        globalCoords[1,0] = 1
        globalCoords[2,0] = 0
        globalCoords[0,1] = 0
        globalCoords[1,1] = 0
        globalCoords[2,1] = 1
        with self.assertRaises(AssertionError):
            globalCoords.shape = (2,3)
            self.UUT._CalculateStiffness(globalCoords)

    def test_CalculateStiffness(self):
        ''' Test that _CalculateStiffness returns the expected stiffness matrix '''
        globalCoords = np.zeros([3,2])
        globalCoords[0,0] = 0
        globalCoords[1,0] = 1
        globalCoords[2,0] = 0
        globalCoords[0,1] = 0
        globalCoords[1,1] = 0
        globalCoords[2,1] = 1
        stiffness = self.UUT._CalculateStiffness(globalCoords)
        self.assertAlmostEqual(stiffness[0,0], 1)
        self.assertAlmostEqual(stiffness[0,1], -0.5)
        self.assertAlmostEqual(stiffness[0,2], -0.5)
        self.assertAlmostEqual(stiffness[1,0], -0.5)
        self.assertAlmostEqual(stiffness[1,1], 0.5)
        self.assertAlmostEqual(stiffness[1,2], 0.0)
        self.assertAlmostEqual(stiffness[2,0], -0.5)
        self.assertAlmostEqual(stiffness[2,1], 0.0)
        self.assertAlmostEqual(stiffness[2,2], 0.5)

    def test_CalculateForceInputs(self):
        ''' Perform testing of the input error checking for _CalculateForce '''
        globalCoords = np.zeros([3,2])
        globalCoords[0,0] = 0
        globalCoords[1,0] = 1
        globalCoords[2,0] = 0
        globalCoords[0,1] = 0
        globalCoords[1,1] = 0
        globalCoords[2,1] = 1
        with self.assertRaises(AssertionError):
            self.UUT._CalculateForce(globalCoords, 1)
        with self.assertRaises(AssertionError):
            globalCoords.shape = (2,3)
            self.UUT._CalculateForce(globalCoords, lambda x:1)


if __name__ == "__main__":
    import xmlrunner
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))


#include "bempp_base_types.h"
#include "bempp_helpers.h"
#include "bempp_spaces.h"
#include "kernels.h"

__kernel __attribute__((vec_type_hint(REALTYPEVEC))) void
kernel_function(
    __global uint *testIndices, __global uint *trialIndices,
    __global int *testNormalSigns, __global int *trialNormalSigns,
    __global REALTYPE *testGrid, __global REALTYPE *trialGrid,
    __global uint *testConnectivity, __global uint *trialConnectivity,
    __global uint *testLocal2Global, __global uint *trialLocal2Global,
    __global REALTYPE *testLocalMultipliers,
    __global REALTYPE *trialLocalMultipliers, __constant REALTYPE *quadPoints,
    __constant REALTYPE *quadWeights, __global REALTYPE *globalResult,
    __global REALTYPE *kernel_parameters, int nTest, int nTrial,
    char gridsAreDisjoint) {

  size_t gid[2] = {get_global_id(0), get_global_id(1)};
  size_t offset = get_global_offset(1);

  size_t testIndex = testIndices[gid[0]];
  DEFINE_TRIAL_INDICES_REGULAR_ASSEMBLY

  size_t testQuadIndex;
  size_t trialQuadIndex;
  size_t i;
  size_t j;
  size_t globalRowIndex;
  size_t globalColIndex;

  REALTYPE3 testGlobalPoint;
  REALTYPEVEC trialGlobalPoint[3];

  REALTYPE3 testCorners[3];
  REALTYPEVEC trialCorners[3][3];

  uint testElement[3];
  uint trialElement[VEC_LENGTH][3];

  uint myTestLocal2Global[3];
  uint myTrialLocal2Global[VEC_LENGTH][3];

  REALTYPE myTestLocalMultipliers[3];
  REALTYPE myTrialLocalMultipliers[VEC_LENGTH][3];

  REALTYPE3 testJac[2];
  REALTYPEVEC trialJac[2][3];

  REALTYPE3 testNormal;
  REALTYPEVEC trialNormal[3];

  REALTYPE2 testPoint;
  REALTYPE2 trialPoint;

  REALTYPE testIntElem;
  REALTYPEVEC trialIntElem;
  REALTYPE testValue[3][2];
  REALTYPE trialValue[3][2];
  REALTYPE3 testElementValue[3];
  REALTYPEVEC trialElementValue[3][3];
  REALTYPE testEdgeLength[3];
  REALTYPEVEC trialEdgeLength[3];

  REALTYPEVEC kernelValue;
  REALTYPEVEC shapeIntegral[3][3];
  REALTYPEVEC tempResult[3][3];

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j)
      shapeIntegral[i][j] = M_ZERO;

  getCorners(testGrid, testIndex, testCorners);
  getCornersVec(trialGrid, trialIndex, trialCorners);

  getElement(testConnectivity, testIndex, testElement);
  getElementVec(trialConnectivity, trialIndex, trialElement);

  getLocal2Global(testLocal2Global, testIndex, myTestLocal2Global, 3);
  getLocal2GlobalVec(trialLocal2Global, trialIndex, &myTrialLocal2Global[0][0],
                      3);

  getLocalMultipliers(testLocalMultipliers, testIndex, myTestLocalMultipliers,
                      3);
  getLocalMultipliersVec(trialLocalMultipliers, trialIndex,
                          &myTrialLocalMultipliers[0][0], 3);

  getJacobian(testCorners, testJac);
  getJacobianVec(trialCorners, trialJac);

  getNormalAndIntegrationElement(testJac, &testNormal, &testIntElem);
  getNormalAndIntegrationElementVec(trialJac, trialNormal, &trialIntElem);

  computeEdgeLength(testCorners, testEdgeLength);
  computeEdgeLengthVec(trialCorners, trialEdgeLength);

  updateNormals(testIndex, testNormalSigns, &testNormal);
  updateNormalsVec(trialIndex, trialNormalSigns, trialNormal);

  for (testQuadIndex = 0; testQuadIndex < NUMBER_OF_QUAD_POINTS;
       ++testQuadIndex) {
    testPoint = (REALTYPE2)(quadPoints[2 * testQuadIndex],
                            quadPoints[2 * testQuadIndex + 1]);
    testGlobalPoint = getGlobalPoint(testCorners, &testPoint);
    BASIS(TEST, evaluate)
    (&testPoint, &testValue[0][0]);
    getPiolaTransform(testIntElem, testJac, testValue, testElementValue);

    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j)
        tempResult[i][j] = M_ZERO;

    for (trialQuadIndex = 0; trialQuadIndex < NUMBER_OF_QUAD_POINTS;
         ++trialQuadIndex) {
      trialPoint = (REALTYPE2)(quadPoints[2 * trialQuadIndex],
                               quadPoints[2 * trialQuadIndex + 1]);
      getGlobalPointVec(trialCorners, &trialPoint, trialGlobalPoint);
      BASIS(TRIAL, evaluate)
      (&trialPoint, &trialValue[0][0]);
      getPiolaTransformVec(trialIntElem, trialJac, trialValue,
                            trialElementValue);

      /* Compute Laplace kernel G(x,y) */
      KERNEL(VEC_STRING)
      (testGlobalPoint, trialGlobalPoint, testNormal, trialNormal,
       kernel_parameters, &kernelValue);

      /* diff = x - y, dist = |x - y| (vectorized over trial) */
      REALTYPEVEC diffx = testGlobalPoint.x - trialGlobalPoint[0];
      REALTYPEVEC diffy = testGlobalPoint.y - trialGlobalPoint[1];
      REALTYPEVEC diffz = testGlobalPoint.z - trialGlobalPoint[2];
      REALTYPEVEC dist2 = diffx * diffx + diffy * diffy + diffz * diffz;

      /* factor = G * (-1) / |x-y|^2 * quad_weight */
      REALTYPEVEC factor =
          kernelValue * (-M_ONE) / dist2 * quadWeights[trialQuadIndex];

      /* Accumulate cross product contributions:
         For each (i,j): factor * diff . (test_i x trial_j) */
      for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j) {
          /* cross(testElementValue[i], trialElementValue[j]) */
          REALTYPEVEC cx = testElementValue[i].y * trialElementValue[j][2] -
                           testElementValue[i].z * trialElementValue[j][1];
          REALTYPEVEC cy = testElementValue[i].z * trialElementValue[j][0] -
                           testElementValue[i].x * trialElementValue[j][2];
          REALTYPEVEC cz = testElementValue[i].x * trialElementValue[j][1] -
                           testElementValue[i].y * trialElementValue[j][0];
          /* dot(diff, cross) */
          tempResult[i][j] += factor * (diffx * cx + diffy * cy + diffz * cz);
        }
    }

    for (i = 0; i < 3; ++i)
      for (j = 0; j < 3; ++j)
        shapeIntegral[i][j] += quadWeights[testQuadIndex] * tempResult[i][j];
  }

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 3; ++j) {
      shapeIntegral[i][j] *= testEdgeLength[i] * trialEdgeLength[j];
      shapeIntegral[i][j] *=
          (testIntElem * myTestLocalMultipliers[i]) * trialIntElem;
    }

  for (int vecIndex = 0; vecIndex < VEC_LENGTH; ++vecIndex)
    if (!elementsAreAdjacent(testElement, trialElement[vecIndex],
                             gridsAreDisjoint)) {
      for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j) {
          globalRowIndex = myTestLocal2Global[i];
          globalColIndex = myTrialLocal2Global[vecIndex][j];
          globalResult[globalRowIndex * nTrial + globalColIndex] +=
              ((REALTYPE *)(&shapeIntegral[i][j]))[vecIndex] *
              myTrialLocalMultipliers[vecIndex][j];
        }
    }
}

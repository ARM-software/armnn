//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DetectionPostProcess.hpp"

#include <armnn/ArmNN.hpp>

#include <boost/numeric/conversion/cast.hpp>

#include <algorithm>
#include <numeric>

namespace
{

std::vector<unsigned int> GenerateRangeK(unsigned int k)
{
    std::vector<unsigned int> range(k);
    std::iota(range.begin(), range.end(), 0);
    return range;
}

void TopKSort(unsigned int k, unsigned int* indices, const float* values, unsigned int numElement)
{
    std::partial_sort(indices, indices + k, indices + numElement,
                      [&values](unsigned int i, unsigned int j) { return values[i] > values[j]; });
}

float IntersectionOverUnion(const float* boxI, const float* boxJ)
{
    // Box-corner format: ymin, xmin, ymax, xmax.
    const int yMin = 0;
    const int xMin = 1;
    const int yMax = 2;
    const int xMax = 3;
    float areaI = (boxI[yMax] - boxI[yMin]) * (boxI[xMax] - boxI[xMin]);
    float areaJ = (boxJ[yMax] - boxJ[yMin]) * (boxJ[xMax] - boxJ[xMin]);
    float yMinIntersection = std::max(boxI[yMin], boxJ[yMin]);
    float xMinIntersection = std::max(boxI[xMin], boxJ[xMin]);
    float yMaxIntersection = std::min(boxI[yMax], boxJ[yMax]);
    float xMaxIntersection = std::min(boxI[xMax], boxJ[xMax]);
    float areaIntersection = std::max(yMaxIntersection - yMinIntersection, 0.0f) *
                                std::max(xMaxIntersection - xMinIntersection, 0.0f);
    float areaUnion = areaI + areaJ - areaIntersection;
    return areaIntersection / areaUnion;
}

std::vector<unsigned int> NonMaxSuppression(unsigned int numBoxes, const std::vector<float>& boxCorners,
                                            const std::vector<float>& scores, float nmsScoreThreshold,
                                            unsigned int maxDetection, float nmsIouThreshold)
{
    // Select boxes that have scores above a given threshold.
    std::vector<float> scoresAboveThreshold;
    std::vector<unsigned int> indicesAboveThreshold;
    for (unsigned int i = 0; i < numBoxes; ++i)
    {
        if (scores[i] >= nmsScoreThreshold)
        {
            scoresAboveThreshold.push_back(scores[i]);
            indicesAboveThreshold.push_back(i);
        }
    }

    // Sort the indices based on scores.
    unsigned int numAboveThreshold = boost::numeric_cast<unsigned int>(scoresAboveThreshold.size());
    std::vector<unsigned int> sortedIndices = GenerateRangeK(numAboveThreshold);
    TopKSort(numAboveThreshold,sortedIndices.data(), scoresAboveThreshold.data(), numAboveThreshold);

    // Number of output cannot be more than max detections specified in the option.
    unsigned int numOutput = std::min(maxDetection, numAboveThreshold);
    std::vector<unsigned int> outputIndices;
    std::vector<bool> visited(numAboveThreshold, false);

    // Prune out the boxes with high intersection over union by keeping the box with higher score.
    for (unsigned int i = 0; i < numAboveThreshold; ++i)
    {
        if (outputIndices.size() >= numOutput)
        {
            break;
        }
        if (!visited[sortedIndices[i]])
        {
            outputIndices.push_back(indicesAboveThreshold[sortedIndices[i]]);
        }
        for (unsigned int j = i + 1; j < numAboveThreshold; ++j)
        {
            unsigned int iIndex = indicesAboveThreshold[sortedIndices[i]] * 4;
            unsigned int jIndex = indicesAboveThreshold[sortedIndices[j]] * 4;
            if (IntersectionOverUnion(&boxCorners[iIndex], &boxCorners[jIndex]) > nmsIouThreshold)
            {
                visited[sortedIndices[j]] = true;
            }
        }
    }
    return outputIndices;
}

void AllocateOutputData(unsigned int numOutput, unsigned int numSelected, const std::vector<float>& boxCorners,
                        const std::vector<unsigned int>& outputIndices, const std::vector<unsigned int>& selectedBoxes,
                        const std::vector<unsigned int>& selectedClasses, const std::vector<float>& selectedScores,
                        float* detectionBoxes, float* detectionScores, float* detectionClasses, float* numDetections)
{
    for (unsigned int i = 0; i < numOutput; ++i)
        {
            unsigned int boxIndex = i * 4;
            if (i < numSelected)
            {
                unsigned int boxCornorIndex = selectedBoxes[outputIndices[i]] * 4;
                detectionScores[i] = selectedScores[outputIndices[i]];
                detectionClasses[i] = boost::numeric_cast<float>(selectedClasses[outputIndices[i]]);
                detectionBoxes[boxIndex] = boxCorners[boxCornorIndex];
                detectionBoxes[boxIndex + 1] = boxCorners[boxCornorIndex + 1];
                detectionBoxes[boxIndex + 2] = boxCorners[boxCornorIndex + 2];
                detectionBoxes[boxIndex + 3] = boxCorners[boxCornorIndex + 3];
            }
            else
            {
                detectionScores[i] = 0.0f;
                detectionClasses[i] = 0.0f;
                detectionBoxes[boxIndex] = 0.0f;
                detectionBoxes[boxIndex + 1] = 0.0f;
                detectionBoxes[boxIndex + 2] = 0.0f;
                detectionBoxes[boxIndex + 3] = 0.0f;
            }
        }
        numDetections[0] = boost::numeric_cast<float>(numSelected);
}

} // anonymous namespace

namespace armnn
{

void DetectionPostProcess(const TensorInfo& boxEncodingsInfo,
                          const TensorInfo& scoresInfo,
                          const TensorInfo& anchorsInfo,
                          const TensorInfo& detectionBoxesInfo,
                          const TensorInfo& detectionClassesInfo,
                          const TensorInfo& detectionScoresInfo,
                          const TensorInfo& numDetectionsInfo,
                          const DetectionPostProcessDescriptor& desc,
                          const float* boxEncodings,
                          const float* scores,
                          const float* anchors,
                          float* detectionBoxes,
                          float* detectionClasses,
                          float* detectionScores,
                          float* numDetections)
{
    // Transform center-size format which is (ycenter, xcenter, height, width) to box-corner format,
    // which represents the lower left corner and the upper right corner (ymin, xmin, ymax, xmax)
    std::vector<float> boxCorners(boxEncodingsInfo.GetNumElements());
    unsigned int numBoxes = boxEncodingsInfo.GetShape()[1];
    for (unsigned int i = 0; i < numBoxes; ++i)
    {
        unsigned int indexY = i * 4;
        unsigned int indexX = indexY + 1;
        unsigned int indexH = indexX + 1;
        unsigned int indexW = indexH + 1;
        float yCentre = boxEncodings[indexY] / desc.m_ScaleY * anchors[indexH] + anchors[indexY];
        float xCentre = boxEncodings[indexX] / desc.m_ScaleX * anchors[indexW] + anchors[indexX];
        float halfH = 0.5f * expf(boxEncodings[indexH] / desc.m_ScaleH) * anchors[indexH];
        float halfW = 0.5f * expf(boxEncodings[indexW] / desc.m_ScaleW) * anchors[indexW];
        // ymin
        boxCorners[indexY] = yCentre - halfH;
        // xmin
        boxCorners[indexX] = xCentre - halfW;
        // ymax
        boxCorners[indexH] = yCentre + halfH;
        // xmax
        boxCorners[indexW] = xCentre + halfW;

        BOOST_ASSERT(boxCorners[indexY] < boxCorners[indexH]);
        BOOST_ASSERT(boxCorners[indexX] < boxCorners[indexW]);
    }

    unsigned int numClassesWithBg = desc.m_NumClasses + 1;

    // Perform Non Max Suppression.
    if (desc.m_UseRegularNms)
    {
        // Perform Regular NMS.
        // For each class, perform NMS and select max detection numbers of the highest score across all classes.
        std::vector<float> classScores(numBoxes);
        std::vector<unsigned int>selectedBoxesAfterNms;
        std::vector<float> selectedScoresAfterNms;
        std::vector<unsigned int> selectedClasses;

        for (unsigned int c = 0; c < desc.m_NumClasses; ++c)
        {
            // For each boxes, get scores of the boxes for the class c.
            for (unsigned int i = 0; i < numBoxes; ++i)
            {
                classScores[i] = scores[i * numClassesWithBg + c + 1];
            }
            std::vector<unsigned int> selectedIndices = NonMaxSuppression(numBoxes, boxCorners, classScores,
                                                                          desc.m_NmsScoreThreshold,
                                                                          desc.m_DetectionsPerClass,
                                                                          desc.m_NmsIouThreshold);

            for (unsigned int i = 0; i < selectedIndices.size(); ++i)
            {
                selectedBoxesAfterNms.push_back(selectedIndices[i]);
                selectedScoresAfterNms.push_back(classScores[selectedIndices[i]]);
                selectedClasses.push_back(c);
            }
        }

        // Select max detection numbers of the highest score across all classes
        unsigned int numSelected = boost::numeric_cast<unsigned int>(selectedBoxesAfterNms.size());
        unsigned int numOutput = std::min(desc.m_MaxDetections,  numSelected);

        // Sort the max scores among the selected indices.
        std::vector<unsigned int> outputIndices = GenerateRangeK(numSelected);
        TopKSort(numOutput, outputIndices.data(), selectedScoresAfterNms.data(), numSelected);

        AllocateOutputData(detectionBoxesInfo.GetShape()[1], numOutput, boxCorners, outputIndices,
                           selectedBoxesAfterNms, selectedClasses, selectedScoresAfterNms,
                           detectionBoxes, detectionScores, detectionClasses, numDetections);
    }
    else
    {
        // Perform Fast NMS.
        // Select max scores of boxes and perform NMS on max scores,
        // select max detection numbers of the highest score
        unsigned int numClassesPerBox = std::min(desc.m_MaxClassesPerDetection, desc.m_NumClasses);
        std::vector<float> maxScores;
        std::vector<unsigned int>boxIndices;
        std::vector<unsigned int>maxScoreClasses;

        for (unsigned int box = 0; box < numBoxes; ++box)
        {
            unsigned int scoreIndex = box * numClassesWithBg + 1;

            // Get the max scores of the box.
            std::vector<unsigned int> maxScoreIndices = GenerateRangeK(desc.m_NumClasses);
            TopKSort(numClassesPerBox, maxScoreIndices.data(), scores + scoreIndex, desc.m_NumClasses);

            for (unsigned int i = 0; i < numClassesPerBox; ++i)
            {
                maxScores.push_back(scores[scoreIndex + maxScoreIndices[i]]);
                maxScoreClasses.push_back(maxScoreIndices[i]);
                boxIndices.push_back(box);
            }
        }

        // Perform NMS on max scores
        std::vector<unsigned int> selectedIndices = NonMaxSuppression(numBoxes, boxCorners, maxScores,
                                                                      desc.m_NmsScoreThreshold,
                                                                      desc.m_MaxDetections,
                                                                      desc.m_NmsIouThreshold);

        unsigned int numSelected = boost::numeric_cast<unsigned int>(selectedIndices.size());
        unsigned int numOutput = std::min(desc.m_MaxDetections,  numSelected);

        AllocateOutputData(detectionBoxesInfo.GetShape()[1], numOutput, boxCorners, selectedIndices,
                           boxIndices, maxScoreClasses, maxScores,
                           detectionBoxes, detectionScores, detectionClasses, numDetections);
    }
}

} // namespace armnn

//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <catch.hpp>
#include <limits>

#include "DataStructures.hpp"
#include "Wav2LetterPreprocessor.hpp"

void PopulateTestWavVector(std::vector<int16_t>& vec)
{
    constexpr int int16max = std::numeric_limits<int16_t>::max();
    int val = 0;
    for (size_t i = 0; i < vec.size(); ++i, ++val)
    {

        /* We want a differential filter response from both - order 1
         * and 2 => Don't have a linear signal here - we use a signal
         * using squares for example. Alternate sign flips might work
         * just as well and will be computationally less work! */
        int valsq = val * val;
        if (valsq > int16max)
        {
            val = 0;
            valsq = 0;
        }
        vec[i] = valsq;
    }
}

TEST_CASE("Preprocessing calculation INT8")
{
    /*Test  Constants: */
    const uint32_t  windowLen             = 512;
    const uint32_t  windowStride          = 160;
    const float     quantScale            = 0.1410219967365265;
    const int       quantOffset           = -11;
    int             numMfccVectors        = 10;
    const int       sampFreq              = 16000;
    const int       frameLenMs            = 32;
    const int       frameLenSamples       = sampFreq * frameLenMs * 0.001;
    const int       numMfccFeats          = 13;
    const int       audioDataToPreProcess = 512 + ((numMfccVectors -1) * windowStride);
    int             outputBufferSize = numMfccVectors * numMfccFeats * 3;

    /* Test wav memory */
    std::vector <int16_t> testWav1((windowStride * numMfccVectors) +
                              (windowLen - windowStride));
    /* Populate with dummy input */
    PopulateTestWavVector(testWav1);

    MfccParams mfccParams(sampFreq, 128, 0, 8000, numMfccFeats,
                          frameLenSamples, false, numMfccVectors);

    std::unique_ptr<Wav2LetterMFCC> mfccInst = std::make_unique<Wav2LetterMFCC>(mfccParams);

    std::vector<float> fullAudioData;

    for(int i = 0; i < 4; ++i)
    {
        for (auto f : testWav1)
        {
            fullAudioData.emplace_back(static_cast<float>(f) / (1<<15));
        }
    }

    Wav2LetterPreprocessor prep(frameLenSamples, windowStride, std::move(mfccInst));

    std::vector<int8_t> outputBuffer(outputBufferSize);

    prep.Invoke(fullAudioData.data(), audioDataToPreProcess, outputBuffer, quantOffset, quantScale);

    int8_t expectedResult[numMfccVectors][numMfccFeats*3] =
    {
            /* Feature vec 0 */
            -32, 4, -9, -8, -10, -10, -11, -11, -11, -11, -12, -11, -11,    /* MFCCs   */
            -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11,    /* Delta 1 */
            -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10,    /* Delta 2 */

            /* Feature vec 1 */
            -31, 4, -9, -8, -10, -10, -11, -11, -11, -11, -12, -11, -11,
            -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11,
            -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10,

            /* Feature vec 2 */
            -31, 4, -9, -9, -10, -10, -11, -11, -11, -11, -12, -12, -12,
            -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11,
            -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10,

            /* Feature vec 3 */
            -31, 4, -9, -9, -10, -10, -11, -11, -11, -11, -11, -12, -12,
            -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11,
            -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10,

            /* Feature vec 4 : this should have valid delta 1 and delta 2 */
            -31, 4, -9, -9, -10, -10, -11, -11, -11, -11, -11, -12, -12,
            -38, -29, -9, 1, -2, -7, -8, -8, -12, -16, -14, -5, 5,
            -68, -50, -13, 5, 0, -9, -9, -8, -13, -20, -19, -3, 15,

            /* Feature vec 5 : this should have valid delta 1 and delta 2 */
            -31, 4, -9, -8, -10, -10, -11, -11, -11, -11, -11, -12, -12,
            -62, -45, -11, 5, 0, -8, -9, -8, -12, -19, -17, -3, 13,
            -27, -22, -13, -9, -11, -12, -12, -11, -11, -13, -13, -10, -6,

            /* Feature vec 6 */
            -31, 4, -9, -8, -10, -10, -11, -11, -11, -11, -12, -11, -11,
            -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11,
            -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10,

            /* Feature vec 7 */
            -32, 4, -9, -8, -10, -10, -11, -11, -11, -12, -12, -11, -11,
            -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11,
            -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10,

            /* Feature vec 8 */
            -32, 4, -9, -8, -10, -10, -11, -11, -11, -12, -12, -11, -11,
            -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11,
            -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10,

            /* Feature vec 9 */
            -31, 4, -9, -8, -10, -10, -11, -11, -11, -11, -12, -11, -11,
            -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11,
            -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10
    };

    /* Check that the elements have been calculated correctly */
    for (uint32_t j = 0; j < numMfccVectors; ++j)
    {
        for (uint32_t i = 0; i < numMfccFeats * 3; ++i)
        {
            size_t tensorIdx = (j * numMfccFeats * 3) + i;
            CHECK(static_cast<int16_t>(outputBuffer.at(tensorIdx) == static_cast<int16_t>(expectedResult[j][i])));
        }
    }
}

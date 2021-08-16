//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "Wav2LetterMFCC.hpp"
#include "MathUtils.hpp"

#include <cfloat>

bool Wav2LetterMFCC::ApplyMelFilterBank(
        std::vector<float>&                 fftVec,
        std::vector<std::vector<float>>&    melFilterBank,
        std::vector<uint32_t>&               filterBankFilterFirst,
        std::vector<uint32_t>&               filterBankFilterLast,
        std::vector<float>&                 melEnergies)
{
    const size_t numBanks = melEnergies.size();

    if (numBanks != filterBankFilterFirst.size() ||
            numBanks != filterBankFilterLast.size()) 
    {
        printf("Unexpected filter bank lengths\n");
        return false;
    }

    for (size_t bin = 0; bin < numBanks; ++bin) 
    {
        auto filterBankIter = melFilterBank[bin].begin();
        auto end = melFilterBank[bin].end();
        // Avoid log of zero at later stages, same value used in librosa.
        // The number was used during our default wav2letter model training. 
        float melEnergy = 1e-10;
        const uint32_t firstIndex = filterBankFilterFirst[bin];
        const uint32_t lastIndex = std::min<uint32_t>(filterBankFilterLast[bin], fftVec.size() - 1);

        for (uint32_t i = firstIndex; i <= lastIndex && filterBankIter != end; ++i) 
        {
            melEnergy += (*filterBankIter++ * fftVec[i]);
        }

        melEnergies[bin] = melEnergy;
    }

    return true;
}

void Wav2LetterMFCC::ConvertToLogarithmicScale(std::vector<float>& melEnergies)
{
    float maxMelEnergy = -FLT_MAX;

    // Container for natural logarithms of mel energies. 
    std::vector <float> vecLogEnergies(melEnergies.size(), 0.f);

    // Because we are taking natural logs, we need to multiply by log10(e).
    // Also, for wav2letter model, we scale our log10 values by 10. 
    constexpr float multiplier = 10.0 *  // Default scalar. 
                                  0.4342944819032518;  // log10f(std::exp(1.0)) 

    // Take log of the whole vector. 
    MathUtils::VecLogarithmF32(melEnergies, vecLogEnergies);

    // Scale the log values and get the max. 
    for (auto iterM = melEnergies.begin(), iterL = vecLogEnergies.begin();
              iterM != melEnergies.end() && iterL != vecLogEnergies.end(); ++iterM, ++iterL) 
    {

        *iterM = *iterL * multiplier;

        // Save the max mel energy. 
        if (*iterM > maxMelEnergy) 
        {
            maxMelEnergy = *iterM;
        }
    }

    // Clamp the mel energies. 
    constexpr float maxDb = 80.0;
    const float clampLevelLowdB = maxMelEnergy - maxDb;
    for (float& melEnergy : melEnergies) 
    {
        melEnergy = std::max(melEnergy, clampLevelLowdB);
    }
}

std::vector<float> Wav2LetterMFCC::CreateDCTMatrix(
                                    const int32_t inputLength,
                                    const int32_t coefficientCount)
{
    std::vector<float> dctMatix(inputLength * coefficientCount);

    // Orthonormal normalization. 
    const float normalizerK0 = 2 * sqrtf(1.0f /
                                    static_cast<float>(4 * inputLength));
    const float normalizer = 2 * sqrtf(1.0f /
                                    static_cast<float>(2 * inputLength));

    const float angleIncr = M_PI / inputLength;
    float angle = angleIncr;  // We start using it at k = 1 loop. 

    // First row of DCT will use normalizer K0. 
    for (int32_t n = 0; n < inputLength; ++n) 
    {
        dctMatix[n] = normalizerK0;  // cos(0) = 1 
    }

    // Second row (index = 1) onwards, we use standard normalizer. 
    for (int32_t k = 1, m = inputLength; k < coefficientCount; ++k, m += inputLength) 
    {
        for (int32_t n = 0; n < inputLength; ++n) 
        {
            dctMatix[m+n] = normalizer * cosf((n + 0.5f) * angle);
        }
        angle += angleIncr;
    }
    return dctMatix;
}

float Wav2LetterMFCC::GetMelFilterBankNormaliser(
                                const float&    leftMel,
                                const float&    rightMel,
                                const bool      useHTKMethod)
{
    // Slaney normalization for mel weights. 
    return (2.0f / (MFCC::InverseMelScale(rightMel, useHTKMethod) -
            MFCC::InverseMelScale(leftMel, useHTKMethod)));
}

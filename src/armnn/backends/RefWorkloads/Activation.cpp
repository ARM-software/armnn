//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Activation.hpp"

#include <boost/log/trivial.hpp>

#include <cmath>

namespace armnn
{

void Activation(const float* in,
               float* out,
               const TensorInfo& tensorInfo,
               ActivationFunction function,
               float a,
               float b)
{
    for (size_t i = 0; i<tensorInfo.GetNumElements(); i++)
    {
        float input = in[i];
        float output;

        // Compute the result of the activation function.
        switch (function)
        {
            case ActivationFunction::Linear:
            {
                output = a * input + b;
                break;
            }
            case ActivationFunction::Sigmoid:
            {
                output = 1.f / (1.f + expf(-input));
                break;
            }
            case ActivationFunction::ReLu:
            {
                output = std::max(0.f, input);
                break;
            }
            case ActivationFunction::BoundedReLu:
            {
                output = std::min(a, std::max(b, input));
                break;
            }
            case ActivationFunction::SoftReLu:
            {
                output = logf(1.0f + expf(input));
                break;
            }
            case ActivationFunction::LeakyReLu:
            {
                output = input > 0.0f ? input : (input * a);
                break;
            }
            case ActivationFunction::Abs:
            {
                output = input < 0 ? -input : input;
                break;
            }
            case ActivationFunction::Sqrt:
            {
                output = sqrtf(input);
                break;
            }
            case ActivationFunction::Square:
            {
                output = input * input;
                break;
            }
            case ActivationFunction::TanH:
            {
                output = a * tanhf(b * input);
                break;
            }
            default:
            {
                BOOST_LOG_TRIVIAL(error) << "Unsupported activation function";
                return;
            }
        }

        out[i] = output;
    }
}

} //namespace armnn

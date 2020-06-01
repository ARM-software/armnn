//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Activation.hpp"

#include <cmath>

namespace armnn
{

float Activation(float in,
                 ActivationFunction function,
                 float a,
                 float b)
{
    float output;

    // Compute the result of the activation function.
    switch (function)
    {
        case ActivationFunction::Linear:
        {
            output = a * in + b;
            break;
        }
        case ActivationFunction::Sigmoid:
        {
            output = 1.f / (1.f + expf(-in));
            break;
        }
        case ActivationFunction::ReLu:
        {
            output = std::max(0.f, in);
            break;
        }
        case ActivationFunction::BoundedReLu:
        {
            output = std::min(a, std::max(b, in));
            break;
        }
        case ActivationFunction::SoftReLu:
        {
            output = logf(1.0f + expf(in));
            break;
        }
        case ActivationFunction::LeakyReLu:
        {
            output = in > 0.0f ? in : (in * a);
            break;
        }
        case ActivationFunction::Abs:
        {
            output = in < 0 ? -in : in;
            break;
        }
        case ActivationFunction::Sqrt:
        {
            output = sqrtf(in);
            break;
        }
        case ActivationFunction::Square:
        {
            output = in * in;
            break;
        }
        case ActivationFunction::TanH:
        {
            output = a * tanhf(b * in);
            break;
        }
        case ActivationFunction::Elu:
        {
            output = (in >= 0) ? in : a * (expf(in) - 1);
            break;
        }
        case ActivationFunction::HardSwish:
        {
            // hard_swish(x) = x * relu6(x+3) / 6
            // relu6(x) = min(max(x,0),6)
            output = in * (std::min(std::max((in + 3),0.0f),6.0f)) / 6;
            break;
        }
        default:
        {
            throw InvalidArgumentException("Unsupported activation function");
        }
    }

    return output;
}


void Activation(Decoder<float>& in,
                Encoder<float>& out,
                const TensorInfo& tensorInfo,
                ActivationFunction function,
                float a,
                float b)
{
    unsigned int numElements = tensorInfo.GetNumElements();

    for (unsigned int i = 0; i < numElements; i++)
    {
        out.Set(Activation(in.Get(), function, a, b));
        ++in;
        ++out;
    }
    in -= numElements;
    out -= numElements;
}

} //namespace armnn

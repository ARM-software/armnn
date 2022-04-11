//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Stack.hpp"
#include "RefWorkloadUtils.hpp"

namespace armnn
{

void Stack(const StackQueueDescriptor& data,
           std::vector<std::unique_ptr<Decoder<float>>>& inputs,
           Encoder<float>& output,
           const TensorInfo& inputInfo,
           const TensorInfo& outputInfo)
{
    unsigned int outputNumDims = outputInfo.GetNumDimensions();
    unsigned int inputNumDims = inputInfo.GetNumDimensions();

    const armnn::TensorShape& outputDims = outputInfo.GetShape();
    const armnn::TensorShape& inputDims = inputInfo.GetShape();

    unsigned int axis = data.m_Parameters.m_Axis;

    // Can perform a simple concatenation when axis == 0
    if (!axis)
    {
        unsigned int numInputs = data.m_Parameters.m_NumInputs;
        unsigned int inputLength = inputInfo.GetNumElements();

        for (unsigned int inputIdx=0; inputIdx<numInputs; ++inputIdx)
        {
            for (unsigned int elmt=0; elmt<inputLength; ++elmt)
            {
                (*inputs[inputIdx])[elmt];
                output[(inputIdx * inputLength) + elmt];
                output.Set(inputs[inputIdx]->Get());
            }
        }
        return;
    }

    const unsigned int iNumTensors = static_cast<unsigned int>(data.m_Inputs.size());
    const unsigned int iBatchSize  = inputDims[0];
    const unsigned int iChannels   = (inputNumDims > 1) ? inputDims[1] : 1;
    const unsigned int iHeight     = (inputNumDims > 2) ? inputDims[2] : 1;
    const unsigned int iWidth      = (inputNumDims > 3) ? inputDims[3] : 1;

    const unsigned int oBatchSize  = outputDims[1];
    const unsigned int oChannels   = (outputNumDims > 2) ? outputDims[2] : 1;
    const unsigned int oHeight     = (outputNumDims > 3) ? outputDims[3] : 1;
    const unsigned int oWidth      = (outputNumDims > 4) ? outputDims[4] : 1;

    // Array to store the input coordinates
    // iCoordinates[0] = i, iCoordinates[1] = bi, iCoordinates[2] = ci
    // iCoordinates[3] = hi, iCoordinates[4] = wi, iCoordinates[5] = 0
    // iCoordinates[5] will be always zero and used for not incrementing
    // the output when the input has less than 4 dimensions
    std::array<unsigned int, 6> iCoordinates{ 0 };

    // Array of pointers used to map the output coordinates to the input ones, in accordance with the axis
    // This array is initialized with &iCoordinates[5] since this will be always zero
    std::array<unsigned int *, 5> oCoordinates = { &iCoordinates[5],
                                                   &iCoordinates[5],
                                                   &iCoordinates[5],
                                                   &iCoordinates[5],
                                                   &iCoordinates[5] };

    // Set the axis coordinate
    oCoordinates[axis] = &iCoordinates[0];

    // Map the output coordinates, accounting for the axis
    unsigned int dim_shift = 0;
    for(unsigned int dim = 0; dim < inputNumDims; ++dim)
    {
        if(dim == axis)
        {
            dim_shift++;
        }
        oCoordinates[dim + dim_shift] = &iCoordinates[dim + 1];
    }

    // Alias for the input coordinates
    unsigned int &i  = iCoordinates[0];
    unsigned int &bi = iCoordinates[1];
    unsigned int &ci = iCoordinates[2];
    unsigned int &hi = iCoordinates[3];
    unsigned int &wi = iCoordinates[4];

    // Alias for the output coordinates
    unsigned int &o  = *(oCoordinates[0]);
    unsigned int &bo = *(oCoordinates[1]);
    unsigned int &co = *(oCoordinates[2]);
    unsigned int &ho = *(oCoordinates[3]);
    unsigned int &wo = *(oCoordinates[4]);

    // Stack tensors
    for(; i < iNumTensors; ++(i))
    {
        for(bi = 0; bi < iBatchSize; ++(bi))
        {
            for(ci = 0; ci < iChannels; ++(ci))
            {
                for(hi = 0; hi < iHeight; ++(hi))
                {
                    for(wi = 0; wi < iWidth; ++(wi))
                    {
                        output[o  * oWidth * oHeight * oChannels * oBatchSize +
                               bo * oWidth * oHeight * oChannels +
                               co * oWidth * oHeight +
                               ho * oWidth +
                               wo];

                        output.Set(inputs[i]->Get());

                        ++(*(inputs[i]));
                    }
                }
            }
        }
    }
}

} // namespace armnn

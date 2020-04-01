//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DepthToSpace.hpp"

#include <armnnUtils/DataLayoutIndexed.hpp>
#include <armnnUtils/Permute.hpp>

#include <armnn/utility/Assert.hpp>

using namespace armnnUtils;

namespace armnn
{

void DepthToSpace(const TensorInfo& inputInfo,
                  const DepthToSpaceDescriptor& descriptor,
                  const void* inputData,
                  void* outputData,
                  unsigned int dataTypeSize)
{
    const unsigned int blockSize = descriptor.m_BlockSize;
    ARMNN_ASSERT(blockSize != 0u);

    const TensorShape& inputShape = inputInfo.GetShape();
    const unsigned int batches = inputShape[0];

    armnnUtils::DataLayoutIndexed dataLayoutIndexed(descriptor.m_DataLayout);
    const unsigned int inDepth  = inputShape[dataLayoutIndexed.GetChannelsIndex()];
    const unsigned int inHeight = inputShape[dataLayoutIndexed.GetHeightIndex()];
    const unsigned int inWidth  = inputShape[dataLayoutIndexed.GetWidthIndex()];

    const unsigned int outDepth = inDepth / (blockSize * blockSize);

    // The 4D input data can be interpreted as 6D (implicitly reshaped) as follows:
    //
    // [batch, block size, block size, inDepth, inHeight, inWidth] for NCHW and
    // [batch, inHeight, inWidth, blockSize, blockSize, outDepth] for NHWC.
    //
    // DepthToSpace can then be implemented as a permutation in 6D resulting in
    // the following shapes:
    //
    // [batch, outDepth, inHeight, blockSize, inWidth, blockSize] for NCHW and
    // [batch, inHeight, blockSize, inWidth, blockSize, outDepth] for NHWC.
    //
    // NOTE:
    // Since 6D tensors are not currently supported, in practice we need to handle each
    // batch separately and execute 5D permutations

    TensorShape permDestShape;
    PermutationVector permVector{};
    if (descriptor.m_DataLayout == DataLayout::NCHW)
    {
        permDestShape = TensorShape({ outDepth, inHeight, blockSize, inWidth, blockSize });
        permVector    = { 2, 4, 0, 1, 3 };
    }
    else
    {
        permDestShape = TensorShape({ inHeight, blockSize, inWidth, blockSize, outDepth });
        permVector    = { 0, 2, 1, 3, 4 };
    }

    const unsigned int numElementsPerBatch = inputShape.GetNumElements() / batches;

    for (unsigned int batchIndex = 0u; batchIndex < batches; ++batchIndex)
    {
        const uintptr_t batchDataOffset = batchIndex * (numElementsPerBatch * dataTypeSize);

        armnnUtils::Permute(permDestShape,
                            permVector,
                            static_cast<const void*>(reinterpret_cast<const uint8_t*>(inputData) + batchDataOffset),
                            static_cast<void*>(reinterpret_cast<uint8_t*>(outputData) + batchDataOffset),
                            dataTypeSize);
    }
}

} // namespace armnn

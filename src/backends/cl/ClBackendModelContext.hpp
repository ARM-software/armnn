//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/IBackendContext.hpp>

#include<string>

namespace armnn
{

/// The ClBackendModelContext is used to pass in CL specific backend ModelOptions. The supported backend ModelOptions
/// are:
///  - "FastMathEnabled"\n
///    Using the fast_math flag can lead to performance improvements in fp32 and fp16 layers but may result in\n
///    results with reduced or different precision. The fast_math flag will not have any effect on int8 performance.
///  - "SaveCachedNetwork"\n
///    Using the save_cached_network flag enables saving the cached network\n
///    to a file given with the cached_network_file_path option.
///  - "CachedNetworkFilePath"\n
///    If the cached_network_file_path is a valid file and the save_cached_network flag is present\n
///    then the cached network will be saved to the given file.\n
///    If the cached_network_file_path is a valid file and the save_cached_network flag is not present\n
///    then the cached network will be loaded from the given file.\n
///    This will remove the time taken for initial compilation of kernels and speed up the first execution.
class ClBackendModelContext : public IBackendModelContext
{
public:
    ClBackendModelContext(const ModelOptions& modelOptions);

    std::string GetCachedNetworkFilePath() const;

    bool IsFastMathEnabled() const;

    bool SaveCachedNetwork() const;

    int GetCachedFileDescriptor() const;

private:
    std::string m_CachedNetworkFilePath;
    bool m_IsFastMathEnabled;
    bool m_SaveCachedNetwork;
    int m_CachedFileDescriptor;

};

} // namespace armnn
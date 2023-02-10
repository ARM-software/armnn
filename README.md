<br>
<div align="center">
  <img src="Arm_NN_horizontal_blue.png" class="center" alt="Arm NN Logo" width="300"/>
</div>

* [Quick Start Guides](#quick-start-guides)
* [Pre-Built Binaries](#pre-built-binaries)
* [Software Overview](#software-overview)
* [Get Involved](#get-involved)
* [Contributions](#contributions)
* [Disclaimer](#disclaimer)
* [License](#license)
* [Third-Party](#third-party)
* [Build Flags](#build-flags)

# Arm NN

**Arm NN** is the **most performant** machine learning (ML) inference engine for Android and Linux, accelerating ML
on **Arm Cortex-A CPUs and Arm Mali GPUs**. This ML inference engine is an open source SDK which bridges the gap
between existing neural network frameworks and power-efficient Arm IP.

Arm NN outperforms generic ML libraries due to **Arm architecture-specific optimizations** (e.g. SVE2) by utilizing
**[Arm Compute Library (ACL)](https://github.com/ARM-software/ComputeLibrary/)**. To target Arm Ethos-N NPUs, Arm NN
utilizes the [Ethos-N NPU Driver](https://github.com/ARM-software/ethos-n-driver-stack). For Arm Cortex-M acceleration,
please see [CMSIS-NN](https://github.com/ARM-software/CMSIS_5).

Arm NN is written using portable **C++14** and built using [CMake](https://cmake.org/) - enabling builds for a wide
variety of target platforms, from a wide variety of host environments. **Python** developers can interface with Arm NN
through the use of our **Arm NN TF Lite Delegate**.


## Quick Start Guides
**The Arm NN TF Lite Delegate provides the widest ML operator support in Arm NN** and is an easy way to accelerate
your ML model. To start using the TF Lite Delegate, first download the **[Pre-Built Binaries](#pre-built-binaries)** for
the latest release of Arm NN. Using a Python interpreter, you can load your TF Lite model into the Arm NN TF Lite
Delegate and run accelerated inference. Please see this
**[Quick Start Guide](delegate/DelegateQuickStartGuide.md)** on GitHub or this more comprehensive
**[Arm Developer Guide](https://developer.arm.com/documentation/102561/latest/)** for information on how to accelerate
your TF Lite model using the Arm NN TF Lite Delegate.

The fastest way to integrate Arm NN into an **Android app** is by using our **Arm NN AAR (Android Archive) file with
Android Studio**. The AAR file nicely packages up the Arm NN TF Lite Delegate, Arm NN itself and ACL; ready to be
integrated into your Android ML application. Using the AAR allows you to benefit from the **vast operator support** of
the Arm NN TF Lite Delegate. We held an **[Arm AI Tech Talk](https://www.youtube.com/watch?v=Zu4v0nqq2FA)** on how to
accelerate an ML Image Segmentation app in 5 minutes using this AAR file. To download the Arm NN AAR file, please see the
**[Pre-Built Binaries](#pre-built-binaries)** section below.

We also provide Debian packages for Arm NN, which are a quick way to start using Arm NN and the TF Lite Parser
(albeit with less ML operator support than the TF Lite Delegate). There is an installation guide available
[here](InstallationViaAptRepository.md) which provides instructions on how to install the Arm NN Core and the TF Lite
Parser for Ubuntu 20.04.

To build Arm NN from scratch, we provide the **[Arm NN Build Tool](build-tool/README.md)**. This tool consists of
**parameterized bash scripts** accompanied by a **Dockerfile** for building Arm NN and its dependencies, including
**[Arm Compute Library (ACL)](https://github.com/ARM-software/ComputeLibrary/)**. This tool replaces/supersedes the
majority of the existing Arm NN build guides as a user-friendly way to build Arm NN. The main benefit of building
Arm NN from scratch is the ability to **exactly choose which components to build, targeted for your ML project**.<br>


## Pre-Built Binaries

| Operating System                              | Architecture-specific Release Archive (Download)                                                                                                                                                                                                                                                                                  |
|-----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Android (AAR)                                 | [![](https://img.shields.io/badge/download-android--aar-orange)](https://github.com/ARM-software/armnn/releases/download/v23.02/armnn_delegate_jni-23.02.aar)                                                                                                                                                                     |
| Android 10 "Q/Quince Tart" (API level 29)     | [![](https://img.shields.io/badge/download-arm64--v8.2a-blue)](https://github.com/ARM-software/armnn/releases/download/v23.02/ArmNN-android-29-arm64-v8.2-a.tar.gz) [![](https://img.shields.io/badge/download-arm64--v8a-red)](https://github.com/ARM-software/armnn/releases/download/v23.02/ArmNN-android-29-arm64-v8a.tar.gz) |
| Android 11 "R/Red Velvet Cake" (API level 30)     | [![](https://img.shields.io/badge/download-arm64--v8.2a-blue)](https://github.com/ARM-software/armnn/releases/download/v23.02/ArmNN-android-30-arm64-v8.2-a.tar.gz) [![](https://img.shields.io/badge/download-arm64--v8a-red)](https://github.com/ARM-software/armnn/releases/download/v23.02/ArmNN-android-30-arm64-v8a.tar.gz) |
| Android 12 "S/Snow Cone" (API level 31)     | [![](https://img.shields.io/badge/download-arm64--v8.2a-blue)](https://github.com/ARM-software/armnn/releases/download/v23.02/ArmNN-android-31-arm64-v8.2-a.tar.gz) [![](https://img.shields.io/badge/download-arm64--v8a-red)](https://github.com/ARM-software/armnn/releases/download/v23.02/ArmNN-android-31-arm64-v8a.tar.gz) |
| Android 13 "T/Tiramisu" (API level 32)     | [![](https://img.shields.io/badge/download-arm64--v8.2a-blue)](https://github.com/ARM-software/armnn/releases/download/v23.02/ArmNN-android-32-arm64-v8.2-a.tar.gz) [![](https://img.shields.io/badge/download-arm64--v8a-red)](https://github.com/ARM-software/armnn/releases/download/v23.02/ArmNN-android-32-arm64-v8a.tar.gz) |
| Linux                                         | [![](https://img.shields.io/badge/download-aarch64-green)](https://github.com/ARM-software/armnn/releases/download/v23.02/ArmNN-linux-aarch64.tar.gz) [![](https://img.shields.io/badge/download-x86__64-yellow)](https://github.com/ARM-software/armnn/releases/download/v23.02/ArmNN-linux-x86_64.tar.gz)                       |


## Software Overview
The Arm NN SDK supports ML models in **TensorFlow Lite** (TF Lite) and **ONNX** formats.

**Arm NN's TF Lite Delegate** accelerates TF Lite models through **Python or C++ APIs**. Supported TF Lite operators
are accelerated by Arm NN and any unsupported operators are delegated (fallback) to the reference TF Lite runtime -
ensuring extensive ML operator support. **The recommended way to use Arm NN is to
[convert your model to TF Lite format](https://www.tensorflow.org/lite/convert) and use the TF Lite Delegate.** Please
refer to the [Quick Start Guides](#quick-start-guides) for more information on how to use the TF Lite Delegate.

Arm NN also provides **TF Lite and ONNX parsers** which are C++ libraries for integrating TF Lite or ONNX models
into your ML application. Please note that these parsers do not provide extensive ML operator coverage as compared
to the Arm NN TF Lite Delegate.

**Android** ML application developers have a number of options for using Arm NN:
* Use our Arm NN AAR (Android Archive) file with **Android Studio** as described in the
[Quick Start Guides](#quick-start-guides) section
* Download and use our [Pre-Built Binaries](#pre-built-binaries) for the Android platform
* Build Arm NN from scratch with the Android NDK using this [GitHub guide](BuildGuideAndroidNDK.md)

Arm also provides an [Android-NN-Driver](https://github.com/ARM-software/android-nn-driver) which implements a
hardware abstraction layer (HAL) for the Android NNAPI. When the Android NN Driver is integrated on an Android device,
ML models used in Android applications will automatically be accelerated by Arm NN.

For more information about the Arm NN components, please refer to our
[documentation](https://github.com/ARM-software/armnn/wiki/Documentation).

Arm NN is a key component of the [machine learning platform](https://mlplatform.org/), which is part of the
[Linaro Machine Intelligence Initiative](https://www.linaro.org/news/linaro-announces-launch-of-machine-intelligence-initiative/).

For FAQs and troubleshooting advice, see the [FAQ](docs/FAQ.md) or take a look at previous
[GitHub Issues](https://github.com/ARM-software/armnn/issues).


## Get Involved
The best way to get involved is by using our software. If you need help or encounter an issue, please raise it as a
[GitHub Issue](https://github.com/ARM-software/armnn/issues). Feel free to have a look at any of our open issues too.
We also welcome feedback on our documentation.

Feature requests without a volunteer to implement them are closed, but have the 'Help wanted' label, these can be
found [here](https://github.com/ARM-software/armnn/issues?q=is%3Aissue+label%3A%22Help+wanted%22+).
Once you find a suitable Issue, feel free to re-open it and add a comment, so that Arm NN engineers know you are
working on it and can help.

When the feature is implemented the 'Help wanted' label will be removed.


## Contributions
The Arm NN project welcomes contributions. For more details on contributing to Arm NN please see the
[Contributing page](https://mlplatform.org/contributing/) on the [MLPlatform.org](https://mlplatform.org/) website,
or see the [Contributor Guide](CONTRIBUTING.md).

Particularly if you'd like to implement your own backend next to our CPU, GPU and NPU backends there are guides for
backend development: [Backend development guide](src/backends/README.md),
[Dynamic backend development guide](src/dynamic/README.md).


## Disclaimer
The armnn/tests directory contains tests used during Arm NN development. Many of them depend on third-party IP, model
protobufs and image files not distributed with Arm NN. The dependencies for some tests are available freely on
the Internet, for those who wish to experiment, but they won't run out of the box.


## License
Arm NN is provided under the [MIT](https://spdx.org/licenses/MIT.html) license.
See [LICENSE](LICENSE) for more information. Contributions to this project are accepted under the same license.

Individual files contain the following tag instead of the full license text.

    SPDX-License-Identifier: MIT

This enables machine processing of license information based on the SPDX License Identifiers that are available
here: http://spdx.org/licenses/


## Inclusive language commitment
Arm NN conforms to Arm's inclusive language policy and, to the best of our knowledge, does not contain any non-inclusive language.

If you find something that concerns you, please email terms@arm.com


## Third-party
Third party tools used by Arm NN:

| Tool           | License (SPDX ID) | Description                    | Version | Provenience
|----------------|-------------------|------------------------------------------------------------------|-------------|-------------------
| cxxopts        | MIT               | A lightweight C++ option parser library | SHA 12e496da3d486b87fa9df43edea65232ed852510 | https://github.com/jarro2783/cxxopts
| doctest        | MIT               | Header-only C++ testing framework | 2.4.6 | https://github.com/onqtam/doctest
| fmt            | MIT               | {fmt} is an open-source formatting library providing a fast and safe alternative to C stdio and C++ iostreams. | 7.0.1 | https://github.com/fmtlib/fmt
| ghc            | MIT               | A header-only single-file std::filesystem compatible helper library | 1.3.2 | https://github.com/gulrak/filesystem
| half           | MIT               | IEEE 754 conformant 16-bit half-precision floating point library | 1.12.0 | http://half.sourceforge.net
| mapbox/variant | BSD               | A header-only alternative to 'boost::variant' | 1.1.3 | https://github.com/mapbox/variant
| stb            | MIT               | Image loader, resize and writer | 2.16 | https://github.com/nothings/stb


## Build Flags
Arm NN uses the following security related build flags in their code:

| Build flags	      |
|---------------------|
| -Wall	              |
| -Wextra             |
| -Wold-style-cast    |
| -Wno-missing-braces |
| -Wconversion        |
| -Wsign-conversion   |
| -Werror             |

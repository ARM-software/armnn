# Introduction

* [Software tools overview](#software-tools-overview)
* [Where to find more information](#where-to-find-more-information)
* [Contributions](#contributions)
* [Disclaimer](#disclaimer)
* [License](#license)
* [Third-Party](#third-party)

Arm NN is a key component of the [machine learning platform](https://mlplatform.org/), which is part of the
[Linaro Machine Intelligence Initiative](https://www.linaro.org/news/linaro-announces-launch-of-machine-intelligence-initiative/).

The Arm NN SDK is a set of open-source software and tools that enables machine learning workloads on power-efficient
devices. It provides a bridge between existing neural network frameworks and power-efficient Cortex-A CPUs,
Arm Mali GPUs and Arm Ethos NPUs.

<img align="center" width="400" src="https://developer.arm.com/-/media/Arm Developer Community/Images/Block Diagrams/Arm-NN/Arm-NN-Frameworks-Diagram.png"/>

Arm NN SDK utilizes the Compute Library to target programmable cores, such as Cortex-A CPUs and Mali GPUs,
as efficiently as possible. To target Ethos NPUs the NPU-Driver is utilized. We also welcome new contributors to provide
their [own driver and backend](src/backends/README.md). Note, Arm NN does not provide support for Cortex-M CPUs.

The latest release supports models created with **Caffe**, **TensorFlow**, **TensorFlow Lite** (TfLite) and **ONNX**.
Arm NN analysis a given model and replaces the operations within it with implementations particularly designed for the
hardware you want to execute it on. This results in a great boost of execution speed. How much faster your neural
network can be executed depends on the operations it contains and the available hardware. Below you can see the speedup
we've been experiencing in our experiments with a few common networks.

<img align="center" width="700" src="https://developer.arm.com/-/media/developer/Other Images/Arm_NN_performance_relative_to_other_NN_frameworks_diagram.png"/>

Arm NN is written using portable C++14 and the build system uses [CMake](https://cmake.org/), therefore it is possible
to build for a wide variety of target platforms, from a wide variety of host environments.


## Getting started: Software tools overview
Depending on what kind of framework (Tensorflow, Caffe, ONNX) you've been using to create your model there are multiple
software tools available within Arm NN that can serve your needs.

Generally, there is a **parser** available **for each supported framework**. Each parser allows you to run models from
one framework e.g. the TfLite-Parser lets you run TfLite models. You can integrate these parsers into your own
application to load, optimize and execute your model. We also provide **python bindings** for our parsers and the Arm NN core.
We call the result **PyArmNN**. Therefore your application can be conveniently written in either C++ using the "original"
Arm NN library or in Python using PyArmNN. You can find tutorials on how to setup and use our parsers in our doxygen
documentation. The latest version can be found in the [wiki section](https://github.com/ARM-software/armnn/wiki/Documentation)
of this repository.

Admittedly, building Arm NN and its parsers from source is not always easy to accomplish. We are trying to increase our
usability by providing Arm NN as a **Debian package**. Our debian package is the most easy way to install the Arm NN Core,
the TfLite Parser and PyArmNN (More support is about to come): [Installation via Apt Repository](InstallationViaAptRepository.md)

The newest member in Arm NNs software toolkit is the **TfLite Delegate**. The delegate can be integrated in TfLite.
TfLite will then delegate operations, that can be accelerated with Arm NN, to Arm NN. Every other operation will still be
executed with the usual TfLite runtime. This is our **recommended way to accelerate TfLite models**. As with our parsers
there are tutorials in our doxygen documentation that can be found in the [wiki section](https://github.com/ARM-software/armnn/wiki/Documentation).

If you would like to use **Arm NN on Android** you can follow this guide which explains [how to build Arm NN using the AndroidNDK]().
But you might also want to take a look at another repository which implements a hardware abstraction layer (HAL) for
Android. The repository is called [Android-NN-Driver](https://github.com/ARM-software/android-nn-driver) and when
integrated into Android it will automatically run neural networks with Arm NN.


## Where to find more information
The section above introduces the most important tools that Arm NN provides.
You can find a complete list in our **doxygen documentation**. The
latest version can be found in the [wiki section](https://github.com/ARM-software/armnn/wiki/Documentation) of our github
repository.

For FAQs and troubleshooting advice, see [FAQ.md](docs/FAQ.md)
or take a look at previous [github issues](https://github.com/ARM-software/armnn/issues).


## Note
1. The following tools are **deprecated** in Arm NN 21.02 and will be removed in 21.05:
    * TensorflowParser
    * CaffeParser
    * Quantizer

2. Ubuntu Linux 16.04 LTS will no longer be supported by April 30, 2021.
   At that time, Ubuntu 16.04 LTS will no longer receive security patches or other software updates.
   Consequently Arm NN will from the 21.08 Release at the end of August 2021 no longer be officially
   supported on Ubuntu 16.04 LTS but will instead be supported on Ubuntu 18.04 LTS.

3. We are currently in the process of removing [boost](https://www.boost.org/) as a dependency to Arm NN. This process
   is finished for everything apart from our unit tests. This means you don't need boost to build and use Arm NN but
   you need it to execute our unit tests. Boost will soon be removed from Arm NN entirely.


## Contributions
The Arm NN project welcomes contributions. For more details on contributing to Arm NN see the [Contributing page](https://mlplatform.org/contributing/)
on the [MLPlatform.org](https://mlplatform.org/) website, or see the [Contributor Guide](ContributorGuide.md).

Particularly if you'd like to implement your own backend next to our CPU, GPU and NPU backends there are guides for
backend development:
[Backend development guide](src/backends/README.md), [Dynamic backend development guide](src/dynamic/README.md)


## Disclaimer
The armnn/tests directory contains tests used during Arm NN development. Many of them depend on third-party IP, model
protobufs and image files not distributed with Arm NN. The dependencies of some of the tests are available freely on
the Internet, for those who wish to experiment, but they won't run out of the box.


## License
Arm NN is provided under the [MIT](https://spdx.org/licenses/MIT.html) license.
See [LICENSE](LICENSE) for more information. Contributions to this project are accepted under the same license.

Individual files contain the following tag instead of the full license text.

    SPDX-License-Identifier: MIT

This enables machine processing of license information based on the SPDX License Identifiers that are available here: http://spdx.org/licenses/


## Third-party
Third party tools used by Arm NN:

| Tool           | License (SPDX ID) | Description                    | Version | Provenience
|----------------|-------------------|------------------------------------------------------------------|-------------|-------------------
| cxxopts        | MIT               | A lightweight C++ option parser library | SHA 12e496da3d486b87fa9df43edea65232ed852510 | https://github.com/jarro2783/cxxopts
| fmt            | MIT               | {fmt} is an open-source formatting library providing a fast and safe alternative to C stdio and C++ iostreams. | 7.0.1 | https://github.com/fmtlib/fmt
| ghc            | MIT               | A header-only single-file std::filesystem compatible helper library | 1.3.2 | https://github.com/gulrak/filesystem
| half           | MIT               | IEEE 754 conformant 16-bit half-precision floating point library | 1.12.0 | http://half.sourceforge.net
| mapbox/variant | BSD               | A header-only alternative to 'boost::variant' | 1.1.3 | https://github.com/mapbox/variant
| stb            | MIT               | Image loader, resize and writer | 2.16 | https://github.com/nothings/stb

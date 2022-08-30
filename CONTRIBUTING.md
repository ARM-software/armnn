# Contribution Guidelines

The Arm NN project is open for external contributors and welcomes contributions. Arm NN is licensed under the [MIT license](https://spdx.org/licenses/MIT.html) and all accepted contributions must have the same license. Below is an overview on contributing code to ArmNN. For more details on contributing to Arm NN see the [Contributing page](https://mlplatform.org/contributing/) on the [MLPlatform.org](https://mlplatform.org/) website.

## Contributing code to Arm NN

- All code reviews are performed on [Linaro ML Platform Gerrit](https://review.mlplatform.org)
- GitHub account credentials are required for creating an account on ML Platform
- Setup Arm NN git repo
  - git clone https://review.mlplatform.org/ml/armnn
  - cd armnn
  - git checkout main
  - git pull (not required upon initial clone but good practice before creating a patch)
  - git config user.name "FIRST_NAME SECOND_NAME"
  - git config user.email your@email.address
- Commit using sign-off and push patch for code review
  - git commit -s
  - git push origin HEAD:refs/for/main
- Patch will appear on ML Platform Gerrit [here](https://review.mlplatform.org/q/is:open+project:ml/armnn+branch:main)
- See below for adding details of copyright notice and developer certificate
of origin sign off

## Developer Certificate of Origin (DCO)

Before the Arm NN project accepts your contribution, you need to certify its origin and give us your permission.  To manage this process we use Developer Certificate of Origin (DCO) V1.1 (https://developercertificate.org/).

To indicate that you agree to the the terms of the DCO, you "sign off" your contribution by adding a line with your name and e-mail address to every git commit message:

Signed-off-by: John Doe <john.doe@example.org>

You must use your real name, no pseudonyms or anonymous contributions are accepted.

## In File Copyright Notice

In each source file, include the following copyright notice:

//  
// Copyright © `<years additions were made to project> <your name>` and Contributors. All rights reserved.  
// SPDX-License-Identifier: MIT  
//

Note: if an existing file does not conform, update it when you next modify it, as convenient.

## Releases

Official Arm NN releases are published through the official [Arm NN Github repository](https://github.com/ARM-software/armnn).

## Development repository

The Arm NN development repository is hosted on the [mlplatform.org git repository](https://git.mlplatform.org/ml/armnn.git/) hosted by [Linaro](https://www.linaro.org/).

## Code reviews

Contributions must go through code review. Code reviews are performed through the [mlplatform.org Gerrit server](https://review.mlplatform.org). Contributors need to signup to this Gerrit server with their GitHub account
credentials.

Only reviewed contributions can go to the main branch of Arm NN.

## Continuous integration

Contributions to Arm NN go through testing at the Arm CI system. All unit, integration and regression tests must pass before a contribution gets merged to the Arm NN main branch.

## Communications

We encourage all Arm NN developers to subscribe to the [Arm NN developer mailing list](https://lists.linaro.org/mailman/listinfo/armnn-dev).

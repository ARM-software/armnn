# Contribution Guidelines

The Arm NN project is open for external contributors and welcomes contributions. Arm NN is licensed under the [MIT license](https://spdx.org/licenses/MIT.html) and all accepted contributions must have the same license. Below is an overview on contributing code to Arm NN.

## Contributing code to Arm NN

All code contributions follow a Pull Request (PR) based workflow:

- Fork the repository: https://github.com/ARM-software/armnn
- Create a branch from the latest main:
  - git checkout main
  - git pull (not required upon initial clone but good practice before creating a patch)
  - git config user.name "FIRST_NAME SECOND_NAME"
  - git config user.email your@email.address
  - git checkout -b your-branch-name
- Commit using sign-off. Keep changes to one commit per PR. Ensure each commit message includes a Signed-off-by line (see DCO section below). Use:
  - git commit -s
- Push your branch to your fork:
  - git push -u <your-remote> your-branch-name
- Open a PR against ARM-software/armnn main.
- Address review feedback by amending, then force-pushing the branch (git push -f). Keep the change in one commit.

See below for adding details of copyright notice and developer certificate
of origin sign off.

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

Official Arm NN releases are published through the [Arm NN Github repository](https://github.com/ARM-software/armnn).
## Code reviews

Contributions must go through code review. Code reviews are performed through the [Arm NN Github repository](https://github.com/ARM-software/armnn).

Only reviewed contributions can go to the main branch of Arm NN.
## Continuous integration

Contributions to Arm NN go through testing at the Arm CI system. All unit, integration and regression tests must pass before a contribution gets merged to the Arm NN main branch.

## Communications

We encourage all Arm NN developers to subscribe to the [Arm NN developer mailing list](https://lists.linaro.org/mailman3/lists/armnn-dev.lists.linaro.org/).

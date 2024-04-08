# Contributing to dcm-classifier

Contributions to the dcm-classifier package are welcome. Please follow the instructions below to contribute to the package.

## Process

To contribute to the dcm-classifier package, fork the repository off main and clone the forked repository to your local machine.

`$ git clone https://github.com/your-username/dcm-classifier.git`

Add the upstream repository

`$ git remote add upstream https://github.com/BRAINSia/dcm-classifier.git`

Pull the latest changes from the upstream repository

`$ git checkout main`

`$ git pull upstream main`

Create a new branch for your changes

`$ git switch -c <feature_branch>`

Choose a descriptive branch name that reflects the changes you are making.

Push the branch with your changes to your forked repository

`$ git push origin <feature_branch>`

After making your changes, create a pull request against the main branch of the upstream repository. The pull request will be reviewed by the maintainers and merged if approved.

## Developer Instructions

### Commit Messages

Commit messages should follow the format of:
```
Prefix: commit message
```
Where the prefix is one of the following:
- BUG: Fix for runtime crash or incorrect result
- COMP: Compiler error or warning fix
- DOC: Documentation change
- ENH: New functionality
- PERF: Performance improvement
- STYLE: No logic impact (indentation, comments)
- WIP: Work In Progress not ready for merge

### Pull Requests

Pull requests should be made against the `main` branch. Use the pull request template provided in the repository to ensure all necessary information is included. For testing changes, please refer to the [README](README.md) file. Additionally, tests will be run automatically on the pull request to ensure the changes do not break the package.

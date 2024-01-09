This package follows semantaic versioning. The releases fall into three categories: major, minor, and patch.

# Release categories
## Major releases

Major releases are for major changes to the package. These include changes to the API, changes to the package's dependencies, and changes to the package's build process. Major releases are indicated by incrementing the first number in the version number.

## Minor releases

Minor releases are for minor changes to the package. These include adding new features to the package, and making backwards-compatible changes to the package's API. Minor releases are indicated by incrementing the second number in the version number.

## Patch releases

Patch releases are for bug fixes and other small changes to the package. These include bug fixes, documentation updates, and refactoring. Patch releases are indicated by incrementing the third number in the version number.

# Development and release procedure

(In terms of release procedure, major releases are just minor releases with 0.0 as minor and patch version numbers.)

1. We generally do development on the master branch. By making pull requests to master branch.
2. When we want to release a new version, we start the release process. Typically once we have added a new feature that is ready for external users or we have fixed a bug that is affecting a released version.
2.1.1. Run the tests and make sure they pass.
2.1.2. Create and switch to a new branch from master named `release-<mm-version>`, where `<mm-version>` is the new version number without patch version. If we are creating a patch release, then just switch to the `release-<mm-version>` branch.
2.1.3. If creating patch release, then cherry-pick the commits that are to be included in the patch release from master to the `release-<mm-version>` branch using `git cherry-pick <commit-hash>`. If creating a minor or major release, then skip this step.
2.1.4. Update the `<version>` number in the VERSION file where version is the full semver with patch version.
2.1.5. Run
```
git commit -a -m "Bump version number to <version>."
# And then tag the commit with
git tag -a "release-<version>" -m "Tagging version <version>."
git push origin
git push origin "release-<version>"
# TODO: Run commands to push the package to PyPI and conda-forge
# TODO: Run commands to generate the documentation and push it to Github pages
```
2.1.6. Now you should have tag and branch in your Github repo.

A Git tag is an immutable named reference to a commit in your Git repo.

A user that wants to run tests or explore examples of a specific version of the package can checkout the tag and run the tests or explore the examples.

```
git clone <repo_url> --branch "release-<version>"
```

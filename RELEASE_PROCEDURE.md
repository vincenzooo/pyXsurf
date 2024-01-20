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
2.1.3. If creating a minor or major release, you can skip this step. Otherwise, if creating patch release, then cherry-pick the commits that are to be included in the patch release from master to the `release-<mm-version>` branch using `git cherry-pick <commit-hash>`.
2.1.4. Update the `<version>` number in the VERSION file where version is the full semver with patch version.
2.1.5. Run
```
git commit -a -m "Bump version number to <version>."
# And then tag the commit with
git tag -a "release-<version>" -m "Tagging version <version>."
git push origin
git push origin "release-<version>"
```
2.1.6. Now you should have tag and branch in your Github repo.

3. Follow the procedures in Publish section to publish artifacts to Github, PyPI, and conda-forge.

A Git tag is an immutable named reference to a commit in your Git repo.

A user that wants to run tests or explore examples of a specific version of the package can checkout the tag and run the tests or explore the examples.

```
git clone <repo_url> --branch "release-<version>"
```

# Publish

## Publish to Github

1. Install the Github client: `conda install gh --channel conda-forge`
2. Run `gh auth login` and follow the instructions to login to Github.

```
> gh auth login
? What account do you want to log into? GitHub.com
? What is your preferred protocol for Git operations on this host? HTTPS
? Authenticate Git with your GitHub credentials? No
? How would you like to authenticate GitHub CLI? Login with a web browser
```

3. Run `pip install build`
4. Run to build the package: `python -m build`
4. Run `gh release create release-<version> dist/pyXsurf-<version>-py3-none-any.whl dist/pyXsurf-<version>.tar.gz` to create a release on Github.

```
> gh release create release-2.0.0.dev1 dist/pyXsurf-2.0.0.dev1-py3-none-any.whl dist/pyXsurf-2.0.0.dev1.tar.gz
? Title (optional) release-2.0.0.dev1
? Release notes Leave blank
? Is this a prerelease? No
? Submit? Publish release
https://github.com/robeyes/pyXsurf/releases/tag/release-2.0.0.dev1
```

## Publish to PyPI

1. Create a PyPI account if you don't have one already.
1.1. Follow instructions [here](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives) to create auth tokens.
2. Run `pip install build twine`
3. Run to build the package: `python -m build`
4. Upload to PyPI with: `twine upload dist/pyXsurf-<version>-py3-none-any.whl`

## Publish to conda-forge

1. Run `pip install build grayskull`
2. Run to build the package: `python -m build`
3. Run `grayskull pypi --tag release-<version> https://github.com/robeyes/pyXsurf`
3.1. This will create a `meta.yaml` file in `pyXurf` directory.
4. Using the `meta.yaml` created in above step, follow steps from step 4 in this guide: https://blog.gishub.org/how-to-publish-a-python-package-on-conda-forge

## Publish docs to Github pages

TODO

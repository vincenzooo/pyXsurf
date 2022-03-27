"""Attempt to generate package and dependencies information 
from https://stackoverflow.com/questions/29751572/how-to-find-a-python-packages-dependencies"""

import pkg_resources

def print_all_in_working_set():
    ws = pkg_resources.working_set
    for package_metadata in ws:
        print(package_metadata)
        
#_package_name = 'yourpackagename'
  
def get_dependencies_with_semver_string(package_name):
    package = pkg_resources.working_set.by_key[package_name]
    return [str(r) for r in package.requires()]

def retrieve_info(package_name):
    """ get dependencies from setup.py"""
    from pip._vendor import pkg_resources
    #_package_name = 'somepackage'
    _package = pkg_resources.working_set.by_key[package_name]

    print([str(r) for r in _package.requires()])  # retrieve deps from setup.py

if __name__ == "__main__":
    _package_name = 'pyxsurf'
    print("ALL PACKAGES:")
    get_dependencies_with_semver_string(_package_name)
    print("DEPENDENCIES:")
    print_all_in_working_set()
    print("INFO:")
    retrieve_info(_package_name)
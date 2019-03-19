#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def python_version_check():
  import sys

  assert sys.version_info.major == 3 and sys.version_info.minor >= 6, (
    f'This project is utilises language features only present Python 3.6 and greater. '
    f'You are running {sys.version_info}.')


python_version_check()

from NeodroidAgentPackage import NeodroidAgentPackage

__author__ = 'cnheider'

from setuptools import setup

if __name__ == '__main__':

  pkg = NeodroidAgentPackage()

  setup(name=pkg.package_name,
        version=pkg.version,
        packages=pkg.packages,
        package_data=pkg.package_data,
        author=pkg.author_name,
        author_email=pkg.author_email,
        maintainer=pkg.maintainer_name,
        maintainer_email=pkg.maintainer_email,
        description=pkg.description,
        license=pkg.license,
        keywords=pkg.keyword,
        url=pkg.url,
        download_url=pkg.download_url,
        install_requires=pkg.requirements,
        extras_require=pkg.extras,
        entry_points=pkg.entry_points,
        classifiers=pkg.classifiers,
        long_description_content_type=pkg.readme_type,
        long_description=pkg.readme,
        tests_require=pkg.dependencies_testing,
        setup_requires=pkg.setup_dependencies,
        include_package_data=True,
        python_requires='>=3'
        )

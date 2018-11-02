#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from NeodroidAgentPackage import NeodroidAgentPackage

__author__ = 'cnheider'

from setuptools import setup

if __name__ == '__main__':

  neo_agent_pkg = NeodroidAgentPackage()

  setup(
      name=neo_agent_pkg.package_name,
      version=neo_agent_pkg.version,
      packages=neo_agent_pkg.packages,
      package_data=neo_agent_pkg.package_data,
      author=neo_agent_pkg.author_name,
      author_email=neo_agent_pkg.author_email,
      maintainer=neo_agent_pkg.maintainer_name,
      maintainer_email=neo_agent_pkg.maintainer_email,
      description=neo_agent_pkg.description,
      license=neo_agent_pkg.license,
      keywords=neo_agent_pkg.keyword,
      url=neo_agent_pkg.url,
      download_url=neo_agent_pkg.download_url,
      install_requires=neo_agent_pkg.requirements,
      extras_require=neo_agent_pkg.extras,
      entry_points=neo_agent_pkg.entry_points,
      classifiers=neo_agent_pkg.classifiers,
      long_description_content_type=neo_agent_pkg.readme_type,
      long_description=neo_agent_pkg.readme,
      tests_require=neo_agent_pkg.test_dependencies,
      include_package_data=True,
      python_requires='>=3'
      )

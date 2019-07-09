#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def python_version_check():
  import sys

  assert sys.version_info.major == 3 and sys.version_info.minor >= 6, (
    f'This project is utilises language features only present Python 3.6 and greater. '
    f'You are running {sys.version_info}.')


python_version_check()
import pathlib
import re

from setuptools import find_packages

with open(pathlib.Path(__file__).parent / "agent" / "version.py", "r") as f:
  content = f.read()
  # get version string from module
  version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", content, re.M).group(1)

  project_name = re.search(r"PROJECT_NAME = ['\"]([^'\"]*)['\"]", content, re.M).group(1)

__author__ = 'cnheider'

from setuptools import setup


class NeodroidAgentPackageMeta(type):

  @property
  def dependencies_testing(self) -> list:
    return [
      'pytest',
      'mock'
      ]

  @property
  def setup_dependencies(self) -> list:
    return [
      'pytest-runner'
      ]

  @property
  def package_name(self) -> str:
    return project_name

  @property
  def url(self) -> str:
    return 'https://github.com/sintefneodroid/agent'

  @property
  def download_url(self):
    return self.url + '/releases'

  @property
  def readme_type(self):
    return 'text/markdown'

  @property
  def packages(self):
    return find_packages(
        exclude=[
          # 'neodroid/environment_utilities'
          ]
        )

  @property
  def author_name(self):
    return 'Christian Heider Nielsen'

  @property
  def author_email(self):
    return 'cnheider@yandex.com'

  @property
  def maintainer_name(self):
    return self.author_name

  @property
  def maintainer_email(self):
    return self.author_email

  @property
  def package_data(self):
    # data = glob.glob('environment_utilities/mab/**', recursive=True)
    return {
      # 'neodroid':[
      # *data
      # 'environment_utilities/mab/**',
      # 'environment_utilities/mab/**_Data/*',
      # 'environment_utilities/mab/windows/*'
      # 'environment_utilities/mab/windows/*_Data/*'
      #  ]
      }

  @property
  def entry_points(self):
    return {
      'console_scripts':[
        # "name_of_executable = module.with:function_to_execute"
        'neodroid-tab = agent.agents.experimental.tabular_q_agent:tabular_test',
        'neodroid-rnd = agent.agents.experimental.random_agent:random_test',

        'neodroid-ppo-gym = agent.agents.model_free.policy_optimisation.ppo_agent:ddpg_test',
        'neodroid-dqn-gym = agent.agents.model_free.q_learning.dqn_agent:dqn_test',
        'neodroid-pg-gym = agent.agents.model_free.policy_optimisation.pg_agent:pg_test',
        'neodroid-ddpg-gym = agent.agents.model_free.hybrid.ddpg_agent:ddpg_test',

        'neodroid-ppo = agent.agents.model_free.policy_optimisation.ppo_agent:ppo_run',
        'neodroid-dqn = agent.agents.model_free.q_learning.dqn_agent:dqn_run',
        'neodroid-pg = agent.agents.model_free.policy_optimisation.pg_agent:pg_run',
        'neodroid-ddpg = agent.agents.model_free.hybrid.ddpg_agent:ddpg_run',

        'neodroid-cma = agent.agents.experimental.cma_agent:cma_test',

        'neodroid-tb = agent.utilities.extra_entry_points.tensorboard_entry_point:main',
        ]
      }

  @property
  def extras(self):
    these_extras = {
      # 'GUI':['kivy'],
      # 'mab':['neodroid-linux-mab; platform_system == "Linux"',
      #       'neodroid-win-mab platform_system == "Windows"']

      }

    all_dependencies = []

    for group_name in these_extras:
      all_dependencies += these_extras[group_name]
    these_extras['all'] = all_dependencies

    return these_extras

  @property
  def requirements(self) -> list:
    requirements_out = []
    with open('requirements.txt') as f:
      requirements = f.readlines()

      for requirement in requirements:
        requirements_out.append(requirement.strip())

    return requirements_out

  @property
  def description(self):
    return 'Reinforcement learning agent implementations, intended for use with the Neodroid platform'

  @property
  def readme(self):
    with open('README.md') as f:
      return f.read()

  @property
  def keyword(self):
    with open('KEYWORDS.md') as f:
      return f.read()

  @property
  def license(self):
    return 'Apache License, Version 2.0'

  @property
  def classifiers(self):
    return [
      'Development Status :: 4 - Beta',
      'Environment :: Console',
      'Intended Audience :: End Users/Desktop',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: Apache Software License',
      'Operating System :: MacOS :: MacOS X',
      'Operating System :: Microsoft :: Windows',
      'Operating System :: POSIX',
      'Operating System :: OS Independent',
      'Programming Language :: Python :: 3',
      'Natural Language :: English',
      # 'Topic :: Scientific/Engineering :: Artificial Intelligence'
      # 'Topic :: Software Development :: Bug Tracking',
      ]

  @property
  def version(self):
    return version


class NeodroidAgentPackage(metaclass=NeodroidAgentPackageMeta):
  pass


if __name__ == '__main__':

  pkg = NeodroidAgentPackage

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

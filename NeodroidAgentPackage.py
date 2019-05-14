from setuptools import find_packages

import os
import re

with open(os.path.join(os.path.dirname(__file__), "agent/version.py"), "r") as f:
  # get version string from module
  version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M).group(1)


class NeodroidAgentPackage:

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
    return 'NeodroidAgent'

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
        'neodroid-ppo = agent.agents.ppo_agent:ppo_test',
        'neodroid-dqn = agent.agents.dqn_agent:dqn_test',
        'neodroid-pg = agent.agents.pg_agent:pg_test',
        'neodroid-ddpg = agent.agents.dppg_agent:ddpg_test',
        'neodroid-cma = agent.agents.experimental.cma_agent:cma_test'
        # 'neodroid-evo = agents.evo_agent:main'
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

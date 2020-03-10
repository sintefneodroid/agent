#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import find_packages, setup


def python_version_check(major=3, minor=6):
    import sys

    assert sys.version_info.major == major and sys.version_info.minor >= minor, (
        f"This project is utilises language features only present Python {major}.{minor} and greater. "
        f"You are running {sys.version_info}."
    )


python_version_check()
import pathlib
import re

with open(
    pathlib.Path(__file__).parent / "neodroidagent" / "__init__.py", "r"
) as project_init_file:
    str_reg_exp = "['\"]([^'\"]*)['\"]"
    content = project_init_file.read()  # get strings from module
    version = re.search(rf"__version__ = {str_reg_exp}", content, re.M).group(1)
    project_name = re.search(rf"__project__ = {str_reg_exp}", content, re.M).group(1)
    author = re.search(rf"__author__ = {str_reg_exp}", content, re.M).group(1)

__author__ = author


class NeodroidAgentPackage:
    @property
    def dependencies_testing(self) -> list:
        return ["pytest", "mock"]

    @property
    def setup_dependencies(self) -> list:
        return ["pytest-runner"]

    @property
    def package_name(self) -> str:
        return project_name

    @property
    def url(self) -> str:
        return "https://github.com/sintefneodroid/agent"

    @property
    def download_url(self):
        return self.url + "/releases"

    @property
    def readme_type(self):
        return "text/markdown"

    @property
    def packages(self):
        return find_packages(
            exclude=[
                # 'Path/To/Exclude'
            ]
        )

    @property
    def author_name(self):
        return author

    @property
    def author_email(self):
        return "christian.heider@alexandra.dk"

    @property
    def maintainer_name(self):
        return self.author_name

    @property
    def maintainer_email(self):
        return self.author_email

    @property
    def package_data(self):
        # data = glob.glob('data/', recursive=True)
        return {
            # 'PackageName':[
            # *data
            #  ]
        }

    @property
    def entry_points(self):
        return {
            "console_scripts": [
                # "name_of_executable = module.with:function_to_execute"
                "neodroid-agent = neodroidagent.entry_points.cli:main",
                "neodroid-tb = neodroidagent.entry_points.tensorboard_entry_point:main",
                "neodroid-clean-all = neodroidagent.entry_points.clean:clean_all",
                "neodroid-open-data = neodroidagent.entry_points.open_data:main",
            ]
        }

    @property
    def extras(self):
        these_extras = {
            # 'ExtraName':['package-name; platform_system == "System(Linux,Windows)"'
        }

        path: pathlib.Path = pathlib.Path(__file__).parent

        for file in path.iterdir():
            if file.name.startswith("requirements_"):

                requirements_group = []
                with open(str(file.absolute())) as f:
                    requirements = f.readlines()

                    for requirement in requirements:
                        requirements_group.append(requirement.strip())

                group_name_ = "_".join(file.name.strip(".txt").split("_")[1:])

                these_extras[group_name_] = requirements_group

        all_dependencies = []

        for group_name in these_extras:
            all_dependencies += these_extras[group_name]
        these_extras["all"] = all_dependencies

        return these_extras

    @property
    def requirements(self) -> list:
        requirements_out = []
        with open("requirements.txt") as f:
            requirements = f.readlines()

            for requirement in requirements:
                requirements_out.append(requirement.strip())

        return requirements_out

    @property
    def description(self):
        return "Reinforcement learning agent implementations, intended for use with the Neodroid platform"

    @property
    def readme(self):
        with open("README.md") as f:
            return f.read()

    @property
    def keyword(self):
        with open("KEYWORDS.md") as f:
            return f.read()

    @property
    def license(self):
        return "Apache License, Version 2.0"

    @property
    def classifiers(self):
        return [
            "Development Status :: 4 - Beta",
            "Environment :: Console",
            "Intended Audience :: End Users/Desktop",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Natural Language :: English",
            # 'Topic :: Scientific/Engineering :: Artificial Intelligence'
            # 'Topic :: Software Development :: Bug Tracking',
        ]

    @property
    def version(self):
        return version


if __name__ == "__main__":

    pkg = NeodroidAgentPackage()

    setup(
        name=pkg.package_name,
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
        python_requires=">=3.6",
    )

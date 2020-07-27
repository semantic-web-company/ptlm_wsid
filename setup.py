#!/usr/bin/env python

from setuptools import setup, find_packages
from setuptools.command.install import install as _install


class Install(_install):
    def run(self):
        _install.do_egg_install(self)
        import nltk
        nltk.download("all")

setup(
    cmdclass={'install': Install},
    install_requires=[
        'nltk',
    ],
    setup_requires=['pbr', 'nltk'],
    pbr=True,
)
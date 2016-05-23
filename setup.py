#!/usr/bin/python
# -*- coding: utf-8 -*-

from setuptools import setup
import glob


setup(name='sparsenetworks',
      version='0.1',
      description='simple simulation framework for sparse neural networks',
      url='',
      author='Ilyas Kuhlemann, Burooj Ghani, Francesca Schönsberg, Diemut Regel',
      author_email='ilyasp.ku@gmail.com',
      license='MIT',
      packages=['sparsenetworks'],#,'sparsenetworks/executables'],
      scripts=glob.glob('sparsenetworks/executables/*.py'),
      install_requires=['numpy','scipy','matplotlib'],
      zip_safe=False)

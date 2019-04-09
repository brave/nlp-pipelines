from setuptools import setup
from setuptools import find_packages

long_description = '''
nlp_pipelines is a simple api to define nlp processing pipelines:  
Specifically:

- Allows to specify a nlp pipeline as an execution graph.
- Save and restore execution graphs exported to protobuf

'''

setup(name='nlp-pipelines',
      version='0.0.1',
      description='Simple nlp pipelines',
      long_description=long_description,
      author='Dimitrios Athanasakis',
      author_email='dathanasakis@brave.com',
      url='',
      install_requires=['numpy>=1.9.1',
                        'scipy>=0.14',
                        'six>=1.9.0',
                        'pyyaml',
                        'h5py',
                        'keras_applications>=1.0.6',
                        'keras_preprocessing>=1.0.5'],
      packages=find_packages())

from setuptools import setup

setup(
    name='template_builder',
    version='0.2',
    packages=['template_builder'],
    package_dir={'template_builder': 'template_builder'},
    package_data={'template_builder': "configs/*"},
    include_package_data=True,
    url='',
    license='',
    author='parsonsrd',
    author_email='',
    description='Creation tools for building ImPACT templates for ctapipe'
)

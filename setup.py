from setuptools import find_packages,setup

HYPEN_E_DOT = "-e ."

def get_requirements(file_path: str)-> list[str]:
    requirements = []
    with open(file_path) as fp:
        requirements = fp.read().splitlines()
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='mlproject',
    packages=find_packages(),
    version='0.0.1',
    description='A short description of the project.',
    author='Maksym',
    author_email='maxm13052004@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    license='MIT',
)
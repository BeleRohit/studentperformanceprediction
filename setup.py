from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    """
    :param file_path:
    :return: list of requirements
    """

    with open (file_path) as file_obj:
        requirements = file_obj.read ().splitlines ()
        requirements = [req.replace ("\n", " ") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove (HYPHEN_E_DOT)

        return requirements


setup (
    name = "studentperformanceprediction",
    version = '0.0.1',
    author = 'Rohit',
    author_email = 'rohitnbele@gmail.com',
    packages = find_packages (),
    install_requires = get_requirements ('requirements.txt'),
)

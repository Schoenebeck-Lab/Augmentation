from setuptools import setup, find_packages

# conda env export | grep -v "^prefix: " > environment.yml

setup(
    name="aixchem", 
    version='0.2',
    author="Julian A. Hueffel",
    author_email="julian.hueffel@outlook.com",
    description='Analysis of computational chemistry datasets',
    long_description="Analysis of computational chemistry datasets",
    include_package_data=True,
    package_data={"aixchem": ["plots/style.mpl"]},
    packages=find_packages(),
    install_requires=[],
    keywords=['python'],
    )

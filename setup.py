import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='cfml_tools',  
     version='0.1dev0',
     author="Guilherme Marmerola",
     author_email="gdmarmerola@gmail.com",
     description="Counterfactual machine learning tools",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/gdmarmerola/cfml_tools",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: Apache License 2.0",
         "Operating System :: OS Independent",
     ],
 )
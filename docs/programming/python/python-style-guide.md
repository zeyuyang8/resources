---
title: Python Style Guide
layout: default
parent: Python
grand_parent: Programming
nav_order: 1
---

## Python Style Guide

**Note:** This page is adapted from [fundamentals of computing specialization](https://www.coursera.org/specializations/computer-fundamentals).

In our projects, adherence to our coding style guidelines is expected. These guidelines are based on the standards outlined in Python's [PEP 8](https://peps.python.org/pep-0008/). For open-source projects, these guidelines will be predominantly enforced through the utilization of [wemake-python-styleguide](https://github.com/wemake-services/wemake-python-styleguide), a tool designed for detecting source code bugs and ensuring code quality in the Python programming language. Additional information about common errors identified by the wemake-python-styleguide can be found [here](https://wemake-python-styleguide.readthedocs.io/en/latest/).

Here are some of the style guidelines that we will follow in our projects. As you interact with we-make-python-styleguide, you will get more exposure to these guidelines.

### Documenation

Documentation strings ("docstrings") are an integral part of the Python language. They need to be in the following places:

- At the top of each file describing the purpose of the program or module.
- Below each function definition describing the purpose of the function.

Docstrings describe what is being done in a program or function, **not** how it is being done. A docstring for a function should explain the arguments, what the function does, and what the function returns. Here is an example that includes a docstring at the start.

```python
"""An example program that illustrates the use of docstrings."""


def welcome(location: str):
    """Return a welcome message.
    
    Args:
        location: A string representing a location.
    
    Returns:
        A string containing a welcome message.
    """
    return 'Welcome to the' + location

```

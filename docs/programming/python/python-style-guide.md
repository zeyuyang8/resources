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

Docstrings describe **what** is being done in a program or function, **not** how it is being done. A docstring for a function should explain the arguments, what the function does, and what the function returns. Here is an example that includes a docstring at the start.

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

These docstrings are treated specially in Python, as they allow the system to automatically give you documentation for programs and functions. At the command prompt, you can type `help(...)`, and it will return the docstring for whatever the argument you passed to help is.

### Comments

Comments should describe how a section of code is accomplishing something. You should not comment obvious code. Instead, you should document complex code and/or design decisions. Comments and docstrings are not interchangeable. Comments start with the "#" character. While you will see some Python programmers do this, you should **not** comment your code by putting a multi-line string in the middle of your program. That is not actually a comment, rather it is just a string in the middle of your program!

A good example:

```python
# This is a block comment 
# that spans multiple lines 
# explaining the following code.
val = some_complicated_expression
```

A bad example:

```python
"""
Somebody told me that a multiline
string is a block comment.

It's not.
"""
val = some_complicated_expression
```

Note that docstrings are multi-line strings, but they do not violate this convention because docstrings and comments are different and serve different purposes in a program.

### Global Variables

Global variables should never be used in this course. Avoiding their use is good programming practice in any language. While programmers will sometimes break this rule, you should not break this rule in this course.

There is one exception to this rule: you may have global constants. Because the Python language does not actually support constants, by convention, Python programmers use global variables with names that are in all capital letters for constants. When you see a variable with a name in all capital letters, you should always assume that it is a constant and you should **never** change it. Again, such global constants are the only global variables that will be allowed in this specialization.

### Names

The first character of a name of a variable or a function should follow these conventions:

- Variable names should always start with a lower case letter. (Except for variables that represent constants, which should use all upper case letters.)
- Function names should always start with a lower case letter.

Further, we will follow the common Python convention that variable and function names should not have any capital letters in them. You can separate words in a name with an underscore character, as follows: `some_variable_name`. As previously noted, constants should be in all capital letters, such as: `THIS_IS_A_CONSTANT`.

### Indentation

Each indentation level should be indented by 4 spaces. As Python requires indentation to be consistent, it is important not to mix tabs and spaces. You should never use tabs for indentation. Instead, all indentation levels should be 4 spaces.  Additionally, lines should be limited to a length of some number of characters.

### Scope

You should not use names that knowingly duplicate other names in an outer scope. This would make the name in the outer scope impossible to access. In particular, you should never use names that are the same as existing Python built-in functions. For example, if you were to name one of your local variables `max` inside of a function, you would then not be able to call `max()` from within that function.

### Arguments and Local Variables

While there is not necessarily a maximum number of arguments a function can take or a maximum number of local variables you can have, too many arguments or variables lead to unnecessarily complex and unreadable programs. The style checker will enforce maximum numbers of arguments and variables, etc. If you run into limits that the style checker complains about, you should restructure your program to break it into smaller pieces. This will result in more readable and maintainable code.

Further, you should not have function arguments or local variables declared that are never used, except in rare circumstances. Sometimes, you do need to have a variable that you never use. A common case is in a loop that just needs to execute a certain number of times:

```python
for num in range(42):
    # do something 42 times
```

In this case, you should name the variable as `_`. This indicates clearly to you, others, and the style checker that the vrariable is intentionally unused.

```python
for _ in range(42):
    # do something 42 times
```

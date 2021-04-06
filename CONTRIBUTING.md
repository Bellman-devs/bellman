# Contribution guidelines


### Who are we?

*Administrators* (Vincent Adam, Felix Leibfried, John McLeod, Hrvoje Stojic) look after the GitHub repository itself.

*Maintainers* (Felix Leibfried, John McLeod, Hrvoje Stojic) steer the project, keep the community thriving, and manage contributions.

*Contributors* (you?) submit issues, make pull requests, answer questions on Slack, and more.

Community is important to us, and we want everyone to feel welcome and be able to contribute to their fullest. Our [code of conduct](CODE_OF_CONDUCT.md) gives an overview of what that means.


### Development tools

Instead of installing Bellman through `pip` you should install `poetry` and then install Bellman using `poetry` (follow the instructions in the [README file](README.md)). This will install all the development tools you will need to contribute - tools for testing, type checking, building documentation etc.


### Reporting a bug

Finding and fixing bugs helps us provide robust functionality to all users. You can either submit a bug report or, if you know how to fix the bug yourself, you can submit a bug fix. We gladly welcome either, but a fix is likely to be released sooner, simply because others may not have time to quickly implement a fix themselves. If you're interested in implementing it, but would like help in doing so, leave a comment on the issue, create a discussion page or ask in the [community Slack workspace](https://bellmangroup.slack.com).

We use GitHub issues for bug reports. You can use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) to start writing yours. Once you've submitted it, the maintainers will take a look as soon as possible and get back to you about how to proceed. If it's a small easy fix, they may implement it then and there. For fixes that are more involved, they will discuss with you about how urgent the fix is, with the aim of providing some timeline of when you can expect to see it.

If you'd like to submit a bug fix, the [pull request templates](.github/PULL_REQUEST_TEMPLATE.md) are a good place to start. We recommend you discuss your changes with the community before you begin working on them, so that questions and suggestions can be made early on.


### Requesting a feature

Bellman is built on features added and improved by the community. You can submit a feature request either as an issue or, if you can implement the change yourself, as a pull request. We gladly welcome either, but a pull request is likely to be released sooner, simply because others may not have time to quickly implement it themselves. If you're interested in implementing it, but would like help in doing so, leave a comment on the issue, create a discussion page or ask in the [community Slack workspace](https://bellmangroup.slack.com).

We use GitHub issues for feature requests. You can use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) to start writing yours. Once you've submitted it, the maintainers will take a look as soon as possible and get back to you about how to proceed. If it's a small easy feature that is backwards compatible, they may implement it then and there. For features that are more involved, they will discuss with you about a timeline for implementing it. It may become apparent during discussions that a feature doesn't lie within the scope of Bellman, in which case we will discuss alternative options with you, such as adding it as a notebook or an external extension to Bellman.

If you'd like to submit a pull request, the [pull request templates](.github/PULL_REQUEST_TEMPLATE.md) are a good place to start. We recommend you discuss your changes with the community before you begin working on them, so that questions and suggestions can be made early on.


### Pull request guidelines

- Limit the pull request to the smallest useful feature or enhancement, or the smallest change required to fix a bug. This makes it easier for reviewers to understand why each change was made, and makes reviews quicker.
- Where appropriate, include [documentation](#documentation), [type hints](#type-checking), and [tests](#tests). See those sections for more details.
- Pull requests that modify or extend the code should include appropriate tests, or be covered by already existing tests. In particular:
  - New features should include a demonstration of how to use the new API, and should include sufficient tests to give confidence that the feature works as expected.
  - Bug fixes should include tests to verify that the updated code works as expected and defend against future regressions.
  - When refactoring code, verify that existing tests are adequate.
- So that notebook users have the option to import things as
  ```python
  import bellman
  bellman.harness.utils.get_metric_values(...)
  ```
  do not import anything in the \_\_init\_\_.py file.
- In commit messages, be descriptive but to the point. Comments such as "further fixes" obscure the more useful information.


### Documentation

Bellman has two primary sources of documentation: the notebooks and the API reference.

For the API reference, we document Python code inline, using [reST markup](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html). See [here](docs/README.md) for details on the documentation build. All parts of the public API need docstrings (indeed anything without docstrings won't appear in the built documentation). Similarly, don't add docstrings to private functionality, else it will appear in the documentation website. Use code comments sparingly, as they incur a maintenance cost and tend to drift out of sync with the corresponding code.


#### Requesting a documentation change

We welcome contributions for improving the documentation. You can submit a documentation change request either as an issue or, if you can implement the change yourself, as a pull request. We gladly welcome either, but a pull request is likely to be released sooner, simply because others may not have time to quickly implement it themselves. If you're interested in implementing it, but would like help in doing so, leave a comment on the issue, create a discussion page or ask in the [community Slack workspace](https://bellmangroup.slack.com).

We use GitHub issues for documentation change requests. You can use the [documentation issue template](.github/ISSUE_TEMPLATE/doc_issue.md) to start writing yours. Once you've submitted it, the maintainers will take a look as soon as possible and get back to you about how to proceed. If it's a small easy change, they may implement it then and there. For more substantial change, they will discuss with you about a timeline for implementing it. 

If you'd like to submit a pull request, the [pull request templates](.github/PULL_REQUEST_TEMPLATE.md) are a good place to start. We recommend you discuss your changes with the community before you begin working on them, so that questions and suggestions can be made early on. Take a look at how you can build documentation and verify all is good in [documentation checking](#documentation-checking).


### Quality checks

We use `tasks` in `poetry` to run reproducible quality checks. These tasks are defined in `pyproject.toml` file.

To run all the quality checks
```bash
$ poetry run task test
```

Next you can find a bit more details about each quality check and how to run them separately.


#### Linting

We use [pylint](https://www.pylint.org/) for style checking according to Python's PEP8 style guide. We do this throughout the source code and tests. You may need to run these before pushing changes, with (in the repository root)
```bash
$ poetry run task lint
```


#### Type checking

We use [type hints](https://docs.python.org/3/library/typing.html) for documentation and static type checking with [mypy](http://mypy-lang.org). We do this throughout the source code and tests. The notebooks are checked for type correctness, but we only add types there if they are required for mypy to pass. This is because we anticipate most readers of notebooks won't be using type hints themselves. If you don't know how to add type hints to your code, leave them out. You can use `typing.Any` where the actual type isn't expressible or practical, but do avoid it where possible.

Run the type checker with
```bash
$ poetry run task mypy
```


#### Tests

We write and run tests with [pytest](https://pytest.org). We aim for all public-facing functionality to have tests for both happy and unhappy paths (that test it works as intended when used as intended, and fails as intended otherwise). We don't test private functionality, as the cost to ease of development is more problematic than the benefit of improved robustness.

Run (only the) tests with
```bash
$ poetry run task quicktest
```


#### Code formatting

We format all Python code, other than the notebooks, with [black](https://black.readthedocs.io/en/stable/) and [isort](https://pycqa.github.io/isort/). You may need to run these before pushing changes, with (in the repository root)
```bash
$ poetry run task black
$ poetry run task isort
```

Or, to run both together, you can use
```bash
$ poetry run task format
```


#### Documentation checking

If you are making changes to the documentation or if you are introducing new functions, please verify whether documentation can be built successfully and with sufficient quality. There are several tasks defined in `pyproject.toml` that help with verifying how the updated documentation looks like. 

To build the documentation run
```bash
$ pip install -r ./docs/docs_requirements.txt
$ poetry run task docsgen
```

After a successful build there are two functions that you can run, with `docserve`
```bash
$ poetry run task docserve
```
you start a basic server and then with `docsview` you can open the documentation, how it will finally look like.
```bash
$ poetry run task docsview
```


#### Continuous integration

[GitHub Actions](https://docs.github.com/en/actions) will automatically run the quality checks against pull requests to the develop branch, by calling into `poetry`. The GitHub repository is set up such that these need to pass in order to merge.


### Updating dependencies

To update the Python dependencies used in any part of the project, use `poetry` to update the existing ones or adding new ones. You will then need to update [setup.py file](setup.py) and/or any relevant `requirements.txt` files. To do that, in the repository root, and with all virtual environments deactivated, run a poetry task that checks dependencies defined by poetry and whether they match the ones in `requirements.txt`.
```bash
$ poetry run task check_requirements
```


# License

[Apache License 2.0](LICENSE)

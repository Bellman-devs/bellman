# Bellman

<!-- [![Build & Test](https://github.com/Bellman-devs/bellman/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/Bellman-devs/bellman/actions/workflows/build-and-test.yml)
[![Docs](https://github.com/Bellman-devs/bellman/actions/workflows/publish-docs.yml/badge.svg)](https://github.com/Bellman-devs/bellman/actions/workflows/publish-docs.yml)
[![Coverage Status](http://codecov.io/github/Bellman-devs/bellman/coverage.svg?branch=master)](http://codecov.io/github/Bellman-devs/bellman?branch=master)
[![Documentation Status](https://readthedocs.org/projects/bellman/badge/?version=master)](http://bellman.readthedocs.io/en/master/?badge=master)
 -->

<!-- [Website](https://bellman.dev) |
[Documentation (release)](https://bellman.readthedocs.io/en/master/) |
[Documentation (develop)](https://bellman.readthedocs.io/en/develop/) |
 -->
 
Bellman is a package for model-based reinforcement learning (MBRL) in Python, using [TensorFlow](http://www.tensorflow.org) and building on top of model-free reinforcement learning package [TensorFlow Agents](https://www.tensorflow.org/agents/overview?hl=en&authuser=0).
It was originally created by (in alphabetical order) 
[Vincent Adam](https://vincentadam87.github.io/), 
[Jordi Grau-Moya](https://sites.google.com/view/graumoya), 
[Felix Leibfried](https://github.com/fleibfried), 
[John McLeod](https://github.com/johnamcleod), 
[Hrvoje Stojic](https://hstojic.re), and 
[Peter Vrancx](https://github.com/pvrancx), 
at [Secondmind Labs](https://www.secondmind.ai/labs/). 
It is now actively maintained by (in alphabetical order)
[Felix Leibfried](https://github.com/fleibfried),
[John McLeod](https://github.com/johnamcleod),
and [Hrvoje Stojic](https://hstojic.re).

Bellman is an open source project. If you have relevant skills and are interested in contributing then please do contact us (see ["The Bellman Community" section](#the-bellman-community) below).


## What does Bellman do?

Bellman provides a framework for flexible composition of model-based reinforcement learning algorithms. It offers two major classes of algorithms: decision time planning and background planning algorithms. With each class any kind of supervised learning method can be easily used to learn certain component of the environment. Bellman was designed with modularity in mind - important components can be flexibly combined, such as type of decision time planning method (e.g. a cross entropy method or a random shooting method) and type of model for state transition (e.g. a probabilistic neural network or an ensemble of neural networks). Bellman also provides implementation of several popular state-of-the-art MBRL algorithms, such as PETS, MBPO and METRPO. The [online documentation (develop)](http://bellman.readthedocs.io/en/develop/)/[(master)](http://bellman.readthedocs.io/en/master/) contains more details. 

Bellman uses [TensorFlow 2.4+](http://www.tensorflow.org) for running computations, which allows fast execution on GPUs. Bellman requires Python 3.7 onwards.


## Install Bellman

#### For users

You can use `pip` to install the toolbox. For the latest release from PyPi simply run
```bash
$ pip install bellman
```
or to install from latest source from GitHub, check-out the `develop` branch of the [Bellman GitHub repository](https://github.com/Bellman-devs/bellman), and run (in editable mode)
```bash
$ pip install -e .
```
in the repository root.


#### For contributors

If you wish to contribute please use [Poetry](https://python-poetry.org/docs) to manage dependencies in a local virtual environment. Poetry [configuration file](pyproject.toml) specifies all the development dependencies (testing, linting, typing, docs etc) and makes it much easier to contribute. To install Poetry, [follow the instructions in the Poetry documentation](https://python-poetry.org/docs/#installation). 

To install this project in editable mode, run the commands below from the root directory of the `bellman` repository.

```bash
poetry install
```

This command creates a virtual environment for this project
in a hidden `.venv` directory under the root directory. You can easily activate it with

```bash
poetry shell
```

You must also run the `poetry install` command to install updated dependencies when
the `pyproject.toml` file is updated, for example after a `git pull`.


#### Installing MuJoCo (Optional)

Many benchmarks in continuous control in MBRL use the MuJoCo physics engine. Some of the TF-Agents examples have been tested against Mujoco environments as well. MuJoCo is proprietary software that requires a license [(see MuJoCo website)](https://www.roboti.us/license.html). As a result installing it is optional, but because of its importance to the research community it is highly recommended. Don't worry if you decide not to install MuJoCo though, all our examples and notebooks rely on standard environments available in OpenAI Gym. 

We interface with MuJoCo through a python library `mujoco-py` via OpenAI Gym [(mujoco-py github page)](https://github.com/openai/mujoco-py). Check the installation instructions there on how to install MuJoCo. Note that you should install MuJoCo 1.5 since OpenAI Gym supports that version. After that you can install mujoco-py library with an additional Poetry command:

```bash
poetry install -E mujoco-py
```

If this command fails, please check troubleshooting sections at [`mujoco-py` github page](https://github.com/openai/mujoco-py), you might need to satisfy other `mujoco-py` dependencies (e.g. Linux system libraries) or set some environment variables.


## The Bellman Community

### Getting help

**Bugs, feature requests, pain points, annoying design quirks, etc:**
Please use [GitHub issues](https://github.com/Bellman-devs/bellman/issues/) to flag up bugs/issues/pain points, suggest new features, and discuss anything else related to the use of Bellman that in some sense involves changing the Bellman code itself. We positively welcome comments or concerns about usability, and suggestions for changes at any level of design. We aim to respond to issues promptly, but if you believe we may have forgotten about an issue, please feel free to add another comment to remind us.

**"How-to-use" questions:**
Please use [Stack Overflow (Bellman tag)](https://stackoverflow.com/tags/Bellman) to ask questions that relate to "how to use Bellman", i.e. questions of understanding rather than issues that require changing Bellman code. (If you are unsure where to ask, you are always welcome to open a GitHub issue; we may then ask you to move your question to Stack Overflow.)


### Contributing

All constructive input is very much welcome. For detailed information, see the [guidelines for contributors](CONTRIBUTING.md).

<!-- 
## Citing Bellman

To cite Bellman, please reference our [arXiv paper](https://arxiv.org/abs/XXXX) where we review the framework and describe the design. Sample Bibtex is given below:

```
@article{bellman2021,
  author = {McLeod, John and Stojic, Hrvoje and Adam, Vincent and Kim, Dongho and Grau-Moya, Jordi and Vrancx, Peter and Leibfried, Felix},
  title = {Bellman: A Framework for Model-based Reinforcement Learning},
  year = {2021},
  journal = {arXiv:XXXX},
  url = {https://arxiv.org/abs/XXXX}
}
```
 -->

## License

[Apache License 2.0](LICENSE)

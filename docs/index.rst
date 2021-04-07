Bellman documentation
=====================

Bellman is a Python toolbox for model-based reinforcement learning (MBRL). That is, for solving sequential
decision problems by learning predictive models of parts of the problem.

The toolbox extends the open source TensorFlow-Agents library, initially conceived to tackle model-free RL problems, 
to enable model-based RL capabilities.


Getting started
---------------

Start with the Approximating MDPs Jupyter Notebook for an overview of the predictive models.

Alternatively, read the API reference documentation.


.. toctree::
   :caption: Jupyter Notebooks
   :titlesonly:
   :hidden:
   :maxdepth: 1

   notebooks/approximate_mdps
   notebooks/model_visualisation
   notebooks/trajectory_optimisation

.. autosummary::
   :toctree: _autosummary
   :caption: API Reference
   :template: custom-module-template.rst
   :recursive:

   bellman

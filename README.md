# cfml_tools: Counterfactual Machine Learning Tools
 
For a long time, ML practitioners and Statisticians repeated the same mantra: *Correlation is not causation*. This warning prevented (at least, some) people from drawing wrong conclusions from models but also created a misconception that ML **cannot** be causal. With a few tweaks drawn from the causal inference literature ([DeepIV](http://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf), [Generalized Random Forests](https://arxiv.org/pdf/1610.01271.pdf), [Causal Trees](https://arxiv.org/abs/1504.01132)) and Reinforcement Learning literature ([Bandits](https://arxiv.org/abs/1711.07077), [Thompson Sampling](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf)) we actually **can** male Machine Learning methods aware of causality!

**`cfml_tools`** is a collection of causal inference algorithms built on top of accessible, simple, out-of-the-box ML methods.

# Installation

Open up your terminal and perform:

`git clone https://github.com/gdmarmerola/cfml_tools.git`
`cd cfml_tools`
`python setup.py install`

# Usage

The package uses a scikit-learn like API, and is fairly easy to use. Check the [examples](https://github.com/gdmarmerola/cfml_tools/tree/master/examples) for applications on toy causal inference datasets!


# aprofs Package - Aproximate Predictions for Feature Selection

This site contains the project documentation for the
`aprofs` project.

This is a side project that allows to implement some of the ideas that came from the
investigation about how can we use shapley values to get approximate predictions.

In here we take account og the additive nature of shapley values, and disregard a bit the potential
problems with correlations/interaction that happened.

You can install it from github like this:
```bash
pip install git+https://github.com/blewy/aprofs
```

or from, **pypy production** like this:

```bash
pip install aprofs==0.0.5
```

The idea of the package is to help you select your features, getting a simpler model, then understand your features
using shapley values and the concept of approximate prediction.

Always trying to use the **marginal effect** of the calculated shapley values.

## Table Of Contents

The documentation consists of four separate parts:

1. [Tutorials](Tutorial.ipynb) Basic Tutorial on how to use the package with some example data.
2. [How-To Guides](guide.md) Explain the intended utilization of the package functionality
3. [Reference](reference.md) Reference code, math and explanations of the built in functionality
4. [API](api.md) Code documentation

Looking into what helps you, and if needed give us some feedback on git repository.


## Acknowledgements

I want to thank my colleagues in insurance that trough the years have challenge me to think deeply about these topics.

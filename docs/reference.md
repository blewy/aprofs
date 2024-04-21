# References for this project
The initial idea for this project came from the medium post [**“Approximate-Predictions” Make Feature Selection Radically Faster**](https://medium.com/towards-data-science/approximate-predictions-make-feature-selection-radically-faster-0f9664877687) from [from https://medium.com/@mazzanti.sam.](https://medium.com/@mazzanti.sam).

This lead me into new ideas of potential usage (even abuse) of shapley values for machine learning models.

In this project I tried to implement the ideas left by **Samuelle** in the above post, and tried to make it easy an accessible to be used on a MLproject. Also, I tried to increase my DS skills creating a python package and also in some way to give-back to the open-source community what I have been receiving from them over the years.

The main premise about the user of this package is that you know how to create you ML models and do some cleaning of the data previously.
Please don't throw garbage into any automatic process and expect that roses will come at the end of it.

The second big premise is that you are acquainted with shapley values and/or getting a shapley values table from some calibration data and respective predictive model.

I will add more resources (if I can) here to help you out:

### Shapley values

- SHAP package used for calculate shapley values [shap package](https://shap.readthedocs.io/en/latest/)

- Details about the Shapley Tree Explainer [shap tree explainer](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Census%20income%20classification%20with%20LightGBM.html)


- Shapley value reference book [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/shapley.html) from Christoph Molnar (https://christophmolnar.com)

- This interesting book about Shapley values [Interpreting Machine Learning Models With SHAP](https://christophmolnar.com/books/shap/) also from Christoph Molnar

### Shap Visualizations

- [stackoverflow](https://stackoverflow.com/questions/65024195/add-regression-line-to-a-shap-dependence-plot)
- [shap_dependence_plot](https://github.com/shap/shap/blob/master/shap/plots/_partial_dependence.py)
- [shap_pdp](https://www.kaggle.com/code/dansbecker/advanced-uses-of-shap-values)


### P-values

About creating **p-values**, please take a look at this chapter from the book [Feature Engineering and Selection: A Practical Approach for Predictive Models](http://www.feat.engineering/greedy-simple-filters) from  Max Kuhn and Kjell Johnson

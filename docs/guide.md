This part of the project documentation focuses on explaining
how the code should be used.

## Get you first aprofs bject up

The initial idea was to create a struture that contains all the irformation needed to use the shapley values
alongside with you calibration data to get the results you need.

so opne instaciats with all the proper inputs we just need to use the correct methos to get our results.

Its very important to get the shapley calues calulated before using the Aprofs object. Ive added a wrapper methos calleff **calculate_shaps** where you brin you own model. This jts wrped teh pythpm code got add the shapley value tabel and the shap measn value into th APofs object.

``` py  title="TreeExplainer"
    shap_explainer = TreeExplainer(model)
    shap_valid = shap_explainer.shap_values(data)
    shap_expected_value = shap_explainer.expected_value
```

Pelase reach to the APi to get all the deatils of how to crete but basically you need the **calibration** data, the **target column** for that data and a ML **model**. The detail was that the model will not be saved inside the Aprofs object. I didn't wanted to bloat the thing more than needed.

```py
from aprofs import code

aprofs_obj = code.Aprofs(X_calibration, y_calibration, link="logistic")
```

[!WARNING]  
**At the moment only the **logistic** or **binary** models where tested and develops, more to come in the future.**

and then you can add the shapley value table and measn shale values with a precalulated like this

``` py
aprofs_obj.shap_values = pre_calc_shaps_df
aprofs_obj.shap_mean = pre_calc_shaps_mean
```

the detail here is that the **aprofs_obj** values need to be in a data frame shape, tho to this just use this snipet of code.
for example
``` py  title="Shap to dataframe"
pre_calc_shaps_df= pd.DataFrame(shap_values, index=self.X_calibration.index, columns=self.X_calibration.columns)
```

or just use the wrapper that is provided:

``` py  title="Shap to dataframe"
aprofs_obj.calculate_shaps(model)
```

## Select features with aprofs object


After having your shapley value created or added into the aprofs object we can start using the built in funcionality.

The firs stop wil be using the feature selection functionality. At the mometno we just have the **brute force** created.
More will be added in the future.

Th eidead is that this mehtod will look into all possibilities of the feature combinations ant claculate the model performance
using the aproximates prediction possible with the shapleo+y value table.

In tis case the perfoemce of the **feature A** and **feature B**. will be using a prediction the following:

    - Shapley A + Shapley Value B + Shap mean value

To this score then its applied the inverse of the link function the get the final prediction.
In the case of binary classification model its applied the sigmoid function into the model score.

``` py  title="Feature selection"
aprofs_obj.brute_force_selection(["list of features"])
```

It will return a tuple with the best subset of features.


## Visualize your shapley value

To valudate our model its interesting to look into the predicted vs observed values comparing their behavior
along the feature used for modelling, bua also any other what you might seam interesting.

An interesting add on can be also to add the marginal effect of the shappley values for that features,
this was you can see the overall behavior of the models, but also the marginal effect of the feature on your model
building even more intuition and understanding into your package.

``` py  title="Feature Visualization"
aprofs_obj.visualize_feature(
        main_feature: "Feature A",
        other_features: ["Feature B"],
        nbins = = 20,
        type_bin = "qcut",
        type_plot = "prob")
```

## Package structure

Download the code from this GitHub repository and place
the `aprofs/` folder in the same directory as your
Python script:

    aprofs/
    │
    ├── aprofs/
        ├── __init__.py
        ├── code.py
        |__ utils.json

# This part of the project documentation focuses on explaining how the code should be used.

## Get you first aprofs object up

The initial idea of the project was to create a structure that contains all the information needed to use the **shapley values**
alongside with you calibration data to get the results you need.

Once instantiated this object with all the proper inputs then we just need to use the correct methods to get our results.

Its very important to get the shapley values calculated before using the Aprofs object. I've added some wrapper method called **calculate_shaps** that will help you you bring your own model.

To get the shapley table I just used some **SHAP package** functionality like you see below

``` py  title="TreeExplainer"
    shap_explainer = TreeExplainer(model) # Shap explainer for tree models
    shap_valid = shap_explainer.shap_values(data) # get you sdapley values for your data
    shap_expected_value = shap_explainer.expected_value # averaga shapley value
```

Please reach to the [API](api.md) to get all the details of how to create the object and use the method. Basically you need the **calibration** data, the **target column** for that data and an ML **model** that can be score on the calibration data.
The structural detail is that the model will not be saved inside the Aprofs object. I didn't want to bloat the object more than needed.

```py
from aprofs import code

aprofs_obj = code.Aprofs(X_calibration, y_target_calibration, link="logistic")
```

[!WARNING]  
**At the moment only the **logistic** or **binary** models where tested and develops, more to come in the future.**

You can add the shapley value table and mean shapley values with pre-calculated results like this:

``` py
aprofs_obj.shap_values = pre_calc_shaps_df
aprofs_obj.shap_mean = pre_calc_shaps_mean
```

A **note** here, the **aprofs_obj shap values** values need to be as a (pd.) dataframe. To do this use this snippet of code for example:

``` py  title="Shap to dataframe"
pre_calc_shaps_df= pd.DataFrame(shap_values, index=self.X_calibration.index, columns=self.X_calibration.columns)
```

If you use the wrapper that is provided, this will work:

``` py  title="Shap to dataframe"
aprofs_obj.calculate_shaps(model)
```

## Select features with aprofs object

After having your shapley value created or added into the aprofs object we can start using the built-in functionality.

The firs step will be using the feature selection functionality. At the moment I have implemented the **brute force** method and the greedy forward selection method. More will be added in the future.

The idea of **brute force** is that this method will look into all possibilities of the feature combinations and calculate the model performance using the approximate prediction possible with the shapley values table.

On the other approach, greedy forward.
    0 - Initialize the **winning solution** with the best individual feature.
    1 - Select best feature individually not in the **winning solution**.
    2 - Add this feature into the best solution and test if it improves the performance.
    3 - If **yes** in the previous step: keep in the wining solution and start from **1**, if **no** we drop the feature from the wining solution and back to **1**

As an example, the performance of the **feature A** and **feature B**. will be calculated using the following logic:

    - (Shapley A + Shapley Value B + Shap mean value)

To this score then its applied the inverse of the link function the get the final prediction.
In the case of binary classification function applied the sigmoid function into the model score.

To run **feature selection** use the following method.

``` py  title="Feature selection"
aprofs_obj.brute_force_selection(["list of features"])
```

It will return a list with the best subset of features.


For the greedy gready_forward_selection use:

``` py  title="Feature selection"
aprofs_obj.brute_force_selection(["list of features"])
```

## Visualize your shapley value

To validate and understand our model its interesting to look into the predicted vs observed values comparing their behavior
along the features used for modelling, but also look at any other feature could look interesting.

An interesting add on can be also to add the marginal effect of the shapley values for that feature,
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

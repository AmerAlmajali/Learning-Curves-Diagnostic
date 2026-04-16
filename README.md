# Learning-Curves-Diagnostic
## Bias–Variance Tradeoff Analysis via Learning Curves in an Imbalanced Dataset:

The learning curves provide a clear diagnostic of the model’s behavior
and indicate that the logistic regression model is primarily affected 
by **high bias (underfitting)** rather than high variance. 
This conclusion is based on two key observations. First, both the training
and validation F1 scores converge to relatively low values (approximately 0.54–0.57),
indicating that the model is unable to achieve high performance even on the training
data. Second, the gap between the training and validation curves is consistently
small across all training set sizes, which suggests that the model is not overfitting
and is generalizing similarly on both seen and unseen data.

At smaller training sizes, the training F1 score is initially higher, which is expected
due to mild overfitting when the model has access to limited data. However, as the training 
set size increases, the training score decreases and stabilizes, while the validation score 
increases slightly and then plateaus. This convergence behavior reflects the model 
transitioning from memorization to generalization. The key observation is that both 
curves stabilize at a relatively low performance level, reinforcing the conclusion that 
the model lacks sufficient capacity to capture the underlying patterns in the data.

Another important aspect of the learning curves is the behavior of the validation score as 
more data is added. While there is a slight improvement in validation performance at the 
beginning, the curve quickly flattens, indicating diminishing returns from additional data. 
This plateau suggests that the model has already learned as much as it can from the available
feature representation, and further increasing the dataset size is unlikely to result in 
significant performance gains. In other words, the limitation is not due to insufficient 
data, but rather due to the simplicity of the model.

The comparison between different regularization strengths (C values) further supports this 
interpretation. Models with stronger regularization (e.g., C=0.01) exhibit even lower 
training and validation scores, which is characteristic of increased bias. On the other 
hand, reducing regularization (e.g., C=10) slightly increases training performance but does 
not significantly improve validation performance, indicating that simply relaxing 
regularization is not sufficient to overcome the model’s limitations. 
The consistent convergence across all configurations highlights that the issue is structural 
rather than parametric.

Additionally, when compared to the dummy baselines, the logistic regression models outperform 
both the “most frequent” and “stratified” classifiers, confirming that the model is learning 
meaningful patterns beyond random or trivial predictions. However, the performance margin is 
relatively modest, suggesting that while the model captures some signal, it fails to fully 
exploit the available information in the dataset.

Given these observations, increasing model complexity is likely to be beneficial. 
Logistic regression is inherently a linear model, which restricts it to learning 
linear decision boundaries. If the relationship between features and the target 
variable is non-linear, the model will systematically underfit. One potential 
improvement is to introduce polynomial features, which allow the model to capture non-linear 
interactions between variables. Alternatively, more flexible models such as decision trees, 
random forests, or gradient boosting methods can be used, as they are better suited for 
modeling complex, non-linear relationships.

In conclusion, the learning curves strongly indicate that the model is limited by 
high bias. Collecting more data is unlikely to significantly improve performance, 
as the validation curve has already plateaued. The most effective next step is to 
increase model capacity through feature engineering or by adopting more expressive models, 
which can better capture the underlying structure of the telecom churn dataset and improve 
predictive performance.

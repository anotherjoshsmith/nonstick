import os.path as op
import pandas as pd

from sklearn.datasets import make_classification

N = 5000
file_name = "example_data.csv"
# Generate data set
X1, true_class = make_classification(
    n_samples=N,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    n_classes=3,
)

# move data to DataFrame for nice output
X_out = pd.DataFrame(X1)
X_out.columns = ["x1", "x2"]
X_out["class"] = true_class

# save example data
X_out.to_csv(op.join(op.dirname(__file__), file_name))

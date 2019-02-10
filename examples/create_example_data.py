import os.path as op
import pandas as pd

from sklearn.datasets import make_classification

N = 50000
file_name = "example_data_3d.csv"
# Generate data set
X1, true_class = make_classification(
    n_samples=N,
    n_features=3,
    n_redundant=0,
    n_informative=3,
    n_clusters_per_class=1,
    n_classes=3,
)

# move data to DataFrame for nice output
X_out = pd.DataFrame(X1)
X_out.columns = ["x1", "x2", "x3"]
X_out["class"] = true_class

# save example data
X_out.to_csv(op.join(op.dirname(__file__), file_name))

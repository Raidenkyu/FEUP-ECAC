import sklearn.metrics as metrics
import matplotlib.pyplot as plt


def plot_auc(model, X_test, y_test, algorithm):
    metrics.plot_roc_curve(model, X_test, y_test)
    plt.savefig(f"{algorithm}.png")

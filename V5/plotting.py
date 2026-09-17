import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(losses, title="Training Loss"):
    """
    Loss per epoch
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(losses)), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss (-ELBO)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_predictions_vs_actual(preds, actuals, title="Predicted vs Actual"):
    """
    Scatter of predicted vs actual credit_spread.
    """
    plt.figure(figsize=(7, 7))
    plt.scatter(actuals, preds, alpha=0.1, s=5)

    lims = [min(actuals.min(), preds.min()), max(actuals.max(), preds.max())]
    plt.plot(lims, lims, 'r--', label='Perfect prediction (y=x)')

    plt.xlabel("Actual credit_spread")
    plt.ylabel("Predicted credit_spread")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(preds, actuals, title="Residuals"):
    """
    Prediction error (actual - predicted) plotted against actual values.
    """
    residuals = actuals - preds

    plt.figure(figsize=(8, 5))
    plt.scatter(actuals, residuals, alpha=0.1, s=5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Actual credit_spread")
    plt.ylabel("Residual (actual - predicted)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    print(f"Residual mean: {residuals.mean():.5f} (should be close to 0)")
    print(f"Residual std: {residuals.std():.5f}")

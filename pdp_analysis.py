"""
XAI - Exercise 5: Partial Dependence Plots (PDP)
=================================================
Exercise 1: 1D PDP - Bike rentals (day.csv)
Exercise 2: 2D PDP - Humidity x Temperature (day.csv)
Exercise 3: 1D PDP - House prices (kc_house_data.csv)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
import warnings
warnings.filterwarnings("ignore")

# ── Color palette ──────────────────────────────────────────────────
BLUE   = "#2166AC"
RED    = "#D6604D"
GREEN  = "#4DAC26"
ORANGE = "#F4A582"
GRAY   = "#AAAAAA"
BG     = "#F7F7F7"

# ══════════════════════════════════════════════════════════════════
# EXERCISE 1 – 1D PDP for Bike Rentals
# ══════════════════════════════════════════════════════════════════
print("Loading bike dataset...")
day = pd.read_csv("/mnt/user-data/uploads/day.csv")

# 'instant' is days since 2011 (day index)
features_bike = ["instant", "temp", "hum", "windspeed",
                 "season", "yr", "mnth", "holiday", "weekday",
                 "workingday", "weathersit", "atemp"]

X_bike = day[features_bike]
y_bike = day["cnt"]

print("Training Random Forest (bike)...")
rf_bike = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_bike.fit(X_bike, y_bike)
print(f"  R² = {rf_bike.score(X_bike, y_bike):.4f}")

# Features to plot for Exercise 1
plot_features = {
    "instant":   "Days since 2011",
    "temp":      "Temperature (normalised)",
    "hum":       "Humidity (normalised)",
    "windspeed": "Wind Speed (normalised)",
}

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.patch.set_facecolor(BG)
fig.suptitle("Exercise 1 – 1D Partial Dependence Plots\nBike Rental Prediction",
             fontsize=15, fontweight="bold", y=1.01)

for ax, (feat, label) in zip(axes.flat, plot_features.items()):
    feat_idx = features_bike.index(feat)
    pd_result = partial_dependence(rf_bike, X_bike, [feat_idx],
                                   kind="average", grid_resolution=100)
    grid_vals = pd_result["grid_values"][0]
    avg_pred  = pd_result["average"][0]

    ax.set_facecolor(BG)
    ax.plot(grid_vals, avg_pred, color=BLUE, linewidth=2.5)
    ax.fill_between(grid_vals, avg_pred,
                    alpha=0.15, color=BLUE)

    # Rug plot (data distribution)
    raw = X_bike[feat].values
    ax.plot(raw, np.full_like(raw, avg_pred.min() - 60),
            "|", color=GRAY, alpha=0.4, markersize=5)

    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Predicted bike count", fontsize=11)
    ax.set_title(f"PDP – {label}", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/exercise1_1D_PDP_bikes.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved exercise1_1D_PDP_bikes.png")


# ══════════════════════════════════════════════════════════════════
# EXERCISE 2 – 2D PDP: Humidity × Temperature
# ══════════════════════════════════════════════════════════════════
print("Generating 2D PDP (bike)...")

# Random sample to keep computation fast
sample = day.sample(300, random_state=42)
X_samp = sample[features_bike]

# Grid for hum × temp
hum_vals  = np.linspace(day["hum"].min(),  day["hum"].max(),  50)
temp_vals = np.linspace(day["temp"].min(), day["temp"].max(), 50)

hum_grid, temp_grid = np.meshgrid(hum_vals, temp_vals)
grid_points = pd.DataFrame({
    feat: X_samp[feat].mean() for feat in features_bike
}, index=range(len(hum_grid.ravel())))

grid_points["hum"]  = hum_grid.ravel()
grid_points["temp"] = temp_grid.ravel()

preds_2d = rf_bike.predict(grid_points).reshape(hum_grid.shape)

fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor(BG)
fig.suptitle("Exercise 2 – 2D Partial Dependence Plot\nHumidity × Temperature → Bike Rentals",
             fontsize=14, fontweight="bold")

gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                       hspace=0.05, wspace=0.05)

# Main 2D tile plot
ax_main = fig.add_subplot(gs[1, 0])
ax_main.set_facecolor(BG)
im = ax_main.contourf(hum_grid, temp_grid, preds_2d,
                      levels=30, cmap="RdYlBu_r")
ax_main.set_xlabel("Humidity (normalised)", fontsize=11)
ax_main.set_ylabel("Temperature (normalised)", fontsize=11)
ax_main.grid(True, linestyle="--", alpha=0.3, color="white")

# Colorbar
cbar = fig.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
cbar.set_label("Predicted bike count", fontsize=10)

# Top density (humidity)
ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
ax_top.set_facecolor(BG)
ax_top.hist(day["hum"], bins=30, color=BLUE, alpha=0.7, density=True)
ax_top.set_ylabel("Density", fontsize=9)
ax_top.set_title("Humidity distribution", fontsize=9)
plt.setp(ax_top.get_xticklabels(), visible=False)
ax_top.spines[["top", "right"]].set_visible(False)

# Right density (temperature)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
ax_right.set_facecolor(BG)
ax_right.hist(day["temp"], bins=30, color=RED, alpha=0.7,
              density=True, orientation="horizontal")
ax_right.set_xlabel("Density", fontsize=9)
ax_right.set_title("Temp\ndistrib.", fontsize=9)
plt.setp(ax_right.get_yticklabels(), visible=False)
ax_right.spines[["top", "right"]].set_visible(False)

plt.savefig("/mnt/user-data/outputs/exercise2_2D_PDP_bikes.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved exercise2_2D_PDP_bikes.png")


# ══════════════════════════════════════════════════════════════════
# EXERCISE 3 – 1D PDP for House Price Prediction
# ══════════════════════════════════════════════════════════════════
print("Loading house dataset...")
kc = pd.read_csv("/mnt/user-data/uploads/kc_house_data.csv")

features_house = ["bedrooms", "bathrooms", "sqft_living",
                  "sqft_lot", "floors", "yr_built"]

X_house = kc[features_house]
y_house = kc["price"]

# Random sample for PDP computation (dataset is large)
sample_idx = kc.sample(3000, random_state=42).index
X_house_samp = X_house.loc[sample_idx]
y_house_samp = y_house.loc[sample_idx]

print("Training Random Forest (houses)...")
rf_house = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_house.fit(X_house_samp, y_house_samp)
print(f"  R² = {rf_house.score(X_house_samp, y_house_samp):.4f}")

plot_features_house = {
    "bedrooms":    "Bedrooms",
    "bathrooms":   "Bathrooms",
    "sqft_living": "Living Area (sqft)",
    "floors":      "Floors",
}

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.patch.set_facecolor(BG)
fig.suptitle("Exercise 3 – 1D Partial Dependence Plots\nHouse Price Prediction (King County)",
             fontsize=15, fontweight="bold", y=1.01)

for ax, (feat, label) in zip(axes.flat, plot_features_house.items()):
    feat_idx = features_house.index(feat)
    pd_result = partial_dependence(rf_house, X_house_samp, [feat_idx],
                                   kind="average", grid_resolution=80)
    grid_vals = pd_result["grid_values"][0]
    avg_pred  = pd_result["average"][0]

    ax.set_facecolor(BG)
    ax.plot(grid_vals, avg_pred / 1e3, color=GREEN, linewidth=2.5)
    ax.fill_between(grid_vals, avg_pred / 1e3,
                    alpha=0.15, color=GREEN)

    # Rug
    raw = X_house_samp[feat].values
    ax.plot(raw, np.full_like(raw, float((avg_pred / 1e3).min()) - 10),
            "|", color=GRAY, alpha=0.3, markersize=4)

    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Predicted price (k$)", fontsize=11)
    ax.set_title(f"PDP – {label}", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/exercise3_1D_PDP_houses.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved exercise3_1D_PDP_houses.png")

print("\nAll plots generated successfully.")

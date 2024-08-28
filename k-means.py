import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib.animation import FuncAnimation, PillowWriter

# Generate synthetic data with more spread-out clusters
n_samples = 300
n_features = 2
n_clusters = 3
random_state = 42

# Create synthetic data with spread-out clusters
X, _ = make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=n_clusters,
    cluster_std=2.5,
    random_state=random_state,
)

# Initialize centroids at the edges of the plot
x_min, x_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
y_min, y_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1
centroids = np.array([[x_min, y_min], [x_max, y_max], [x_min, y_max]])

# Define step size to control the movement speed of centroids
step_size = 0.1  # Smaller value for slower, more gradual movement

# Define colors for clusters and centroids
colors = ["#FF5733", "#33FF57", "#3357FF"]
centroid_colors = ["#780010", "#165204", "#160078"]

# Initialize plot with background color
fig, ax = plt.subplots()
ax.set_facecolor("#f0f0f0")  # Light gray background
ax.set_title("K-Means Clustering: Centroid Movement")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Plot the data points initially
scatters = [
    ax.scatter([], [], c=color, s=30, label=f"Cluster {i+1}")
    for i, color in enumerate(colors)
]
centroid_scatters = [
    ax.scatter([], [], c=centroid_colors[i], s=100, marker="X")
    for i in range(n_clusters)
]


# Function to update the plot
def update(frame):
    global centroids
    # Assign clusters based on current centroids
    labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

    # Gradually update centroids by moving a small step towards the mean
    new_centroids = np.array(
        [
            X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else centroids[i]
            for i in range(n_clusters)
        ]
    )
    centroids[:] += step_size * (new_centroids - centroids)

    # Clear and redraw the plot with updated data points and centroids
    ax.clear()
    ax.set_facecolor("#f0f0f0")
    ax.set_title(f"K-Means Clustering: Iteration {frame + 1}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Plot clusters with different colors
    for i, scatter in enumerate(scatters):
        scatter.set_offsets(X[labels == i])
        ax.scatter(
            X[labels == i][:, 0],
            X[labels == i][:, 1],
            c=colors[i],
            s=30,
            label=f"Cluster {i+1}",
        )

    # Plot centroids with different colors
    for i, scatter in enumerate(centroid_scatters):
        scatter.set_offsets(centroids[i])
        ax.scatter(
            centroids[i][0],
            centroids[i][1],
            c=centroid_colors[i],
            s=100,
            marker="X",
            label=f"Centroid {i+1}",
        )

    # Add legend
    ax.legend(loc="lower right")


# Animate the process
ani = FuncAnimation(fig, update, frames=range(75), repeat=False)

# Save the animation as a GIF
ani.save("kmeans_animation.gif", writer=PillowWriter(fps=10))

# Show the animation
plt.show()

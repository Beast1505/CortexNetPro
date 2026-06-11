# CortexNetPro
Custom PyTorch Deep Neural Network for complex non-linear pattern recognition, featuring an interactive CSV upload UI and dynamic decision boundary visualization.
# CortexNet: Deep Neural Network for Pattern Recognition

## Overview
CortexNet is a custom, multi-layer Deep Neural Network (DNN) engineered from scratch using **PyTorch**. It is specifically designed to solve complex, non-linearly separable classification problems where traditional linear algorithms fail. 

Built to handle highly noisy 2D spatial data, the model dynamically maps and visualizes decision boundaries for challenging geometries such as concentric circles ("bullseyes"), interlocking spirals, and XOR crosshairs.

## Key Features
* **Custom Architecture:** Engineered with multiple hidden layers and ReLU activation functions to bend and warp decision boundaries around complex data clusters.
* **Overfitting Prevention:** Strategically integrated **Dropout layers** and **L2 Regularization** (weight decay) via the Adam optimizer to maintain high validation accuracy on noisy datasets.
* **Hardware Acceleration:** Seamlessly utilizes Colab's cloud-hosted NVIDIA GPUs (**CUDA**) for parallel matrix operations to slash training latency.
* **Interactive Data Processing:** Features a custom UI built with `ipywidgets` that allows users to upload their own spatial data via CSV. The pipeline automatically scales the data (`StandardScaler`), trains the network, and renders the results.
* **Dynamic Visualization:** Leverages `matplotlib` and `numpy` meshgrids to visually plot the network's mathematical decision boundary in real-time.

## Technologies Used
* **Framework:** PyTorch
* **Language:** Python
* **Data Processing:** Pandas, NumPy, Scikit-learn
* **Visualization:** Matplotlib
* **Environment:** Google Colab, Jupyter Notebooks

## Datasets Conquered
CortexNet has been rigorously stress-tested against highly complex, mathematically generated spatial geometries, consistently achieving **90%+ validation accuracy**:
1. **The Bullseye:** Concentric circles with overlapping noise.
2. **Interlocking Spirals:** Two continuous, intertwined geometric spirals.
3. **The XOR Crosshair:** Four opposing quadrants forcing simultaneous multi-feature evaluation.
4. **Extreme Moons:** Interlocking crescent shapes with a 35% noise overlap.

---
*Engineered by Taresh Atrey*

# Attribution Methods for Interpreting and Optimizing Deep Neural Networks

This notebook provides an explanation of the key concepts and methods from Marco Ancona's doctoral thesis on attribution methods for interpreting and optimizing deep neural networks.

## Table of Contents
1. [Introduction to Attribution Methods](#introduction)
2. [Gradient-Based Attribution Methods](#gradient-based)
3. [Perturbation-Based Attribution Methods](#perturbation-based)
4. [Integrated Gradients and Path Methods](#integrated-gradients)
5. [Gradient Sensitivity Analysis](#gradient-sensitivity)
6. [DeepLIFT and Layer-wise Relevance Propagation](#deeplift)
7. [Applications of Attribution Methods](#applications)
8. [Experimental Implementation](#experiments)

## 1. Introduction to Attribution Methods <a name="introduction"></a>

Attribution methods aim to explain the predictions of deep neural networks by attributing importance scores to input features. These methods help understand what parts of the input contribute most to a particular output.

Let's first set up our environment with the necessary libraries:

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.cm as cm

# For visualization
from PIL import Image
import requests
from io import BytesIO
```

### Core Concepts

At a fundamental level, attribution methods analyze how changes in the input affect changes in the output of a neural network. 

Mathematically, given a neural network $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ and an input $x \in \mathbb{R}^n$, an attribution method aims to assign an attribution or relevance score $R_i(f, x)$ to each input feature $x_i$, indicating its contribution to the output $f(x)$.

Let's define a simple neural network to demonstrate these concepts:

```python
class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize a network
model = SimpleConvNet()
```

## 2. Gradient-Based Attribution Methods <a name="gradient-based"></a>

Gradient-based methods are among the simplest approaches for generating attributions. They use the gradient of the output with respect to the input features.

### Vanilla Gradients

The simplest gradient-based method is the vanilla gradient, where the attribution is defined as:

$$R(x_i) = \frac{\partial f(x)}{\partial x_i}$$

Implementation:

```python
def vanilla_gradients(model, inputs, target_class=None):
    """
    Compute vanilla gradients for a given model and input.
    
    Args:
        model: Neural network model
        inputs: Input tensor (requires_grad=True)
        target_class: Target class index (if None, uses the predicted class)
    
    Returns:
        Gradient attributions
    """
    # Ensure inputs require gradients
    inputs.requires_grad_(True)
    
    # Forward pass
    outputs = model(inputs)
    
    # If no target class is provided, use the predicted class
    if target_class is None:
        target_class = outputs.argmax(dim=1)
    
    # Create a one-hot encoding for the target class
    one_hot = torch.zeros_like(outputs)
    one_hot.scatter_(1, target_class.view(-1, 1), 1)
    
    # Backward pass
    outputs.backward(gradient=one_hot)
    
    # Get the gradients
    gradients = inputs.grad.clone()
    
    # Reset gradients
    inputs.grad.zero_()
    
    return gradients
```

### Gradient × Input

A simple but effective modification is to multiply the gradient by the input:

$$R(x_i) = x_i \cdot \frac{\partial f(x)}{\partial x_i}$$

Implementation:

```python
def gradient_x_input(model, inputs, target_class=None):
    """
    Compute gradient * input for a given model and input.
    
    Args:
        model: Neural network model
        inputs: Input tensor (requires_grad=True)
        target_class: Target class index (if None, uses the predicted class)
    
    Returns:
        Gradient * Input attributions
    """
    # Compute vanilla gradients
    gradients = vanilla_gradients(model, inputs, target_class)
    
    # Multiply gradients by input
    attributions = gradients * inputs
    
    return attributions
```

### GuidedBackprop

Guided Backpropagation modifies the backpropagation process by only allowing positive gradients to flow through ReLU activations, effectively combining vanilla gradients with DeConvNet.

```python
# Hook to capture and modify gradients during backpropagation
class GuidedBackpropReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        positive_mask = (input > 0).float()
        output = input * positive_mask
        ctx.save_for_backward(input, positive_mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, positive_mask = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Zero out gradients for negative inputs
        grad_input = grad_input * positive_mask
        # Zero out negative gradients
        grad_input = grad_input * (grad_input > 0).float()
        return grad_input

class ModelWithGuidedBackprop(nn.Module):
    def __init__(self, model):
        super(ModelWithGuidedBackprop, self).__init__()
        self.model = model
        # Replace all ReLU activations with GuidedBackpropReLU
        self._replace_relu()
        
    def _replace_relu(self):
        for name, module in self.model.named_children():
            if isinstance(module, nn.ReLU):
                setattr(self.model, name, GuidedBackpropReLU.apply)
            elif isinstance(module, nn.Sequential):
                for idx, submodule in enumerate(module):
                    if isinstance(submodule, nn.ReLU):
                        module[idx] = GuidedBackpropReLU.apply
            else:
                self._replace_relu(module)
                
    def forward(self, x):
        return self.model(x)

def guided_backprop(model, inputs, target_class=None):
    """
    Compute GuidedBackprop attributions for a given model and input.
    
    Args:
        model: Neural network model
        inputs: Input tensor (requires_grad=True)
        target_class: Target class index (if None, uses the predicted class)
    
    Returns:
        GuidedBackprop attributions
    """
    # Create a model with GuidedBackprop
    guided_model = ModelWithGuidedBackprop(model)
    
    # Compute vanilla gradients with the guided model
    attributions = vanilla_gradients(guided_model, inputs, target_class)
    
    return attributions
```

## 3. Perturbation-Based Attribution Methods <a name="perturbation-based"></a>

Perturbation-based methods analyze the effect of perturbing input features on the model's output. These methods are conceptually simple but computationally intensive.

### LIME (Local Interpretable Model-agnostic Explanations)

LIME approximates a complex model locally using a simple, interpretable model:

```python
# A simple implementation of LIME for images
def lime_for_images(model, image, num_samples=1000, num_features=10):
    """
    Simplified LIME implementation for image classification.
    
    Args:
        model: Neural network model
        image: Input image tensor
        num_samples: Number of perturbed samples to generate
        num_features: Number of top features to return
    
    Returns:
        Top features with their importance scores
    """
    # Original prediction
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        prediction = output.argmax(dim=1).item()
    
    # Generate perturbed samples
    perturbations = []
    binary_mask = []
    
    # Image dimensions
    _, h, w = image.shape
    
    # Create superpixels (simplified - just using grids)
    grid_size = 8
    num_segments = (h // grid_size) * (w // grid_size)
    segments = np.zeros((h, w))
    
    for i in range(h // grid_size):
        for j in range(w // grid_size):
            segments[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size] = i * (w // grid_size) + j
    
    # Generate perturbed samples
    for _ in range(num_samples):
        # Random binary mask for segments
        mask = np.random.randint(0, 2, num_segments)
        binary_mask.append(mask)
        
        # Create perturbed image
        perturbed_image = image.clone()
        for segment_id in range(num_segments):
            if mask[segment_id] == 0:  # If segment is turned off
                segment_indices = np.where(segments == segment_id)
                perturbed_image[:, segment_indices[0], segment_indices[1]] = 0
        
        perturbations.append(perturbed_image)
    
    # Stack perturbations
    perturbed_batch = torch.stack(perturbations)
    
    # Get predictions for perturbed samples
    with torch.no_grad():
        outputs = model(perturbed_batch)
        scores = outputs[:, prediction].numpy()
    
    # Convert binary masks to numpy array
    binary_mask = np.array(binary_mask)
    
    # Fit a linear model (Ridge regression)
    from sklearn.linear_model import Ridge
    
    # Fit linear model to predict target score from binary features
    interpreter = Ridge(alpha=1.0)
    interpreter.fit(binary_mask, scores)
    
    # Get feature importances
    importances = interpreter.coef_
    
    # Get top features
    top_features = np.argsort(np.abs(importances))[-num_features:]
    
    return {segment_id: importances[segment_id] for segment_id in top_features}
```

### Occlusion Sensitivity

Occlusion sensitivity systematically occludes parts of the input and measures the change in prediction:

```python
def occlusion_sensitivity(model, image, target_class=None, patch_size=8, stride=4):
    """
    Compute occlusion sensitivity map for an image.
    
    Args:
        model: Neural network model
        image: Input image tensor (C, H, W)
        target_class: Target class index
        patch_size: Size of the occlusion patch
        stride: Stride of the sliding window
        
    Returns:
        Sensitivity map
    """
    # Set model to evaluation mode
    model.eval()
    
    # Get image dimensions
    _, h, w = image.shape
    
    # Original prediction and score
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        original_score = output[0, target_class].item()
    
    # Initialize sensitivity map
    sensitivity_map = torch.zeros(h, w)
    
    # Sliding window
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            # Create a copy of the input image
            occluded_image = image.clone()
            
            # Apply occlusion patch (set to zero)
            occluded_image[:, i:i+patch_size, j:j+patch_size] = 0
            
            # Predict on occluded image
            with torch.no_grad():
                output = model(occluded_image.unsqueeze(0))
                occluded_score = output[0, target_class].item()
            
            # Calculate score difference
            score_diff = original_score - occluded_score
            
            # Update sensitivity map
            sensitivity_map[i:i+patch_size, j:j+patch_size] += score_diff
    
    return sensitivity_map
```

## 4. Integrated Gradients and Path Methods <a name="integrated-gradients"></a>

Integrated Gradients (IG) addresses the gradient saturation problem by integrating gradients along a path from a baseline to the input.

The mathematical formulation for IG is:

$$IG_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial f(x' + \alpha \times (x - x'))}{\partial x_i} d\alpha$$

where $x'$ is the baseline input (often a zero tensor).

```python
def integrated_gradients(model, inputs, baseline=None, target_class=None, steps=50):
    """
    Compute Integrated Gradients attributions.
    
    Args:
        model: Neural network model
        inputs: Input tensor
        baseline: Baseline input (if None, uses zero tensor)
        target_class: Target class index
        steps: Number of steps for the path integral
        
    Returns:
        Integrated Gradients attributions
    """
    # If baseline is not provided, use a zero tensor
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    
    # Generate path inputs
    path_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(steps + 1)]
    path_inputs = torch.stack(path_inputs)
    
    # Batch compute gradients
    gradients = []
    for path_input in path_inputs:
        path_input = path_input.detach().requires_grad_(True)
        output = model(path_input.unsqueeze(0) if path_input.dim() == 3 else path_input)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Create a one-hot encoding for the target class
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, target_class.view(-1, 1), 1)
        
        # Backward pass
        output.backward(gradient=one_hot)
        
        # Get the gradients
        gradient = path_input.grad.clone()
        gradients.append(gradient)
        
    # Stack gradients
    gradients = torch.stack(gradients)
    
    # Approximate the integral using the trapezoidal rule
    grads = (gradients[:-1] + gradients[1:]) / 2.0
    avg_grads = torch.mean(grads, dim=0)
    
    # Scale attributions by the input difference
    integrated_grads = (inputs - baseline) * avg_grads
    
    return integrated_grads
```

### Expected Gradients

Expected Gradients (EG) extends IG by using a dataset distribution for the baseline:

```python
def expected_gradients(model, inputs, reference_samples, target_class=None, num_samples=10):
    """
    Compute Expected Gradients attributions.
    
    Args:
        model: Neural network model
        inputs: Input tensor
        reference_samples: Dataset of reference samples to use as baselines
        target_class: Target class index
        num_samples: Number of reference samples to use
        
    Returns:
        Expected Gradients attributions
    """
    # Sample random baseline inputs from the reference dataset
    if len(reference_samples) < num_samples:
        num_samples = len(reference_samples)
    
    indices = np.random.choice(len(reference_samples), num_samples, replace=False)
    baselines = [reference_samples[i] for i in indices]
    
    # Compute IG for each baseline
    all_attributions = []
    for baseline in baselines:
        attributions = integrated_gradients(model, inputs, baseline, target_class)
        all_attributions.append(attributions)
    
    # Average attributions across baselines
    expected_attributions = torch.mean(torch.stack(all_attributions), dim=0)
    
    return expected_attributions
```

## 5. Gradient Sensitivity Analysis <a name="gradient-sensitivity"></a>

Gradient sensitivity analysis examines how small changes in input features affect the model's predictions.

### Sensitivity Maps

Sensitivity maps measure how sensitive the output is to small perturbations of the input:

```python
def sensitivity_map(model, inputs, epsilon=0.01, target_class=None):
    """
    Compute sensitivity map for a given model and input.
    
    Args:
        model: Neural network model
        inputs: Input tensor
        epsilon: Small perturbation value
        target_class: Target class index
        
    Returns:
        Sensitivity map
    """
    # Create a copy of the input
    perturbed_inputs = inputs.clone().detach().requires_grad_(True)
    
    # Forward pass
    outputs = model(perturbed_inputs.unsqueeze(0) if perturbed_inputs.dim() == 3 else perturbed_inputs)
    
    if target_class is None:
        target_class = outputs.argmax(dim=1)
    
    # Create a one-hot encoding for the target class
    one_hot = torch.zeros_like(outputs)
    one_hot.scatter_(1, target_class.view(-1, 1), 1)
    
    # Backward pass
    outputs.backward(gradient=one_hot)
    
    # Get the gradients
    gradients = perturbed_inputs.grad.clone()
    
    # Compute sensitivity map
    sensitivity = torch.abs(gradients)
    
    return sensitivity
```

### SmoothGrad

SmoothGrad reduces noise by averaging gradients over multiple noisy versions of the input:

```python
def smoothgrad(model, inputs, target_class=None, noise_level=0.1, num_samples=50):
    """
    Compute SmoothGrad attributions.
    
    Args:
        model: Neural network model
        inputs: Input tensor
        target_class: Target class index
        noise_level: Standard deviation of Gaussian noise
        num_samples: Number of noisy samples
        
    Returns:
        SmoothGrad attributions
    """
    # Generate noisy samples
    noisy_inputs = []
    for _ in range(num_samples):
        noise = torch.randn_like(inputs) * noise_level * (inputs.max() - inputs.min())
        noisy_input = inputs + noise
        noisy_inputs.append(noisy_input)
    
    # Compute gradients for each noisy sample
    all_gradients = []
    for noisy_input in noisy_inputs:
        noisy_input = noisy_input.detach().requires_grad_(True)
        output = model(noisy_input.unsqueeze(0) if noisy_input.dim() == 3 else noisy_input)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Create a one-hot encoding for the target class
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, target_class.view(-1, 1), 1)
        
        # Backward pass
        output.backward(gradient=one_hot)
        
        # Get the gradients
        gradient = noisy_input.grad.clone()
        all_gradients.append(gradient)
    
    # Average the gradients
    smoothed_gradients = torch.mean(torch.stack(all_gradients), dim=0)
    
    return smoothed_gradients
```

## 6. DeepLIFT and Layer-wise Relevance Propagation <a name="deeplift"></a>

### DeepLIFT (Deep Learning Important FeaTures)

DeepLIFT compares activations to a reference activation and assigns contribution scores based on the difference:

$$C_{\Delta x \Delta t} = \frac{\Delta t}{\Delta x} \cdot (x - x_0)$$

where $\Delta t$ is the difference in target output, and $\Delta x$ is the difference in input from the reference input $x_0$.

```python
# A simplified version of DeepLIFT's Rescale rule implementation
class DeepLIFTRescale:
    def __init__(self, model, baseline):
        self.model = model
        self.baseline = baseline
        self.activations = {}  # Stores activations for each layer
        self.baseline_activations = {}  # Stores baseline activations
        self.deltas = {}  # Stores differences in activations
        self.relevances = {}  # Stores relevance scores
        
        # Register hooks to capture activations
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output, name):
            self.activations[name] = output.detach()
            
        # Register hooks for each layer
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
                module.register_forward_hook(
                    lambda module, input, output, name=name: forward_hook(module, input, output, name)
                )
    
    def compute_baseline_activations(self):
        """Compute activations for the baseline input"""
        _ = self.model(self.baseline)
        self.baseline_activations = {k: v.clone() for k, v in self.activations.items()}
        
    def compute_relevance(self, inputs, target_class=None):
        """Compute DeepLIFT relevance scores"""
        # Forward pass for the actual input
        output = self.model(inputs)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Initialize relevance with the output difference
        output_diff = output - self.baseline_output
        self.relevances[list(self.activations.keys())[-1]] = output_diff[:, target_class]
        
        # Backward pass to compute relevances
        for i in range(len(self.activations) - 1, 0, -1):
            layer_name = list(self.activations.keys())[i]
            prev_layer_name = list(self.activations.keys())[i-1]
            
            # Compute difference in activations
            delta_out = self.activations[layer_name] - self.baseline_activations[layer_name]
            delta_in = self.activations[prev_layer_name] - self.baseline_activations[prev_layer_name]
            
            # Skip if delta_out is zero
            if torch.all(delta_out == 0):
                self.relevances[prev_layer_name] = torch.zeros_like(delta_in)
                continue
            
            # Compute multiplier (delta_out / delta_in)
            # Add small epsilon to avoid division by zero
            multiplier = delta_out / (delta_in + 1e-10)
            
            # Propagate relevance
            self.relevances[prev_layer_name] = multiplier * self.relevances[layer_name]
        
        # Return input layer relevance
        return self.relevances[list(self.activations.keys())[0]]
```

### Layer-wise Relevance Propagation (LRP)

LRP attributes relevance scores to each neuron by redistributing the prediction backward through the network:

$$R_i^{(l)} = \sum_j \frac{a_i^{(l)} w_{ij}^{(l,l+1)}}{\sum_i a_i^{(l)} w_{ij}^{(l,l+1)}} R_j^{(l+1)}$$

```python
# A simplified implementation of LRP for a basic neural network
class LRP:
    def __init__(self, model):
        self.model = model
        self.activations = {}  # Stores activations for each layer
        self.relevances = {}  # Stores relevance scores
        
        # Register hooks to capture activations
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output, name):
            self.activations[name] = output.detach()
            
        # Register hooks for each layer
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
                module.register_forward_hook(
                    lambda module, input, output, name=name: forward_hook(module, input, output, name)
                )
    
    def compute_relevance(self, inputs, target_class=None, epsilon=1e-9):
        """Compute LRP relevance scores using the epsilon-rule"""
        # Forward pass
        output = self.model(inputs)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Initialize relevance at the output layer
        relevance = torch.zeros_like(output)
        relevance[:, target_class] = output[:, target_class]
        
        # Backward pass to compute relevances
        layers = list(self.activations.keys())
        self.relevances[layers[-1]] = relevance
        
        # Iterate backward through layers
        for i in range(len(layers) - 1, 0, -1):
            current_layer = self.model._modules[layers[i]]
            prev_layer = self.model._modules[layers[i-1]]
            
            if isinstance(current_layer, nn.Linear):
                # Linear layer propagation
                weights = current_layer.weight
                activations = self.activations[layers[i-1]]
                
                # Compute denominator (z + small epsilon to avoid division by zero)
                z = torch.mm(activations, weights.t()) + current_layer.bias + epsilon
                
                # Propagate relevance
                s = self.relevances[layers[i]] / z
                c = torch.mm(s, weights)
                self.relevances[layers[i-1]] = activations * c
            
            elif isinstance(current_layer, nn.Conv2d):
                # Simplified for demonstration - for actual implementation,
                # you'd need to handle convolution properly
                self.relevances[layers[i-1]] = torch.ones_like(self.activations[layers[i-1]])
        
        # Return input layer relevance
        return self.relevances[layers[0]]
```

## 7. Applications of Attribution Methods <a name="applications"></a>

Attribution methods have several important applications in deep learning:

### Model Debugging and Understanding

```python
def visualize_attributions(image, attributions, cmap='seismic', alpha=0.5):
    """
    Visualize attributions overlaid on the original image.
    
    Args:
        image: Original image tensor (C, H, W)
        attributions: Attribution map tensor (C, H, W)
        cmap: Colormap for visualization
        alpha: Transparency of the overlay
        
    Returns:
        Visualization figure
    """
    # Convert tensors to numpy arrays
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(attributions, torch.Tensor):
        attributions = attributions.detach().cpu().numpy()
    
    # For RGB images, take the mean across channels for attributions
    if image.shape[0] == 3:
        attributions = np.mean(attributions, axis=0)
        img = np.transpose(image, (1, 2, 0))
    else:
        img = image[0]
    
    # Normalize attributions to [-1, 1]
    attributions = attributions / (np.max(np.abs(attributions)) + 1e-10)
    
    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    # Attributions
    ax[1].imshow(img)
    im = ax[1].imshow(attributions, cmap=cmap, alpha=alpha, vmin=-1, vmax=1)
    ax[1].set_title('Attribution Overlay')
    ax[1].axis('off')
    
    # Add colorbar
    fig.colorbar(im, ax=ax[1], orientation='vertical', label='Attribution')
    
    plt.tight_layout()
    return fig
```

### Sensitivity Analysis for Feature Selection

```python
def feature_importance_ranking(model, dataset, num_features=10):
    """
    Rank features by importance using integrated gradients.
    
    Args:
        model: Neural network model
        dataset: Dataset of samples
        num_features: Number of top features to return
        
    Returns:
        Ranked list of feature indices
    """
    all_attributions = []
    
    # Compute attributions for each sample
    for sample, label in dataset:
        sample = sample.unsqueeze(0)  # Add batch dimension
        
        # Compute integrated gradients
        attributions = integrated_gradients(model, sample, target_class=label)
        
        # Flatten and take absolute value
        flat_attr = torch.abs(attributions).view(-1)
        all_attributions.append(flat_attr)
    
    # Average attributions across samples
    avg_attributions = torch.mean(torch.stack(all_attributions), dim=0)
    
    # Get top feature indices
    _, top_indices = torch.topk(avg_attributions, num_features)
    
    return top_indices.tolist()
```

### Testing for Bias and Fairness

```python
def analyze_bias_with_attributions(model, dataset, sensitive_features, target_class):
    """
    Analyze potential bias using attribution methods.
    
    Args:
        model: Neural network model
        dataset: Dataset of samples
        sensitive_features: Indices of sensitive features
        target_class: Target class to analyze
        
    Returns:
        Bias analysis results
    """
    # Initialize results
    results = {
        'sensitive_feature_importance': [],
        'non_sensitive_feature_importance': []
    }
    
    for sample, label in dataset:
        if label == target_class:
            # Compute attributions
            attributions = integrated_gradients(model, sample, target_class=label)
            
            # Flatten attributions
            flat_attr = torch.abs(attributions).view(-1)
            
            # Separate attributions for sensitive and non-sensitive features
            sensitive_attr = flat_attr[sensitive_features].mean().item()
            mask = torch.ones(flat_attr.size(0), dtype=torch.bool)
            mask[sensitive_features] = False
            non_sensitive_attr = flat_attr[mask].mean().item()
            
            results['sensitive_feature_importance'].append(sensitive_attr)
            results['non_sensitive_feature_importance'].append(non_sensitive
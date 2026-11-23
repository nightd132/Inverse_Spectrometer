# Inverse_Spectrometer

## Introduction

Project 2 trains a neural network that maps an image to three continuous values (X, Y, Z). The network is trained on ~100,000 simulated images, which have labels, but it has to work on ~1,500 experimental images. Simulation and experiment look different (a domain gap), so the main challenge is making the model transfer from one to the other.

## Objective

Train and compare three models on the same backbone, split and seed:

1. **Baseline:** CNN feature extractor + regressor, trained on simulation only (MSE).
2. **Augmentation:** the same model, trained on augmented simulation data.
3. **Domain adaptation:** the same model plus a domain discriminator, trained adversarially so that simulated and experimental features become indistinguishable.

Evaluate each model on the simulated test set and on the experimental data, and compare how well each one closes the gap (error metrics and t-SNE of the latent space).

## Idea

### 1. Training and evaluate with simulation images

```mermaid
flowchart LR
    SIM["Sim H5<br/>images + XYZ"] --> TR["Train+Eval Sim H5<br/>images + XYZ"]
    SIM --> EV["Test Sim H5<br/>images + XYZ"]
    TR --> |Train+Tune|RES["Regressor<br/>CNN + FNN???"]
    EV --> |Evaluate| RES
    RES --> COOR["X, Y, Z"]
```

### 2. Evaluate with simulation images

```mermaid
flowchart LR
    EXP["Exp H5<br/>images + XYZ"] --> RES["Treained Regressor<br/>CNN + FNN???"]
    
    RES --> COOR["X, Y, Z"]
```

### 3. Improve experimental evaluations via data augmentations

```mermaid
flowchart LR
    SIM["Sim H5<br/>images + XYZ"] --> AUG["Augmentation"]
    AUG --> RES["Regressor<br/>CNN + FNN???"]
    EXP["Exp H5<br/>images + XYZ"] --> RES
    RES --> COOR["X, Y, Z"]
```

### 4. Improve experimental evaluations via domain adaptation

```mermaid
flowchart LR
    SIM["Sim H5<br/>images + XYZ"] --> AUG["Augmentation<br/>"]
    AUG --> E
    EXP["Exp H5<br/>images only"] --> E

    E["Feature Extractor<br/>CNN: ??? layers, latent dim ???"]
    E -->|sim features| R["Regressor<br/>MLP: ???"]
    E -->|sim, 0 + exp, 1| D["Discriminator<br/>MLP: ???"]

    R --> XYZ["X, Y, Z"]
    XYZ --> L1["MSE loss<br/>vs true XYZ"]
    D --> DOM["sim / exp logit"]
    DOM --> L2["BCE loss"]

    L1 -->|"optimize"| R
    L1 -->|"optimize"| E
    L2 -->|"true labels, optimize Disciminator"| D
    L2 -->|"flipped labels, optimuze Extractor"| E
```

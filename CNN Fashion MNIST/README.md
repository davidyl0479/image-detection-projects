# Fashion-MNIST CNN Models

This project contains three convolutional neural network (CNN) architectures trained on the Fashion-MNIST dataset:

- **Baseline CNN**
- **Deeper CNN** (two conv layers per block)
- **Wider CNN** (more channels per block)

The aim is to compare how model depth and width affect performance while keeping the codebase simple and easy to follow.

---

## Project Structure
.
├── cnn_model.py # Baseline, Deeper, Wider CNN architectures

├── utils.py # Data loading, training loop, evaluation, plotting

├── notebook.ipynb # Full experiment: training, evaluation, plots

└── README.md

## Models

All models follow the same structure of three convolutional blocks and a two-layer classifier.

- Baseline: 1 conv per block
- Deeper: 2 convs per block
- Wider: More channels per block (64→128→256)

All models output 10 logits, one for each Fashion-MNIST class.

## Results Summary

Test accuracy:
| Model        | Accuracy |
| ------------ | -------- |
| Baseline CNN | 0.9040   |
| Deeper CNN   | 0.9143   |
| Wider CNN    | 0.9047   |

The deeper architecture performs best overall.

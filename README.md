# More-To-Fed-Learning

Federated learning (FL) experiments on Fashion-MNIST and UCI Adult Census, built on a hand-written neural network (no autograd). Includes a replication of the model-poisoning attacks from Bhagoji et al., *Analyzing Federated Learning through an Adversarial Lens* (ICML 2019).

## Notebooks (`Federated Learning from scratch/`)

| Notebook | Data | Model | What it does | Test acc* |
|---|---|---|---|---|
| `FL_mlp_iid.ipynb` | Fashion-MNIST, IID | hand-written MLP | Clean FL baseline: 10 clients, size-weighted FedAvg | 87.7% |
| `FL_mlp_non_iid.ipynb` | Fashion-MNIST, non-IID (Dirichlet α=0.5) | hand-written MLP | Same, with clients holding skewed class mixes | 85.0% |
| `FL_cnn_non_iid.ipynb` | Fashion-MNIST, non-IID | hand-written network + PyTorch conv layers | CNN version (`model_type` switch: `cnn` / `mlp`) | 90.1% |
| `FL_torch_cnn_non_iid.ipynb` | Fashion-MNIST, non-IID | PyTorch `nn.Module` | Same setup with built-in layers and autograd, ~3× faster | 90.8% |
| **`FL_mlp_attack_adult_iid.ipynb`** | **Adult Census, IID** | hand-written MLP (104-256-2) | **Model-poisoning attacks and defenses from Bhagoji et al.** | see below |

\*Single run, 10 rounds.

## Adult Census attack (Bhagoji et al. replication)

One of the 10 clients tries to make the global model misclassify one chosen record while keeping it accurate otherwise. Pick the setup in the notebook's settings cell:

- `attack_type`: `baseline` (explicit boosting), `stealthy`, or `altmin` (alternating minimization)
- `aggregation`: `fedavg`, `krum`, or `coordinate_median`

Settings follow the paper's code ([inspire-group/ModelPoisoning](https://github.com/inspire-group/ModelPoisoning)).

- **Reproduced:**
  - Boosting flips the target but cuts accuracy to about 80%.
  - The stealthy attacks flip it, keep about 84% accuracy, and pass the server's accuracy check.
- **Not reproduced on Adult:** the attacker's updates don't look benign by distance, so Krum and coordinate median stop it. This holds across 5 seeds and both label directions. The paper shows its Krum and median results on Fashion-MNIST.

The full results table and caveats are in the notebook's "What we did and found" cell.

## Running

- **Environment:** Python with `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn` and `tqdm`. Everything runs on CPU.
- **Working directory:** run the notebooks from their own folder, because the data paths are relative.
- **Fashion-MNIST** downloads itself into `data/` on first run.
- **Adult Census** needs a one-time download (`data/` is gitignored):
  ```bash
  mkdir -p "Federated Learning from scratch/data/adult" && cd "Federated Learning from scratch/data/adult"
  curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
  curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
  ```
- **After cloning,** enable the notebook clean filter. It keeps per-run noise such as execution counts out of commits:
  ```bash
  git config filter.nbnormalize.clean "python3 tools/nb_normalize.py"
  ```

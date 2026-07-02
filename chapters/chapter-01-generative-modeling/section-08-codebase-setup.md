# Section 1.8: Codebase Setup

> **Source inheritance:** Foster, Ch. 1 - `generative-deep-learning` repository and Docker environment  
> **Enhanced with:** GPU verification, TensorFlow setup, project structure tour  
> **Vocabulary:** [GLOSSARY.md](../../GLOSSARY.md)    
> **Math conventions:** [MATH_CONVENTIONS.md](../../MATH_CONVENTIONS.md)

---

## Why Reproducible Environments Matter

Generative deep learning is sensitive to software versions. TensorFlow 2.10 vs 2.15, CUDA 11.8 vs 12.x, a mismatched `cuDNN` - any of these can turn a training run into a cryptic GPU error or silently slow CPU fallback.

Foster ships a **Docker image** with the book's exact dependencies. Think of it as a pre-packed flight case: every cable labeled, every tool in its slot. You spend time learning [generative models](../../GLOSSARY.md#generative-model), not debugging `libcuda.so`.

> **Readable form:** Docker is a shipping container for software. The book's code runs the same on your laptop, a cloud VM, or a colleague's machine - because the container *is* the machine.


---

## Prerequisites Checklist

Before you begin, confirm:

| Requirement | Minimum | Recommended
|-------------|---------|-------------|
| **GPU** | NVIDIA with 8 GB VRAM | 16+ GB (RTX 3080/4080, A100) |
| **Driver** | NVIDIA driver ≥ 525 | Latest stable |
| **Docker** | Docker Engine 20+ | Docker Desktop or native |
| **Disk** | 20 GB free | 50 GB (datasets grow fast) |
| **RAM** | 16 GB | 32 GB |
| **Git** | Any recent version | - |

CPU-only works for [Section 1.4](./section-04-our-first-generative-model.md) (coin flip) and small experiments. From [Chapter 03](../chapter-03-variational-autoencoders/README.md) onward, a GPU is effectively required.

You should already be comfortable with TensorFlow/Keras from [Course 1, Chapters 08-09](https://github.com/Collaborative-ai/ai-academy-applied-ml-engineering/blob/main/chapters/chapter-08-deep-learning/README.md).

---

## Step 1: Clone Foster's Repository

```bash
# Create a workspace directory
mkdir -p ~/gdl-workspace && cd ~/gdl-workspace

# Clone the official book codebase
git clone https://github.com/davidADSP/GDL_code.git generative-deep-learning
cd generative-deep-learning

# Check out the branch matching the 2nd edition (if applicable)
git branch -a
git checkout main   # or the edition-specific branch listed in the README
```

Repository: [https://github.com/davidADSP/GDL_code](https://github.com/davidADSP/GDL_code)

Browse the structure:

```
generative-deep-learning/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── notebooks/          # Chapter notebooks
│   ├── 01_intro/
│   ├── 03_vae/
│   ├── 04_gan/
│   └── ...
├── models/             # Reusable model definitions
└── data/               # Downloaded datasets land here
```

---

## Step 2: Build the Docker Image

```bash
cd ~/gdl-workspace/generative-deep-learning

# Build the image (first run takes 10-20 minutes)
docker build -t gdl:latest .

# Alternative: use docker-compose if provided
docker compose build
```

> **Readable form:** `docker build` reads the Dockerfile recipe - base OS, CUDA, Python, TensorFlow, Jupyter - and bakes it into an image called `gdl:latest`.

### Run the container with GPU access

```bash
# NVIDIA Container Toolkit must be installed on the host
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

docker run --gpus all -it --rm \
  -p 8888:8888 \
  -v $(pwd):/workspace \
  -v $(pwd)/data:/workspace/data \
  --name gdl-dev \
  gdl:latest bash
```

| Flag | Purpose
|------|---------|
| `--gpus all` | Pass all host GPUs into the container |
| `-p 8888:8888` | Expose Jupyter on localhost:8888 |
| `-v $(pwd):/workspace` | Mount code for live editing |
| `-it` | Interactive terminal |

### docker-compose shortcut

```bash
docker compose up -d
docker compose exec gdl bash
```

---

## Step 3: Verify TensorFlow GPU

Inside the container (or your native venv), run:

```python
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())

gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs detected: {len(gpus)}")
for gpu in gpus:
    print(f"  {gpu}")

if gpus:
    # Optional: limit memory growth (prevents TF grabbing all VRAM)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Quick GPU computation test
    with tf.device("/GPU:0"):
        a = tf.random.normal((2000, 2000))
        b = tf.random.normal((2000, 2000))
        c = tf.matmul(a, b)
    print("GPU matmul successful:", c.shape)
else:
    print("WARNING: No GPU detected - training will be slow.")
```

Expected output (example):

```
TensorFlow version: 2.15.0
Built with CUDA: True
GPUs detected: 1
  PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
GPU matmul successful: (2000, 2000)
```

Save this as `verify_gpu.py` and run:

```bash
python verify_gpu.py
```

**Bold milestone:** If you see `GPUs detected: 1` (or more), your environment is ready for Part II.

---

## Step 4: Native Install (Alternative to Docker)

If you prefer a local virtual environment:

```bash
# Ubuntu / Linux with NVIDIA GPU
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip

python3 -m venv ~/gdl-venv
source ~/gdl-venv/bin/activate

pip install --upgrade pip
pip install tensorflow[and-cuda]   # TF 2.15+ bundles CUDA via pip
pip install tensorflow-probability
pip install jupyterlab matplotlib numpy pandas scikit-learn

# Verify
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

For specific CUDA/driver pairings, see the official guide: [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip)

| Component | Typical version (2024-2026)
|-----------|----------------------------|
| TensorFlow | 2.15 - 2.16 |
| CUDA (bundled) | 12.x |
| cuDNN (bundled) | 8.9+ |
| Python | 3.10 - 3.11 |

---

## Step 5: Launch Jupyter and Run Chapter 1

```bash
# Inside container or venv
cd /workspace   # or repo root
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Open the URL printed in the terminal (includes a token). Navigate to `notebooks/01_intro/` and run the Chapter 1 notebook - Foster's coin-flip [generative model](../../GLOSSARY.md#generative-model) from [Section 1.4](./section-04-our-first-generative-model.md).

### Quick coin-flip sanity check (no notebook required)

```python
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

flips = tf.constant([1., 0., 1., 1., 0., 1., 1., 1.])
theta = tf.Variable(0.5)

coin = lambda: tfd.Bernoulli(probs=theta)
optimizer = tf.keras.optimizers.Adam(0.1)

for _ in range(100):
    with tf.GradientTape() as tape:
        loss = -tf.reduce_sum(coin().log_prob(flips))
    optimizer.apply_gradients([(tape.gradient(loss, theta), theta)])

print(f"θ = {theta.numpy():.2f}")
print("Samples:", coin().sample(10).numpy().astype(int))
```

If this runs on your GPU machine (CPU is fine here), the full pipeline is unblocked.

---

## Troubleshooting Common Issues

| Symptom | Likely cause | Fix
|---------|-------------|-----|
| `Could not load dynamic library 'libcudart.so'` | CUDA not in path | Use Docker, or `pip install tensorflow[and-cuda]` |
| `GPUs detected: 0` inside Docker | NVIDIA Container Toolkit missing | Install `nvidia-container-toolkit`, restart Docker |
| `CUDA_ERROR_OUT_OF_MEMORY` | Batch size too large | Reduce batch size; enable memory growth (above) |
| Version mismatch warnings | TF vs driver | Match versions per TF install guide |
| Jupyter won't connect | Port not forwarded | Check `-p 8888:8888` and firewall |

```bash
# Host-side GPU check (outside Docker)
nvidia-smi

# Docker GPU check
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

---

## Project Structure for This Course

As you progress, organize your work:

```
~/gdl-workspace/
├── generative-deep-learning/    # Foster's repo (don't edit upstream files)
├── my-experiments/              # Your notebooks and scripts
│   ├── chapter-01/
│   │   ├── coin_flip.py
│   │   └── taxonomy_diagram.png
│   ├── chapter-03-vae/
│   └── ...
└── data/                        # Shared datasets
```

Keep Foster's repo clean (`git pull` for updates). Copy notebooks into `my-experiments/` before modifying.

---

## Connections and Next Steps

| You just set up | You will use it in
|----------------|-------------------|
| TensorFlow + GPU | [Chapter 02](../chapter-02-deep-learning/README.md) - MLPs and CNNs |
| TensorFlow Probability | [Chapter 03](../chapter-03-variational-autoencoders/README.md) - VAE distributions |
| Jupyter workflow | Every lab in this course |
| Docker | Reproducible experiments, cloud deployment |

Probability foundations from [Section 1.6](./section-06-core-probability-theory.md) and the model map from [Section 1.7](./section-07-generative-model-taxonomy.md) are your conceptual toolkit. The environment you just built is the engineering toolkit.

**Bold milestone:** Environment verified. You are ready for [Chapter 02 - Deep Learning](../chapter-02-deep-learning/README.md).

---

## Key Vocabulary

| Term | Definition
|------|-----------|
| **Docker image** | Snapshot of OS + dependencies |
| **Container** | Running instance of an image |
| **`--gpus all`** | Docker flag to expose NVIDIA GPUs |
| **CUDA** | NVIDIA's parallel computing platform for GPUs |
| **cuDNN** | NVIDIA's deep learning primitives library |

---

## Reflection Questions

1. Why is Docker preferable to `pip install` on a shared team project?
2. What does `tf.config.list_physical_devices('GPU')` returning `[]` tell you?
3. Which chapter in this course will first require serious GPU training time?

---

## References

- Foster, D. (2023). *Generative Deep Learning* (2nd ed.). Ch. 1 - setup. [https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/](https://www.oreilly.com/library/view/generative-deep-learning/9781098134188/)
- Foster's codebase: [https://github.com/davidADSP/GDL_code](https://github.com/davidADSP/GDL_code)
- TensorFlow GPU install: [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip)
- NVIDIA Container Toolkit: [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker documentation: [https://docs.docker.com/get-started/](https://docs.docker.com/get-started/)
- TensorFlow Probability: [https://www.tensorflow.org/probability](https://www.tensorflow.org/probability)

---

**Previous:** [Section 1.7 - Generative Model Taxonomy](./section-07-generative-model-taxonomy.md)  
**Next:** [Chapter 02 - Deep Learning](../chapter-02-deep-learning/README.md)
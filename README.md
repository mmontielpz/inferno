# 🔥 Inferno

**Inferno** is a lightweight benchmarking toolkit for PyTorch models that computes:

- ✅ FLOPs (Floating Point Operations)
- ✅ MACs (Memory Access Costs)
- ✅ Memory footprint (MB)
- ✅ Inference time (ms)
- ✅ Frames Per Second (FPS)

Built for rapid validation of CNNs, with support for ResNet, MobileNet, VGG, and more. Optimized for easy integration and experimentation.

---

## ✅ Supported Models (v0.1 Experimental)

| Model           | Status   | Notes                                  |
|----------------|----------|----------------------------------------|
| `mobilenet_v2` | ✅ Passed | Validated, no warnings or crashes      |
| `resnet18`      | ✅ Passed | Fixed kernel unpack issue              |
| `resnet50`      | ✅ Passed | Confirmed after patch                  |
| `vgg16`         | ✅ Passed | Post-unpacking fix                     |
| `alexnet`       | ✅ Passed | Standard conv blocks                   |
| `squeezenet1_0` | ✅ Passed | Compatible with minor adjustments      |

> ⚠️ In future versions: we'll expand to EfficientNet, Transformer blocks (ViT, Swin), and hybrid CNNs.

---

## 🧪 How to Use

```python
from inferno import Performance
from torchvision.models import resnet18

model = resnet18(weights=None)  # no pretrained weights
perf = Performance()
perf.benchmark(model, input_shape=(32, 3, 224, 224))
```

---

## 📁 Project Structure
```
inferno/
├── inferno/           # Core package
│   ├── core.py        # Performance logic
│   ├── hooks.py       # Forward hooks for FLOPs, MACs
│   └── __init__.py    # Shortcut exports
├── examples/          # Jupyter notebooks
├── tests/             # (todo) unit tests
├── README.md          # Project description
└── setup.py           # Install config
```

---

## 🔧 Install (Local Dev)
```bash
git clone https://github.com/mmontielpz/inferno.git
cd inferno
pip install -e .
```

---

## 📦 Roadmap
- [x] CNN benchmarking support (MobileNet, ResNet, etc.)
- [ ] CLI interface (`inferno benchmark resnet18`)
- [ ] Support for Transformer models (ViT, Swin)
- [ ] Export to CSV, JSON, and comparison reports

---

MIT License © 2024 Miguel López

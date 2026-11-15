# KnowledgeProbe

Model behavior signatures and power-law spectra.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

![CI](https://github.com/kapildhinwa2010-dev/KnowledgeProbe/actions/workflows/ci.yml/badge.svg)
![CodeQL](https://github.com/kapildhinwa2010-dev/KnowledgeProbe/actions/workflows/codeql.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- Power-law fitting over singular-value spectra
- Deterministic SHA-256 behavior signatures
- Spectral distance between two models

## 📦 Install

```bash
pip install -r requirements.txt
```

## 🚀 Quick start

```python
from knowledgeprobe import behavior_signature, spectral_distance

print(behavior_signature(W))
print(spectral_distance(W1, W2))
```

## 🧪 Tests

```bash
pytest tests/ -v
```

## 📄 License

MIT — see [LICENSE](LICENSE).

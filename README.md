# KnowledgeProbe

[![CI](https://github.com/kapildhinwa2010-dev/KnowledgeProbe/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/KnowledgeProbe/actions/workflows/ci.yml)
[![CI](https://github.com/kapildhinwa2010-dev/KnowledgeProbe/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/KnowledgeProbe/actions/workflows/codeql.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Model behavior signatures: power-law spectra, SHA-256 fingerprints, spectral distance.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

## What it does

behavior_signature() turns a weight matrix into a deterministic
fingerprint (SHA-256 over quantized spectral features) plus a fitted
power-law exponent. spectral_distance() compares two matrices by their
spectral profiles - a cheap way to ask "did these models change"
without loading full weights.

## Install

    pip install -r requirements.txt

## Quick start

    from knowledgeprobe import behavior_signature, spectral_distance

    print(behavior_signature(W))
    print(spectral_distance(W1, W2))

## API

- behavior_signature(W) - deterministic signature dict
- spectral_distance(W1, W2) - float >= 0

## Honest results and limitations

- The signature captures spectral shape only: two very different
  models with similar singular-value decay can collide conceptually
  (low distance is not "same knowledge").
- Power-law exponent fitting on small matrices is noisy; treat alpha
  as an order-of-magnitude indicator.
- No experiment here links signatures to task-level knowledge; the
  name is aspirational, the mechanism is spectral.

## License

MIT - see [LICENSE](LICENSE).

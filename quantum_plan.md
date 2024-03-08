# Quantum Plan

Since the first target of building GAN models using classical machine learning methods has already been achieved in [Letter Generation with GAN (MNIST)](/notebooks/mnist/mnist_gan.ipynb) and [Stock Price Prediction with GAN](/notebooks/stock/stock_gan.ipynb). The next step is to translates the model to use Quantum General Adversarial Network (qGAN). This document details the plan moving forward including how to translate the problem into quantum space, cloud deployment plan, etc.

## Data Transformation

For stock prices, we mainly deal with time series data which is continuous. Hence, **Amplitude Encoding** or **Quantum Fourier Transform (QFT)** could be a good idea. Another idea is to do **Rotation Encoding** which allows reuse since we do not do direct encoding anymore with this approach.

#### Resources

- [Quantum Fourier Transform](https://jonathan-hui.medium.com/qc-quantum-fourier-transform-45436f90a43)
- [Quantum Embedding (PennyLane)](https://pennylane.ai/qml/glossary/quantum_embedding/)
- [Time series quantum classifiers with amplitude embedding](https://link.springer.com/article/10.1007/s42484-023-00133-0#:~:text=In%20this%20work%2C%20we%20encode,2%5En%20with%20unitary%20norm.)
- [Quantum Machine Learning in Finance: Time Series Forecasting](https://arxiv.org/abs/2202.00599)

## Parameter Considerations

Ideally, the number of qubits should be $\lt 5$ otherwise the computational cost would be too huge. As for libraries, if possible sticking with [PennyLane](https://pennylane.ai) would be a good idea since the classical counterparts uses [PyTorch](https://pytorch.org) and they play nicely together. Another option would be [Qiskit](http://qiskit.org) since there are more online resources and it has powerful visualisation tools which can help immensely when building a circuit.

## Timeline

### Study Break

- Study the resources, especially on qGANs. (refer to notebooks and reproduce them)

### Week 1️⃣ (25/03 - 29/03)

- Complete data transformation for stock prices (at least settle on an approach)

### Week 2️⃣ (1/04 - 05/04)

- Start building qGANs. (Start with a circuit design, validate ideas)

### Week 3️⃣ (08/04 - 12/04)

- Have something that can be simulated locally and continue on fixing the model.

### Week 4️⃣ (15/03 - 19/04)

- Start deploying what is available to Azure using Azure Quantum backends.

### Week 5️⃣ (22/04 - 26/04)

- Monitor on training progress and work on bug-fixing / improvements.
- Work on the report.

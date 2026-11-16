# From-Scratch Neural Networks & CNN for MNIST (Educational Project)

This is a **self-made learning project**. Everything is implemented from scratch in pure NumPy (no PyTorch, no TensorFlow, no Keras) so that every matrix multiply, every gradient, and every cache can be inspected and understood.

> Making this felt like I aged 10 years in debugging. Commonly spending 6–8 hours on debugging the latest version alone, just to find one colon was misplaced. Haha.

All the old / commented-out versions are intentionally kept. They document the real learning and debugging process.

---

## Project Structure

| File | Description |
|------|-------------|
| `Prepare_MNIST_dataset.py` | Downloads MNIST and creates the `.npy` files required by every network |
| `BinaryNeuralNet.py` | First-ever network (binary logic problems) |
| `MNIST_NeuralNet_First_Implementation.py` | Fully-connected MNIST – Sigmoid hidden layers |
| `MNIST_NeuralNet_Second_Implmentation.py` | Fully-connected MNIST – ReLU + Softmax |
| `CNN_NeuralNet_Initial_Implementation.py` | Intermediate CNN version |
| `CNN_NeuralNet_Final_Implementation.py` | Final CNN (data augmentation, LR decay, timing, best-model tracking) |

---

## How to Prepare the Data & Run Training

### 1. Install the required packages
```bash
pip install numpy matplotlib scikit-learn
```

### 2. Prepare MNIST (run once)
```bash
python Prepare_MNIST_dataset.py
```
This creates in the current directory:
- `X_train.npy`  (784, 60000)
- `Y_train.npy`  (10, 60000)   – one-hot
- `X_test.npy`   (784, 10000)
- `Y_test.npy`   (10, 10000)   – one-hot
- `y_test.npy`   (10000,)      – integer labels

### 3. Run any network
```bash
python BinaryNeuralNet.py
python MNIST_NeuralNet_First_Implementation.py
python MNIST_NeuralNet_Second_Implmentation.py
python CNN_NeuralNet_Initial_Implementation.py
python CNN_NeuralNet_Final_Implementation.py   # recommended final version
```

All scripts expect the `.npy` files to live in the **same folder** as the script you are running.

---

## Documented Results (from the terminal logs kept inside the files)

- Fully-connected ReLU network: ~98.0 % test accuracy
- Final CNN version: **~98.4 – 98.7 %** test accuracy  
  Training accuracy regularly reaches **99.7 – 99.9 %**

---

## Notes about the code

- This is an educational codebase. Clarity and the ability to step through every gradient were prioritized.
- Large blocks of commented-out code are **intentionally left in place**. They show the evolution of the implementation and the debugging journey.
- The final CNN file contains a lot of timing instrumentation that was used to locate performance bottlenecks (especially the convolution backward pass).
- My original code is free of errors but python decided otherwise and decided to show multiple errors hence please quite literally ignore the **#type: ignore** comments.

---

## Bugs that were fixed

Only real runtime / logic errors were fixed. Every fix is marked in the source with a comment of the form:

```python
# FIXED: short explanation of what was wrong and what was changed
```

The most important fixes were:

1. Inconsistent return signature of `predict()` (sometimes returned a tuple, sometimes only the predictions array).
2. In the optimized training loop, `backprop` was changed to also return timing information, but several call sites still expected only the gradients dictionary.
3. A few fragile one-hot / shape handling cases with the test labels.

No working logic was removed and no commented-out sections were deleted.

---

Happy learning and debugging.  
May your gradients stay finite and your colons stay in the right place.

import math
import numpy as np
import torch
import einops


def torch_conv1d_DL(x: np.ndarray, w: np.ndarray, s: int = 1, p: int = 0) -> np.ndarray:
    """1D convolution using PyTorch."""
    assert x.ndim == 1 and w.ndim == 1
    x = torch.as_tensor(einops.rearrange(x, "... -> 1 1 ..."))
    w = torch.as_tensor(einops.rearrange(w, "... -> 1 1 ..."))
    return torch.nn.functional.conv1d(x, w, stride=s, padding=p).reshape(-1).numpy()


def my_conv1d_DL(x: np.ndarray, w: np.ndarray, s: int = 1, p: int = 0) -> np.ndarray:
    assert x.ndim == 1 and w.ndim == 1
    I, K = x.shape[-1], w.shape[-1]
    y = []
    for t in range(1 + math.floor((I - K + 2 * p) / s)):
        end = min(K - 1, I - 1 + p - s * t)
        start = max(0, p - s * t)
        y.append(sum(w[tau] * x[s * t - p + tau] for tau in range(start, end + 1)))
    return np.stack(y, axis=-1)


def torch_convtransposed1d_DL(
    v: np.ndarray,
    w: np.ndarray,
    s: int = 1,
    p: int = 0,
    pstar: int = 0,
) -> np.ndarray:
    """1D transposed convolution using PyTorch."""
    assert v.ndim == 1 and w.ndim == 1
    v = torch.as_tensor(einops.rearrange(v, "... -> 1 1 ..."))
    w = torch.as_tensor(einops.rearrange(w, "... -> 1 1 ..."))
    return (
        torch.nn.functional.conv_transpose1d(
            v, w, stride=s, padding=p, output_padding=pstar
        )
        .reshape(-1)
        .numpy()
    )


def my_convtransposed1d_DL(
    v: np.ndarray,
    w: np.ndarray,
    s: int = 1,
    p: int = 0,
    pstar: int = 0,
) -> np.ndarray:
    assert v.ndim == 1 and w.ndim == 1
    T, K = v.shape[-1], w.shape[-1]
    u = []
    for i in range(s * (T - 1) - 2 * p + K + pstar):
        end = min(T - 1, math.floor((i + p) / s))
        start = max(0, math.ceil((i + p - K + 1) / s))
        u.append(sum(v[t] * w[i + p - s * t] for t in range(start, end + 1)))
    return np.stack(u, axis=-1)


if __name__ == "__main__":
    # Test configurations
    conv1d_tests = [
        {"x": np.array([1.0, 2, 3, 4, 5]), "w": np.array([1.0, 1, 1]), "s": 1, "p": 0},
        {"x": np.array([1.0, 0, 1]), "w": np.array([1.0, 1]), "s": 1, "p": 0},
        {"x": np.random.randn(10), "w": np.random.randn(3), "s": 2, "p": 1},
        {"x": np.ones(7), "w": np.ones(3), "s": 3, "p": 2},
        {"x": np.zeros(5), "w": np.random.randn(4), "s": 1, "p": 3},
    ]

    print("Testing 1D Convolution:")
    for i, test in enumerate(conv1d_tests, 1):
        torch_result = torch_conv1d_DL(**test)
        my_result = my_conv1d_DL(**test)
        assert np.allclose(torch_result, my_result, atol=1e-12)
        print(f"Test {i} passed")

    convtransposed1d_tests = [
        {"v": np.array([1.0, 2, 3]), "w": np.array([1.0, 1]), "s": 1, "p": 0},
        {"v": np.random.randn(5), "w": np.random.randn(3), "s": 2, "p": 1},
        {"v": np.ones(4), "w": np.ones(2), "s": 3, "p": 2},
        {"v": np.zeros(3), "w": np.random.randn(4), "s": 1, "p": 1},
        {"v": np.random.randn(5), "w": np.random.randn(3), "s": 2, "p": 1, "pstar": 1},
    ]

    print("\nTesting 1D Transposed Convolution:")
    for i, test in enumerate(convtransposed1d_tests, 1):
        torch_result = torch_convtransposed1d_DL(**test)
        my_result = my_convtransposed1d_DL(**test)
        assert np.allclose(torch_result, my_result, atol=1e-12)
        print(f"Test {i} passed")

    print("\nAll tests passed successfully!")

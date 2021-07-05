import numpy as np

def numerical_derivative(f, x):
  delta_x = 1e-4
  grad = np.zeros_like(x)    # 0으로 초기화

  # x가 행렬 또는 벡터이므로 처음부터 끝까지 가리키기 위해 iterator 사용
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

  while not it.finished:
    idx = it.multi_index

    tmp_val = x[idx]
    x[idx] = float(tmp_val) + delta_x
    fx1 = f(x)

    x[idx] = tmp_val - delta_x
    fx2 = f(x)
    grad[idx] = (fx1 - fx2) / (2*delta_x)

    x[idx] = tmp_val
    it.iternext()

  return grad

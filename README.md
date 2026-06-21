# FDLS 5-Bus Power Flow Solver

Mini-project Python triển khai **FDLS / FDLF — Fast Decoupled Load-flow** cho hệ 5-bus MATPOWER/PJM.

Bộ dữ liệu mặc định là `case5`, một modified 5-bus, 5-generator case dựa trên PJM 5-bus system. Dữ liệu gốc được MATPOWER phân phối và dựa trên F. Li & R. Bo, “Small Test Systems for Power System Economic Studies”, IEEE PES General Meeting 2010.

## 1. Cài đặt

Yêu cầu Python >= 3.10.

```bash
# clone project của bạn
cd fdls-5bus

# tạo virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate      # Windows PowerShell

# cài package ở chế độ editable + dev tools
python -m pip install --upgrade pip
pip install -e .[dev]
```

## 2. Chạy thuật toán

```bash
fdls5bus
```

Hoặc:

```bash
PYTHONPATH=src python -m fdls5bus.cli
```

Kết quả tham khảo:

```text
Converged: True in 5 iterations

bus | Vm(pu)    | Va(deg)    | Pcalc(pu)   | Qcalc(pu)
----+-----------+------------+-------------+------------
  1 |  1.000000 |   3.273361 |    2.100000 |   0.307252
  2 |  0.989261 |  -0.759269 |   -3.000000 |  -0.986100
  3 |  1.000000 |  -0.492259 |    0.234900 |   0.960447
  4 |  1.000000 |   0.000000 |   -3.949728 |   0.526529
  5 |  1.000000 |   4.112031 |    4.665100 |  -0.382096
```

## 3. Chạy test

```bash
pytest -q
ruff check .
```

## 4. Cấu trúc dự án

```text
fdls-5bus/
├── .github/workflows/ci.yml
├── docs/FDLS_MATH.md
├── examples/run_case5.py
├── src/fdls5bus/
│   ├── __init__.py
│   ├── cli.py
│   ├── data.py
│   ├── solver.py
│   └── ybus.py
├── tests/test_fdls_case5.py
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

## 5. Ý tưởng thuật toán

Newton-Raphson đầy đủ dùng:

\[
\begin{bmatrix}
\Delta P\\
\Delta Q
\end{bmatrix}
=
\begin{bmatrix}
H & N\\
M & L
\end{bmatrix}
\begin{bmatrix}
\Delta\delta\\
\Delta |V|
\end{bmatrix}.
\]

FDLS dùng giả thiết lưới truyền tải cao áp:

\[
R\ll X,\quad |\delta_i-\delta_j|\ll1,\quad |V|\approx1.
\]

Do đó:

\[
N=\frac{\partial P}{\partial |V|}\approx0,
\quad
M=\frac{\partial Q}{\partial\delta}\approx0.
\]

Ta tách thành hai hệ tuyến tính:

\[
B'\Delta\delta = \frac{\Delta P}{|V|},
\]

\[
B''\Delta |V| = \frac{\Delta Q}{|V|}.
\]

Xem chứng minh chi tiết tại [`docs/FDLS_MATH.md`](docs/FDLS_MATH.md).

## 6. Ghi chú mô hình

- Bus type theo MATPOWER: `1 = PQ`, `2 = PV`, `3 = Slack`.
- Slack bus trong case này là bus 4.
- PV bus: bus 1, 3, 5.
- PQ bus: bus 2.
- Solver hiện tại không enforce giới hạn `Qmax/Qmin` của máy phát. Đây là chủ ý để giữ phần FDLS lõi rõ ràng.

## 7. Đẩy lên GitHub

```bash
git init
git add .
git commit -m "Initial FDLS 5-bus solver"

# tạo repo rỗng trên GitHub, ví dụ: https://github.com/<username>/fdls-5bus

git branch -M main
git remote add origin https://github.com/<username>/fdls-5bus.git
git push -u origin main
```

Nếu dùng GitHub CLI:

```bash
gh repo create fdls-5bus --public --source=. --remote=origin --push
```

## 8. License

MIT.

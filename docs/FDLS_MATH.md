# Chứng minh toán học thuật toán FDLS / FDLF

## 1. Bài toán trào lưu công suất AC

Với hệ thống `n` bus, điện áp phức tại bus `i` là

\[
V_i = |V_i| e^{j\delta_i}.
\]

Ma trận admittance nút là

\[
Y_{bus}=G+jB.
\]

Dòng điện bơm vào bus `i`:

\[
I_i = \sum_{k=1}^{n}Y_{ik}V_k.
\]

Công suất phức bơm vào bus `i`:

\[
S_i = P_i+jQ_i = V_i I_i^*.
\]

Thay `Y_{ik}=G_{ik}+jB_{ik}` và `V_i=|V_i|e^{j\delta_i}`, ta được:

\[
P_i = \sum_{k=1}^{n}|V_i||V_k|
\left(G_{ik}\cos\delta_{ik}+B_{ik}\sin\delta_{ik}\right),
\]

\[
Q_i = \sum_{k=1}^{n}|V_i||V_k|
\left(G_{ik}\sin\delta_{ik}-B_{ik}\cos\delta_{ik}\right),
\]

trong đó

\[
\delta_{ik}=\delta_i-\delta_k.
\]

Bài toán load-flow là tìm các biến chưa biết sao cho:

\[
\Delta P_i=P_i^{spec}-P_i^{calc}=0,
\]

\[
\Delta Q_i=Q_i^{spec}-Q_i^{calc}=0.
\]

Slack bus không có phương trình `P, Q`; PV bus có phương trình `P` và giữ `|V|`; PQ bus có cả `P` và `Q`.

---

## 2. Newton-Raphson đầy đủ

Véc-tơ ẩn được chia thành góc điện áp và biên độ điện áp:

\[
x=\begin{bmatrix}\delta\\ |V|\end{bmatrix}.
\]

Tuyến tính hóa Newton-Raphson:

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
\end{bmatrix},
\]

với:

\[
H=\frac{\partial P}{\partial \delta},\quad
N=\frac{\partial P}{\partial |V|},\quad
M=\frac{\partial Q}{\partial \delta},\quad
L=\frac{\partial Q}{\partial |V|}.
\]

Newton-Raphson chính xác hơn, nhưng phải cập nhật và phân rã Jacobian nhiều lần.

---

## 3. Cơ sở tách rời nhanh

Trong lưới truyền tải cao áp, thường có:

\[
R \ll X \quad \Rightarrow \quad G_{ik}\approx 0.
\]

Góc lệch giữa hai bus kề nhau cũng thường nhỏ:

\[
|\delta_i-\delta_k|\ll 1,
\]

nên:

\[
\sin\delta_{ik}\approx\delta_{ik},\quad
\cos\delta_{ik}\approx 1.
\]

Từ công thức công suất tác dụng:

\[
P_i \approx \sum_k |V_i||V_k|B_{ik}\sin\delta_{ik}.
\]

Vì `P` phụ thuộc mạnh vào `sin(delta_ik)`, nên `P` nhạy nhất với góc `delta`.

Từ công thức công suất phản kháng:

\[
Q_i \approx -\sum_k |V_i||V_k|B_{ik}\cos\delta_{ik}.
\]

Vì `cos(delta_ik) ≈ 1`, `Q` phụ thuộc mạnh vào biên độ điện áp `|V|`.

Do đó:

\[
\frac{\partial P}{\partial |V|}\approx 0,
\quad
\frac{\partial Q}{\partial \delta}\approx 0.
\]

Tức là:

\[
N\approx 0,\quad M\approx 0.
\]

Newton-Raphson trở thành hai hệ con:

\[
\Delta P \approx H\Delta\delta,
\]

\[
\Delta Q \approx L\Delta |V|.
\]

---

## 4. Từ Jacobian đến B' và B''

Với các giả thiết:

\[
G_{ik}\approx0,
\quad |V_i|\approx1,
\quad \cos\delta_{ik}\approx1,
\quad \sin\delta_{ik}\approx0,
\]

khối `H` xấp xỉ:

\[
H_{ik}\approx -|V_i||V_k|B_{ik}\quad (i\ne k).
\]

Vì `|V_i|≈|V_k|≈1`, đặt:

\[
B'_{ik}=-B_{ik}\quad (i\ne k).
\]

Sau chuẩn hóa theo điện áp, ta dùng:

\[
B'\Delta\delta = \frac{\Delta P}{|V|}.
\]

Tương tự, khối `L` dẫn đến:

\[
B''\Delta |V| = \frac{\Delta Q}{|V|}.
\]

Trong project này dùng biến thể XB phổ biến:

- `B'`: lập từ mạng chỉ có điện kháng nối tiếp `x`, bỏ `r`, bỏ charging, bỏ shunt, rồi lấy `-Im(Ybus)` và loại slack bus.
- `B''`: lấy `-Im(Ybus)` của mạng đầy đủ, chỉ giữ các PQ bus.

---

## 5. Quy trình lặp FDLS

Tại vòng lặp `k`:

1. Tính điện áp phức:

\[
V^{(k)} = |V|^{(k)}e^{j\delta^{(k)}}.
\]

2. Tính công suất:

\[
S^{calc}=V\odot (Y_{bus}V)^*.
\]

3. Tính sai số:

\[
\Delta P=P^{spec}-P^{calc},
\quad
\Delta Q=Q^{spec}-Q^{calc}.
\]

4. Cập nhật góc tại non-slack bus:

\[
\Delta\delta = (B')^{-1}\frac{\Delta P}{|V|}.
\]

\[
\delta^{(k+1)}=\delta^{(k)}+\Delta\delta.
\]

5. Cập nhật điện áp tại PQ bus:

\[
\Delta |V|=(B'')^{-1}\frac{\Delta Q}{|V|}.
\]

\[
|V|^{(k+1)}=|V|^{(k)}+\Delta |V|.
\]

6. Dừng khi:

\[
\max_i |\Delta P_i| < \varepsilon
\quad\text{và}\quad
\max_i |\Delta Q_i| < \varepsilon.
\]

---

## 6. Ý nghĩa vật lý

Với đường dây gần thuần cảm:

\[
P_{ij}\approx \frac{|V_i||V_j|}{X_{ij}}\sin(\delta_i-\delta_j),
\]

nên thay đổi góc pha làm thay đổi công suất tác dụng rất rõ.

Còn phản kháng gần với quan hệ biên độ điện áp:

\[
Q_i \sim \frac{|V_i|^2}{X},
\]

nên thay đổi `|V|` làm thay đổi `Q` mạnh.

FDLS vì vậy không phải là một mẹo tính toán tùy tiện; nó phản ánh cấu trúc vật lý của lưới truyền tải cao áp.

---

## 7. Giới hạn của chứng minh

Các xấp xỉ trên yếu đi khi:

- hệ có `R/X` lớn,
- điện áp lệch xa 1 pu,
- góc lệch lớn,
- hệ gần điểm sụp đổ điện áp,
- có nhiều thiết bị điều khiển phi tuyến hoặc giới hạn Q của máy phát bị kích hoạt.

Khi đó Newton-Raphson đầy đủ hoặc các biến thể robust hơn thường đáng tin cậy hơn.

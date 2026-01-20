import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class partner_estimation:
    def __init__(self, dt, B, k, d, Q, R, H):
        # 入力をPyTorchテンソルに変換 (デフォルトデバイスとdtypeに従う)
        self.dt = torch.tensor(dt, dtype=torch.float32, device=device)
        self.B = B.to(device=device, dtype=torch.float32) if isinstance(B, torch.Tensor) else torch.tensor(B, dtype=torch.float32, device=device)
        self.k = torch.tensor(k, dtype=torch.float32, device=device)
        self.d = torch.tensor(d, dtype=torch.float32, device=device)
        self.Q = Q.to(device=device, dtype=torch.float32) if isinstance(Q, torch.Tensor) else torch.tensor(Q, dtype=torch.float32, device=device)
        self.R = R.to(device=device, dtype=torch.float32) if isinstance(R, torch.Tensor) else torch.tensor(R, dtype=torch.float32, device=device)
        self.H = H.to(device=device, dtype=torch.float32) if isinstance(H, torch.Tensor) else torch.tensor(H, dtype=torch.float32, device=device)

        self.n = self.H.shape[1]  # dim of state vector
        self.m = self.H.shape[0]  # dim of observation vector

        # 状態と共分散行列をデフォルトデバイスとdtypeで初期化
        self.xhat = torch.zeros(self.n, dtype=torch.float32, device=device)
        self.P = torch.eye(self.n, dtype=torch.float32, device=device)

        # A行列はinternal_model内で計算されるため、ここでは初期化しない
        self.A = None

    def internal_model(self, x):
        # xがテンソルでない場合、またはdtype/deviceが異なる場合は変換
        if not isinstance(x, torch.Tensor) or x.dtype != self.xhat.dtype or x.device != self.xhat.device:
            x = torch.tensor(x, dtype=self.xhat.dtype, device=self.xhat.device)

        dt, B, k, d = self.dt, self.B, self.k, self.d
        # NumPyのreshape(-1, 1)に相当
        x = x.view(-1, 1)

        # 状態の分解 (テンソルインデックスを使用)
        px_1, vx_1, ax_1 = x[0, 0], x[1, 0], x[2, 0]
        py_1, vy_1, ay_1 = x[3, 0], x[4, 0], x[5, 0]
        px_2, vx_2, ax_2 = x[6, 0], x[7, 0], x[8, 0]
        py_2, vy_2, ay_2 = x[9, 0], x[10, 0], x[11, 0]
        px_t, vx_t, ax_t = x[12, 0], x[13, 0], x[14, 0]
        py_t, vy_t, ay_t = x[15, 0], x[16, 0], x[17, 0]
        Fx = x[18, 0]
        Fy = x[19, 0]
        Lx_1, Lx_2, Lx_3 = x[20, 0], x[21, 0], x[22, 0]
        Ly_1, Ly_2, Ly_3 = x[23, 0], x[24, 0], x[25, 0]

        # 次ステップの状態推定 (PyTorch演算を使用)
        px1_next = px_1 + vx_1 * dt + 0.5 * ax_1 * dt**2
        vx1_next = vx_1 + ax_1 * dt
        ax1_next = ax_1
        py1_next = py_1 + vy_1 * dt + 0.5 * ay_1 * dt**2
        vy1_next = vy_1 + ay_1 * dt
        ay1_next = ay_1

        px2_next = px_2 + vx_2 * dt + 0.5 * ax_2 * dt**2
        vx2_next = vx_2 + ax_2 * dt
        ax2_next = ax_2 + ((px_t - px_2) * Lx_1 + (vx_t - vx_2) * Lx_2 + (ax_t - ax_2) * Lx_3) * B[2, 0]
        py2_next = py_2 + vy_2 * dt + 0.5 * ay_2 * dt**2
        vy2_next = vy_2 + ay_2 * dt
        ay2_next = ay_2 + ((py_t - py_2) * Ly_1 + (vy_t - vy_2) * Ly_2 + (ay_t - ay_2) * Ly_3) * B[5, 0]

        pxt_next = px_t + vx_t * dt + 0.5 * ax_t * dt**2
        vxt_next = vx_t + ax_t * dt
        axt_next = ax_t

        pyt_next = py_t + vy_t * dt + 0.5 * ay_t * dt**2
        vyt_next = vy_t + ay_t * dt
        ayt_next = ay_t

        Fx_next = k * (px_2 - px_1) + d * (vx_2 - vx_1)
        Fy_next = k * (py_2 - py_1) + d * (vy_2 - vy_1)

        Lx1_next, Lx2_next, Lx3_next = Lx_1, Lx_2, Lx_3
        Ly1_next, Ly2_next, Ly3_next = Ly_1, Ly_2, Ly_3

        # 次ステップの状態ベクトルをテンソルとして構築
        x_next = torch.stack([
            px1_next, vx1_next, ax1_next,
            py1_next, vy1_next, ay1_next,
            px2_next, vx2_next, ax2_next,
            py2_next, vy2_next, ay2_next,
            pxt_next, vxt_next, axt_next,
            pyt_next, vyt_next, ayt_next,
            Fx_next, Fy_next,
            Lx1_next, Lx2_next, Lx3_next,
            Ly1_next, Ly2_next, Ly3_next
        ]).flatten()

        # 状態遷移行列 A をテンソルとして構築
        # A行列は内部状態(Lx, Lyなど)に依存するため、ステップごとに再構築が必要
        self.A = torch.eye(26, dtype=self.xhat.dtype, device=self.xhat.device)

        # 各行を設定
        self.A[0, :3] = torch.tensor([1, dt, 0.5 * dt**2], dtype=self.xhat.dtype, device=self.xhat.device)
        self.A[1, :3] = torch.tensor([0, 1, dt], dtype=self.xhat.dtype, device=self.xhat.device)

        self.A[3, 3:6] = torch.tensor([1, dt, 0.5 * dt**2], dtype=self.xhat.dtype, device=self.xhat.device)
        self.A[4, 3:6] = torch.tensor([0, 1, dt], dtype=self.xhat.dtype, device=self.xhat.device)

        self.A[6, 6:9] = torch.tensor([1, dt, 0.5 * dt**2], dtype=self.xhat.dtype, device=self.xhat.device)
        self.A[7, 6:9] = torch.tensor([0, 1, dt], dtype=self.xhat.dtype, device=self.xhat.device)

        self.A[9, 9:12] = torch.tensor([1, dt, 0.5 * dt**2], dtype=self.xhat.dtype, device=self.xhat.device)
        self.A[10, 9:12] = torch.tensor([0, 1, dt], dtype=self.xhat.dtype, device=self.xhat.device)

        self.A[12, 12:15] = torch.tensor([1, dt, 0.5 * dt**2], dtype=self.xhat.dtype, device=self.xhat.device)
        self.A[13, 12:15] = torch.tensor([0, 1, dt], dtype=self.xhat.dtype, device=self.xhat.device)

        self.A[15, 15:18] = torch.tensor([1, dt, 0.5 * dt**2], dtype=self.xhat.dtype, device=self.xhat.device)
        self.A[16, 15:18] = torch.tensor([0, 1, dt], dtype=self.xhat.dtype, device=self.xhat.device)

        # Fx_next = k * (px_2 - px_1) + d * (vx_2 - vx_1)
        self.A[18, 0] = -k
        self.A[18, 1] = -d
        self.A[18, 6] = k
        self.A[18, 7] = d

        # Fy_next = k * (py_2 - py_1) + d * (vy_2 - vy_1)
        self.A[19, 3] = -k
        self.A[19, 4] = -d
        self.A[19, 9] = k
        self.A[19, 10] = d
        
        self.A[8, 6] = -Lx_1 * B[2, 0]
        self.A[8, 7] = -Lx_2 * B[2, 0]
        self.A[8, 8] = 1 - Lx_3 * B[2, 0]
        self.A[8, 12] = Lx_1 * B[2, 0]
        self.A[8, 13] = Lx_2 * B[2, 0]
        self.A[8, 14] = Lx_3 * B[2, 0]
        self.A[8, 20] = (px_t - px_2) * B[2, 0]
        self.A[8, 21] = (vx_t - vx_2) * B[2, 0]
        self.A[8, 22] = (ax_t - ax_2) * B[2, 0]

        self.A[11, 9] = -Ly_1 * B[5, 0]
        self.A[11, 10] = -Ly_2 * B[5, 0]
        self.A[11, 11] = 1 - Ly_3 * B[5, 0]
        self.A[11, 15] = Ly_1 * B[5, 0]
        self.A[11, 16] = Ly_2 * B[5, 0]
        self.A[11, 17] = Ly_3 * B[5, 0]
        self.A[11, 23] = (py_t - py_2) * B[5, 0]
        self.A[11, 24] = (vy_t - vy_2) * B[5, 0]
        self.A[11, 25] = (ay_t - ay_2) * B[5, 0]

        return x_next

    def step(self, y):
        # yがテンソルでない場合、またはdtype/deviceが異なる場合は変換
        if isinstance(y, torch.Tensor):
            # If y is already a tensor, ensure it's on the correct device and detached.
            # We use .to() for device/dtype conversion and .detach() to remove from graph.
            y = y.to(dtype=self.xhat.dtype, device=self.xhat.device).detach()
        else:
            # If y is not a tensor (e.g., a numpy array or list), convert it.
            y = torch.tensor(y, dtype=self.xhat.dtype, device=self.xhat.device)

        x_odo = self.internal_model(self.xhat)
        # 共分散予測 (転置には .T を使用、行列積には @ を使用)
        P_odo = self.A @ self.P @ self.A.T + self.Q
        I = torch.eye(self.n, dtype=self.xhat.dtype, device=self.xhat.device)

        # 更新
        S = self.H @ P_odo @ self.H.T + self.R
        eps = 1e-6
        S = S + eps * torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
        # 逆行列計算 (torch.linalg.inv を使用)
        S_inv = torch.linalg.inv(S)
        K = P_odo @ self.H.T @ S_inv

        # innovation計算
        innovation = y - (self.H @ x_odo)

        self.xhat = x_odo + K @ innovation
        self.P = (I - K @ self.H) @ P_odo

        # 推定されたパートナーのターゲット位置をPyTorchテンソルとして返す
        # self.xhatは既に適切なデバイスとdtypeを持っている
        partner_target_pos = torch.stack([self.xhat[12], self.xhat[15]])

        return partner_target_pos

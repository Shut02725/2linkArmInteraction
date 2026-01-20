import torch
import torch.nn as nn
import numpy as np
import random

class InverseDynamicsLSTM(nn.Module):
    def __init__(self, input_dim=6, output_dim=2, hidden_dim=256, 
                 num_layers=2, dropout=0.1, seed=None):
        super().__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # output layer
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # init LSTM weights
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param.data, nonlinearity='relu')
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # init output weights
        for m in self.output_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden: Initial hidden state (optional)
        
        Returns:
            output: Tensor of shape (batch_size, output_dim)
            hidden: Final hidden state
        """
        batch_size, seq_len, _ = x.size()
        
        # 入力正規化
        x = self.input_norm(x)
        rnn_output, hidden = self.rnn(x, hidden)
        last_output = rnn_output[:, -1, :]  # (batch_size, hidden_dim)
        output = self.output_layers(last_output)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

    def predict_sequence(self, x, seq_len, hidden=None):
        """
        Multi-step prediction
        
        Args:
            x: Input tensor of shape (batch_size, input_seq_len, input_dim)
            seq_len: Number of future steps to predict
            hidden: Initial hidden state (optional)
        
        Returns:
            predictions: Tensor of shape (batch_size, seq_len, output_dim)
        """
        self.eval()
        predictions = []
        current_hidden = hidden
        
        with torch.no_grad():
            output, current_hidden = self.forward(x, current_hidden)
            predictions.append(output)
            
            # 自己回帰的な予測（outputをinputの一部として使用する場合）
            # この部分は具体的な用途に応じて調整が必要
            for _ in range(seq_len - 1):
                # 最後のタイムステップとして予測結果を使用
                # input_dimとoutput_dimが異なる場合は適切な変換が必要
                if self.input_dim == self.output_dim:
                    next_input = output.unsqueeze(1)  # (batch_size, 1, output_dim)
                else:
                    # 予測結果をinput_dimに合わせて変換（例：ゼロパディングなど）
                    if self.output_dim < self.input_dim:
                        padding = torch.zeros(output.size(0), 1, 
                                            self.input_dim - self.output_dim, 
                                            device=output.device)
                        next_input = torch.cat([output.unsqueeze(1), padding], dim=2)
                    else:
                        next_input = output[:, :self.input_dim].unsqueeze(1)
                
                output, current_hidden = self.forward(next_input, current_hidden)
                predictions.append(output)
        
        return torch.stack(predictions, dim=1)  # (batch_size, seq_len, output_dim)


# 使用例とテスト用の関数
def test_model():
    """モデルのテスト関数"""
    # パラメータ設定
    batch_size = 32
    seq_len = 10
    input_dim = 6
    output_dim = 2
    
    # 異なるRNNタイプでモデルを作成
    models = {
        'LSTM': InverseDynamicsLSTM(input_dim, output_dim, seed=42),
    }
    
    # テストデータ作成
    x = torch.randn(batch_size, seq_len, input_dim)
    
    for rnn_type, model in models.items():
        print(f"\n=== LSTM Model ===")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 順伝播テスト
        model.train()
        output, hidden = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
        # 隠れ状態の初期化テスト
        init_hidden = model.init_hidden(batch_size)
        if rnn_type == 'LSTM':
            print(f"Hidden state shape: h={init_hidden[0].shape}, c={init_hidden[1].shape}")
        else:
            print(f"Hidden state shape: {init_hidden.shape}")
        
        # 多段階予測テスト
        predictions = model.predict_sequence(x, seq_len=5)
        print(f"Multi-step predictions shape: {predictions.shape}")


if __name__ == "__main__":
    test_model()
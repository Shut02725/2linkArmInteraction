import torch
import torch.nn as nn
import random
import pickle
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from datetime import datetime
from pathlib import Path
from Env.Dyad_Pointmass_Env import DualTrackingEnv

class DataNormalizer:
    """
    データの正規化・標準化を管理するクラス
    注意: このクラスは学習時に使用されます。データ収集時には使用しません。
    """
    def __init__(self):
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
        self.fitted = False
    
    def fit(self, input_data, output_data):
        """正規化パラメータを計算（Many-to-One対応）"""
        # 入力データの統計量を計算（3次元の場合）
        if input_data.dim() == 3:  # (N, seq_len, input_dim)
            self.input_mean = torch.mean(input_data, dim=(0, 1))
            self.input_std = torch.std(input_data, dim=(0, 1))
        else:  # (N, input_dim)
            self.input_mean = torch.mean(input_data, dim=0)
            self.input_std = torch.std(input_data, dim=0)
        
        # 出力データの統計量を計算（Many-to-One: 2次元）
        if output_data.dim() == 2:  # (N, output_dim)
            self.output_mean = torch.mean(output_data, dim=0)
            self.output_std = torch.std(output_data, dim=0)
        else:  # (N, seq_len, output_dim) - 念のため
            self.output_mean = torch.mean(output_data, dim=(0, 1))
            self.output_std = torch.std(output_data, dim=(0, 1))
        
        # 標準偏差が0の場合は1に設定（除算エラー回避）
        self.input_std = torch.where(self.input_std == 0, torch.ones_like(self.input_std), self.input_std)
        self.output_std = torch.where(self.output_std == 0, torch.ones_like(self.output_std), self.output_std)
        
        self.fitted = True
        print(f"Normalizer fitted:")
        print(f"  Input mean: {self.input_mean}")
        print(f"  Input std: {self.input_std}")
        print(f"  Output mean: {self.output_mean}")
        print(f"  Output std: {self.output_std}")
    
    def save(self, filepath):
        """正規化パラメータを保存"""
        normalizer_data = {
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'output_mean': self.output_mean,
            'output_std': self.output_std,
            'fitted': self.fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(normalizer_data, f)
        print(f"Normalizer saved to {filepath}")
    
    def load(self, filepath):
        """正規化パラメータを読み込み"""
        with open(filepath, 'rb') as f:
            normalizer_data = pickle.load(f)
        
        self.input_mean = normalizer_data['input_mean']
        self.input_std = normalizer_data['input_std']
        self.output_mean = normalizer_data['output_mean']
        self.output_std = normalizer_data['output_std']
        self.fitted = normalizer_data['fitted']
        print(f"Normalizer loaded from {filepath}")
        return self
    
    def normalize_input(self, input_data):
        """入力データを正規化"""
        if not self.fitted:
            raise ValueError("Normalizer has not been fitted yet")
        return (input_data - self.input_mean) / self.input_std
    
    def denormalize_output(self, normalized_output):
        """正規化された出力データを元のスケールに戻す"""
        if not self.fitted:
            raise ValueError("Normalizer has not been fitted yet")
        return normalized_output * self.output_std + self.output_mean

class DataSaver:
    """
    データ収集とファイル保存を管理するクラス（改善版）
    正規化器は保存せず、データのみを保存します。
    """
    def __init__(self, save_dir="./Data_Training_ForwardModel/dyad_tracking_data_kp30_cd2"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.metadata_file = self.save_dir / "data_metadata.json"
        self.metadata = self.load_metadata()
    
    def load_metadata(self):
        """メタデータファイルを読み込み"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}
    
    def save_metadata(self):
        """メタデータファイルを保存"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def generate_filename(self, data_type, agent_type, episodes, timesteps, sequence_length):
        """ファイル名を生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{data_type}_{agent_type}_ep{episodes}_ts{timesteps}_seq{sequence_length}_{timestamp}"
    
    def save_datasets(self, dataset1, dataset2, data_type, agent_type, episodes, timesteps, 
                     sequence_length, stride=None):
        """
        データセットを保存（改善版）
        
        注意: 正規化器は保存しません。学習時に訓練データから新規作成します。
        
        Args:
            dataset1: Agent1のデータセット
            dataset2: Agent2のデータセット
            data_type: データタイプ
            agent_type: エージェントタイプ
            episodes: エピソード数
            timesteps: ステップ数
            sequence_length: シーケンス長
            stride: ストライド（オプション）
        """
        # ファイル名生成
        base_filename = self.generate_filename(data_type, agent_type, episodes, timesteps, sequence_length)
        
        # Agent1データの保存
        agent1_filename = f"{base_filename}_agent1.pkl"
        agent1_path = self.save_dir / agent1_filename
        with open(agent1_path, 'wb') as f:
            pickle.dump(dataset1, f)
        
        # Agent2データの保存
        agent2_filename = f"{base_filename}_agent2.pkl"
        agent2_path = self.save_dir / agent2_filename
        with open(agent2_path, 'wb') as f:
            pickle.dump(dataset2, f)
        
        # メタデータの更新
        metadata_key = base_filename
        metadata = {
            'data_type': data_type,
            'agent_type': agent_type,
            'episodes': episodes,
            'timesteps': timesteps,
            'sequence_length': sequence_length,
            'agent1_dataset_size': len(dataset1),
            'agent2_dataset_size': len(dataset2),
            'agent1_file': agent1_filename,
            'agent2_file': agent2_filename,
            'created_at': datetime.now().isoformat(),
            'input_dim': dataset1[0]['input'].shape[1] if len(dataset1) > 0 else None,
            'output_dim': dataset1[0]['output'].shape[0] if len(dataset1) > 0 else None  # Many-to-One形式
        }
        
        # stride情報をメタデータに追加
        if stride is not None:
            overlap_rate = (sequence_length - stride) / sequence_length * 100
            metadata['stride'] = stride
            metadata['overlap_rate'] = f"{overlap_rate:.1f}%"
        
        self.metadata[metadata_key] = metadata
        self.save_metadata()
        
        print(f"\n✓ データ保存完了:")
        print(f"  Agent1: {agent1_filename} (サイズ: {len(dataset1)})")
        print(f"  Agent2: {agent2_filename} (サイズ: {len(dataset2)})")
        if stride is not None:
            overlap_rate = (sequence_length - stride) / sequence_length * 100
            print(f"  Stride: {stride} (overlap: {overlap_rate:.1f}%)")
        
        return metadata_key
    
    def load_datasets(self, metadata_key):
        """
        データセットを読み込み（改善版）
        
        注意: 正規化器は読み込みません。学習時に訓練データから新規作成します。
        
        Args:
            metadata_key: メタデータキー
        
        Returns:
            dataset1, dataset2
        """
        if metadata_key not in self.metadata:
            raise FileNotFoundError(f"Metadata key not found: {metadata_key}")
        
        meta = self.metadata[metadata_key]
        
        # Agent1データの読み込み
        agent1_path = self.save_dir / meta['agent1_file']
        with open(agent1_path, 'rb') as f:
            dataset1 = pickle.load(f)
        
        # Agent2データの読み込み
        agent2_path = self.save_dir / meta['agent2_file']
        with open(agent2_path, 'rb') as f:
            dataset2 = pickle.load(f)
        
        print(f"\n✓ データ読み込み完了: {metadata_key}")
        print(f"  Agent1: {len(dataset1)} samples")
        print(f"  Agent2: {len(dataset2)} samples")
        
        return dataset1, dataset2
    
    def list_saved_data(self):
        """保存されているデータの一覧を表示"""
        if not self.metadata:
            print("保存されているデータがありません。")
            return
        
        print("\n=== 保存されているデータ一覧 ===")
        print(f"{'Key':<50} {'Type':<15} {'Episodes':<8} {'Timesteps':<9} {'Seq Len':<7} {'Stride':<7} {'Size':<10} {'Created':<16}")
        print("-" * 140)
        
        for key, info in self.metadata.items():
            created_date = datetime.fromisoformat(info['created_at']).strftime('%m-%d %H:%M')
            total_size = info['agent1_dataset_size'] + info['agent2_dataset_size']
            stride_info = str(info.get('stride', 'N/A'))
            print(f"{key:<50} {info['data_type']:<15} {info['episodes']:<8} "
                  f"{info['timesteps']:<9} {info['sequence_length']:<7} {stride_info:<7} {total_size:<10} {created_date:<16}")
    
    def delete_data(self, metadata_key):
        """データを削除"""
        if metadata_key not in self.metadata:
            print(f"データが見つかりません: {metadata_key}")
            return
        
        meta = self.metadata[metadata_key]
        
        # ファイルを削除
        agent1_path = self.save_dir / meta['agent1_file']
        agent2_path = self.save_dir / meta['agent2_file']
        
        if agent1_path.exists():
            agent1_path.unlink()
        if agent2_path.exists():
            agent2_path.unlink()
        
        # メタデータから削除
        del self.metadata[metadata_key]
        self.save_metadata()
        
        print(f"✓ データ削除完了: {metadata_key}")

def collect_dyad_tracking_data_strided(total_timesteps=100000, sequence_length=10, stride=5,
                                              auto_save=True, save_dir="./Data_Training_ForwardModel/dyad_tracking_data_kp60_cd7"):
    """
    Dual環境からStrided Window方式でシーケンシャルデータを収集する関数（Many-to-One形式）
    
    改善版: 正規化器は保存せず、データ統計情報のみを表示します。
    正規化器は学習時に訓練データから新規作成されます。
    
    Args:
        total_timesteps: 総ステップ数
        sequence_length: シーケンス長
        stride: ストライド（デフォルト: 5）
        auto_save: 自動保存するかどうか
        save_dir: 保存先ディレクトリ
    
    Returns:
        dataset1, dataset2, saved_key
    """
    dataset1 = []
    dataset2 = []
    
    overlap_rate = (sequence_length - stride) / sequence_length * 100
    print(f"\n{'='*80}")
    print(f"時系列データ収集開始 (Strided Window + Many-to-One):")
    print(f"{'='*80}")
    print(f"  総ステップ数: {total_timesteps:,}")
    print(f"  シーケンス長: {sequence_length}")
    print(f"  Stride: {stride} (overlap: {overlap_rate:.1f}%)")
    print(f"  予想サンプル数: 約 {(total_timesteps - sequence_length) // stride:,} samples")
    print(f"{'='*80}\n")

    if auto_save:
        saver = DataSaver(save_dir)

    env = DualTrackingEnv(dt=0.01)

    agent1_sequence_inputs = []
    agent2_sequence_inputs = []

    agent1_state = torch.cat([
        env.agent1_pos_noisy - env.target_pos_noisy,
        env.agent1_vel_noisy - env.target_vel_noisy,
        env.agent1_acc_noisy - env.target_acc_noisy
    ])

    agent2_state = torch.cat([
        env.agent2_pos_noisy - env.target_pos_noisy,
        env.agent2_vel_noisy - env.target_vel_noisy,
        env.agent2_acc_noisy - env.target_acc_noisy
    ])

    agent1_true_state = torch.cat([
        env.agent1_pos_error,
        env.agent1_vel_error,
        env.agent1_acc_error
    ])

    agent2_true_state = torch.cat([
        env.agent2_pos_error,
        env.agent2_vel_error,
        env.agent2_acc_error
    ])

    for step in range(total_timesteps):
        # 現在の状態を保存
        agent1_state_prev = agent1_state.clone()
        agent2_state_prev = agent2_state.clone()

        agent1_true_state_prev = agent1_true_state.clone()
        agent2_true_state_prev = agent2_true_state.clone()

        agent1_input = torch.cat([
            agent1_state_prev,
            env.agent1_control_noisy,
            env.F_interaction_noisy,
            env.agent1_self_obs
        ])

        agent2_input = torch.cat([
            agent2_state_prev,
            env.agent2_control_noisy,
            -env.F_interaction_noisy,
            env.agent2_self_obs
        ])

        agent1_true_state = torch.cat([
            env.agent1_pos_error,
            env.agent1_vel_error,
            env.agent1_acc_error
        ])

        agent2_true_state = torch.cat([
            env.agent2_pos_error,
            env.agent2_vel_error,
            env.agent2_acc_error
        ])
        
        # シーケンスバッファに追加
        agent1_sequence_inputs.append(agent1_input.clone())
        agent2_sequence_inputs.append(agent2_input.clone())

        env.step()

        agent1_state = torch.cat([
            env.agent1_pos_noisy - env.target_pos_noisy,
            env.agent1_vel_noisy - env.target_vel_noisy,
            env.agent1_acc_noisy - env.target_acc_noisy
        ])

        agent2_state = torch.cat([
            env.agent2_pos_noisy - env.target_pos_noisy,
            env.agent2_vel_noisy - env.target_vel_noisy,
            env.agent2_acc_noisy - env.target_acc_noisy
        ])

        agent1_true_state = torch.cat([
            env.agent1_pos_error,
            env.agent1_vel_error,
            env.agent1_acc_error
        ])

        agent2_true_state = torch.cat([
            env.agent2_pos_error,
            env.agent2_vel_error,
            env.agent2_acc_error
        ])

        if len(agent1_sequence_inputs) >= sequence_length:
            # 出力: 次の1ステップの状態変化量
            agent1_output = agent1_true_state - agent1_true_state_prev
            agent2_output = agent2_true_state - agent2_true_state_prev
            
            # Input: 直近のsequence_length個
            input_sequence_agent1 = torch.stack(agent1_sequence_inputs[-sequence_length:])
            input_sequence_agent2 = torch.stack(agent2_sequence_inputs[-sequence_length:])
            
            dataset1.append({
                'input': input_sequence_agent1,  # (sequence_length, input_dim)
                'output': agent1_output          # (output_dim,) ← Many-to-One
            })
            dataset2.append({
                'input': input_sequence_agent2,
                'output': agent2_output
            })
            
            # Strided Window: stride個だけ削除
            for _ in range(stride):
                if len(agent1_sequence_inputs) > 0:
                    agent1_sequence_inputs.pop(0)
                    agent2_sequence_inputs.pop(0)

        # プログレス表示
        if step % 10000 == 0 and step > 0:
            print(f"  Step: {step:,}/{total_timesteps:,} ({step/total_timesteps*100:.1f}%) - Samples: {len(dataset1):,}")
    
    print(f"\n{'='*80}")        
    print(f"データ収集完了！")
    print(f"{'='*80}")
    print(f"  Agent1データセット: {len(dataset1)} sequences")
    print(f"  Agent2データセット: {len(dataset2)} sequences")
    print(f"  理論的最大サンプル数: {(total_timesteps - sequence_length) // stride + 1:,}")
    
    # データ統計情報の表示（参考情報として）
    if len(dataset1) > 0 and len(dataset2) > 0:
        print(f"\n{'='*80}")
        print("=== データ統計情報（参考） ===")
        print("注意: 正規化器は学習時に訓練データから新規作成されます。")
        print(f"{'='*80}")
        
        # Agent1の統計
        agent1_inputs = torch.stack([data['input'] for data in dataset1])
        agent1_outputs = torch.stack([data['output'] for data in dataset1])
        
        input_mean = agent1_inputs.mean(dim=(0, 1))
        input_std = agent1_inputs.std(dim=(0, 1))
        output_mean = agent1_outputs.mean(dim=0)
        output_std = agent1_outputs.std(dim=0)
        
        print(f"\nAgent1:")
        print(f"  Input shape: {agent1_inputs.shape}")
        print(f"  Output shape: {agent1_outputs.shape}")
        print(f"  Input  - Mean: {input_mean.mean():.6f}, Std: {input_std.mean():.6f}")
        print(f"  Output - Mean: {output_mean.mean():.6f}, Std: {output_std.mean():.6f}")
        
        # Agent2の統計
        agent2_inputs = torch.stack([data['input'] for data in dataset2])
        agent2_outputs = torch.stack([data['output'] for data in dataset2])
        
        input_mean = agent2_inputs.mean(dim=(0, 1))
        input_std = agent2_inputs.std(dim=(0, 1))
        output_mean = agent2_outputs.mean(dim=0)
        output_std = agent2_outputs.std(dim=0)
        
        print(f"\nAgent2:")
        print(f"  Input shape: {agent2_inputs.shape}")
        print(f"  Output shape: {agent2_outputs.shape}")
        print(f"  Input  - Mean: {input_mean.mean():.6f}, Std: {input_std.mean():.6f}")
        print(f"  Output - Mean: {output_mean.mean():.6f}, Std: {output_std.mean():.6f}")
    
    # データの自動保存
    saved_key = None
    if auto_save and len(dataset1) > 0:
        saved_key = saver.save_datasets(
            dataset1, dataset2, 
            'dual_tracking', 'strided',
            0, total_timesteps, sequence_length,
            stride=stride
        )
    
    return dataset1, dataset2, saved_key

def collect_multiple_episodes_strided(num_episodes=100, timesteps_per_episode=4000, 
                                      sequence_length=10, stride=5,
                                      auto_save=True, save_dir="./Data_Training_ForwardModel/dyad_tracking_data_kp60_cd7"):
    """
    複数エピソードにわたってStrided Window方式でデータを収集する関数（改善版）
    
    改善版: 正規化器は保存せず、データ統計情報のみを表示します。
    
    Args:
        num_episodes: エピソード数
        timesteps_per_episode: エピソードあたりのステップ数
        sequence_length: シーケンス長
        stride: ストライド
        auto_save: 自動保存するかどうか
        save_dir: 保存先ディレクトリ
    
    Returns:
        all_dataset1, all_dataset2, saved_key
    """
    all_dataset1 = []
    all_dataset2 = []
    
    overlap_rate = (sequence_length - stride) / sequence_length * 100
    print(f"\n{'='*80}")
    print(f"複数エピソードデータ収集開始 (Strided Window):")
    print(f"{'='*80}")
    print(f"  エピソード数: {num_episodes}")
    print(f"  エピソードあたりのステップ数: {timesteps_per_episode}")
    print(f"  シーケンス長: {sequence_length}")
    print(f"  Stride: {stride} (overlap: {overlap_rate:.1f}%)")
    print(f"{'='*80}\n")
    
    if auto_save:
        saver = DataSaver(save_dir)
    
    for episode in range(num_episodes):
        print(f"\nエピソード {episode+1}/{num_episodes}")
        
        # 環境を新しく初期化
        env = DualTrackingEnv(dt=0.01)
        
        # 各エピソードでデータ収集
        episode_dataset1 = []
        episode_dataset2 = []
        
        agent1_sequence_inputs = []
        agent2_sequence_inputs = []

        agent1_state = torch.cat([
            env.agent1_pos_noisy - env.target_pos_noisy,
            env.agent1_vel_noisy - env.target_vel_noisy,
            env.agent1_acc_noisy - env.target_acc_noisy
        ])

        agent2_state = torch.cat([
            env.agent2_pos_noisy - env.target_pos_noisy,
            env.agent2_vel_noisy - env.target_vel_noisy,
            env.agent2_acc_noisy - env.target_acc_noisy
        ])

        agent1_true_state = torch.cat([
            env.agent1_pos_error,
            env.agent1_vel_error,
            env.agent1_acc_error
        ])

        agent2_true_state = torch.cat([
            env.agent2_pos_error,
            env.agent2_vel_error,
            env.agent2_acc_error
        ])

        for step in range(timesteps_per_episode):
            agent1_state_prev = agent1_state.clone()
            agent2_state_prev = agent2_state.clone()

            agent1_true_state_prev = agent1_true_state.clone()
            agent2_true_state_prev = agent2_true_state.clone()

            agent1_input = torch.cat([
                agent1_state_prev,
                env.agent1_control_noisy,
                env.F_interaction_noisy,
                env.agent1_self_obs
            ])

            agent2_input = torch.cat([
                agent2_state_prev,
                env.agent2_control_noisy,
                -env.F_interaction_noisy,
                env.agent2_self_obs
            ])

            agent1_sequence_inputs.append(agent1_input.clone())
            agent2_sequence_inputs.append(agent2_input.clone())

            env.step()

            agent1_state = torch.cat([
                env.agent1_pos_noisy - env.target_pos_noisy,
                env.agent1_vel_noisy - env.target_vel_noisy,
                env.agent1_acc_noisy - env.target_acc_noisy
            ])

            agent2_state = torch.cat([
                env.agent2_pos_noisy - env.target_pos_noisy,
                env.agent2_vel_noisy - env.target_vel_noisy,
                env.agent2_acc_noisy - env.target_acc_noisy
            ])

            agent1_true_state = torch.cat([
                env.agent1_pos_error,
                env.agent1_vel_error,
                env.agent1_acc_error
            ])

            agent2_true_state = torch.cat([
                env.agent2_pos_error,
                env.agent2_vel_error,
                env.agent2_acc_error
            ])

            if len(agent1_sequence_inputs) >= sequence_length:
                agent1_output = agent1_true_state - agent1_true_state_prev
                agent2_output = agent2_true_state - agent2_true_state_prev
                
                input_sequence_agent1 = torch.stack(agent1_sequence_inputs[-sequence_length:])
                input_sequence_agent2 = torch.stack(agent2_sequence_inputs[-sequence_length:])
                
                episode_dataset1.append({
                    'input': input_sequence_agent1,
                    'output': agent1_output
                })
                episode_dataset2.append({
                    'input': input_sequence_agent2,
                    'output': agent2_output
                })
                
                for _ in range(stride):
                    if len(agent1_sequence_inputs) > 0:
                        agent1_sequence_inputs.pop(0)
                        agent2_sequence_inputs.pop(0)
        
        all_dataset1.extend(episode_dataset1)
        all_dataset2.extend(episode_dataset2)
        
        print(f"  エピソード{episode+1}完了: Agent1={len(episode_dataset1)}, Agent2={len(episode_dataset2)} sequences")
    
    print(f"\n{'='*80}")
    print(f"全エピソード完了！")
    print(f"{'='*80}")
    print(f"  総データ数: Agent1={len(all_dataset1)}, Agent2={len(all_dataset2)} sequences")
    
    # データ統計情報の表示（参考情報として）
    if len(all_dataset1) > 0 and len(all_dataset2) > 0:
        print(f"\n{'='*80}")
        print("=== データ統計情報（参考） ===")
        print("注意: 正規化器は学習時に訓練データから新規作成されます。")
        print(f"{'='*80}")
        
        # Agent1の統計
        agent1_inputs = torch.stack([data['input'] for data in all_dataset1])
        agent1_outputs = torch.stack([data['output'] for data in all_dataset1])
        
        input_mean = agent1_inputs.mean(dim=(0, 1))
        input_std = agent1_inputs.std(dim=(0, 1))
        output_mean = agent1_outputs.mean(dim=0)
        output_std = agent1_outputs.std(dim=0)
        
        print(f"\nAgent1:")
        print(f"  Input shape: {agent1_inputs.shape}")
        print(f"  Output shape: {agent1_outputs.shape}")
        print(f"  Input  - Mean: {input_mean.mean():.6f}, Std: {input_std.mean():.6f}")
        print(f"  Output - Mean: {output_mean.mean():.6f}, Std: {output_std.mean():.6f}")
        
        # Agent2の統計
        agent2_inputs = torch.stack([data['input'] for data in all_dataset2])
        agent2_outputs = torch.stack([data['output'] for data in all_dataset2])
        
        input_mean = agent2_inputs.mean(dim=(0, 1))
        input_std = agent2_inputs.std(dim=(0, 1))
        output_mean = agent2_outputs.mean(dim=0)
        output_std = agent2_outputs.std(dim=0)
        
        print(f"\nAgent2:")
        print(f"  Input shape: {agent2_inputs.shape}")
        print(f"  Output shape: {agent2_outputs.shape}")
        print(f"  Input  - Mean: {input_mean.mean():.6f}, Std: {input_std.mean():.6f}")
        print(f"  Output - Mean: {output_mean.mean():.6f}, Std: {output_std.mean():.6f}")
    
    # 全データの自動保存
    saved_key = None
    if auto_save and len(all_dataset1) > 0:
        saved_key = saver.save_datasets(
            all_dataset1, all_dataset2,
            'dual_tracking', 'multi_episode_strided',
            num_episodes, num_episodes * timesteps_per_episode, sequence_length,
            stride=stride
        )
    
    return all_dataset1, all_dataset2, saved_key

def main():
    """
    メイン実行関数 - データ収集のデモ（改善版）
    """
    print("\n" + "="*80)
    print("Dual Tracking データ収集システム (Strided Window)")
    print("="*80)
    print("注意: 正規化器はデータ収集時には保存されません。")
    print("      学習時に訓練データから新規作成されます。")
    print("="*80)
    
    # DataSaverのインスタンス作成
    saver = DataSaver()
    
    # 既存データの一覧表示
    saver.list_saved_data()
    
    # データ収集の選択肢
    print("\nデータ収集オプション:")
    print("1. 単一セッション（100,000ステップ、stride=5）【推奨】")
    print("2. 複数エピソード（100エピソード × 4,000ステップ、stride=5）")
    print("3. カスタム設定")
    print("4. 既存データの読み込みテスト")
    print("5. 既存データの削除")
    print("6. 終了")
    
    while True:
        try:
            choice = input("\n選択してください (1-6): ").strip()
            
            if choice == '1':
                print("\n単一セッションデータ収集を開始...")
                dataset1, dataset2, key = collect_dyad_tracking_data_strided(
                    total_timesteps=100000,
                    sequence_length=10,
                    stride=5
                )
                print(f"\n保存キー: {key}")
                break
                
            elif choice == '2':
                print("\n複数エピソードデータ収集を開始...")
                dataset1, dataset2, key = collect_multiple_episodes_strided(
                    num_episodes=100,
                    timesteps_per_episode=4000,
                    sequence_length=10,
                    stride=5
                )
                print(f"\n保存キー: {key}")
                break
                
            elif choice == '3':
                print("\nカスタム設定:")
                timesteps = int(input("総ステップ数を入力: "))
                seq_len = int(input("シーケンス長を入力: "))
                stride_input = int(input("Stride値を入力 (推奨: 3-5): "))
                dataset1, dataset2, key = collect_dyad_tracking_data_strided(
                    total_timesteps=timesteps,
                    sequence_length=seq_len,
                    stride=stride_input
                )
                print(f"\n保存キー: {key}")
                break
                
            elif choice == '4':
                saver.list_saved_data()
                if saver.metadata:
                    key = input("\n読み込むデータのキーを入力: ").strip()
                    try:
                        dataset1, dataset2 = saver.load_datasets(key)
                        print(f"\n✓ 読み込み成功！")
                        if len(dataset1) > 0:
                            print(f"  サンプルデータ形状:")
                            print(f"    Input: {dataset1[0]['input'].shape}")
                            print(f"    Output: {dataset1[0]['output'].shape}")
                    except Exception as e:
                        print(f"❌ 読み込みエラー: {e}")
                else:
                    print("保存されているデータがありません。")
                
            elif choice == '5':
                saver.list_saved_data()
                if saver.metadata:
                    key = input("\n削除するデータのキーを入力: ").strip()
                    confirm = input(f"本当に削除しますか？ (y/n): ").strip().lower()
                    if confirm == 'y':
                        saver.delete_data(key)
                else:
                    print("保存されているデータがありません。")
                
            elif choice == '6':
                print("\n終了します。")
                break
                
            else:
                print("無効な選択です。1-6を入力してください。")
                
        except KeyboardInterrupt:
            print("\n\n処理が中断されました。")
            break
        except ValueError as e:
            print(f"入力エラー: {e}")
            print("数値を正しく入力してください。")

if __name__ == "__main__":
    main()
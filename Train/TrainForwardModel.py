import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import random
import json
from pathlib import Path
from datetime import datetime

from ForwardDynamics import ForwardDynamicsLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# データ処理クラス
# ==============================================================================

class DataNormalizer:
    """改善版: デバイス管理を明確化したデータ正規化クラス"""
    def __init__(self):
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
        self.fitted = False
    
    def fit(self, dataset):
        """データセットから統計量を計算"""
        print("データ統計量を計算中...")
        
        if not dataset:
            raise ValueError("データセットが空です")
        
        inputs = torch.stack([data['input'] for data in dataset])
        outputs = torch.stack([data['output'] for data in dataset])
        
        inputs_flat = inputs.view(-1, inputs.size(-1))
        outputs_flat = outputs.view(-1, outputs.size(-1))
        
        # GPU上で統計量を計算（高速化）
        self.input_mean = inputs_flat.mean(dim=0).to(device)
        self.input_std = inputs_flat.std(dim=0).to(device) + 1e-8
        self.output_mean = outputs_flat.mean(dim=0).to(device)
        self.output_std = outputs_flat.std(dim=0).to(device) + 1e-8
        
        self.fitted = True
        
        print("\n=== データ統計情報 ===")
        print(f"入力データ: 平均 {self.input_mean.mean().item():.4f}, 標準偏差 {self.input_std.mean().item():.4f}")
        print(f"出力データ: 平均 {self.output_mean.mean().item():.4f}, 標準偏差 {self.output_std.mean().item():.4f}")
        
        return self
    
    def normalize_input(self, input_data):
        """入力データを正規化"""
        if not self.fitted:
            raise ValueError("Normalizer has not been fitted yet")
        return (input_data.to(device) - self.input_mean) / self.input_std
    
    def normalize_output(self, output_data):
        """出力データを正規化"""
        if not self.fitted:
            raise ValueError("Normalizer has not been fitted yet")
        return (output_data.to(device) - self.output_mean) / self.output_std
    
    def denormalize_output(self, normalized_output):
        """正規化を元に戻す"""
        if not self.fitted:
            raise ValueError("Normalizer has not been fitted yet")
        return normalized_output * self.output_std + self.output_mean
    
    def save(self, filepath):
        """正規化パラメータを保存"""
        normalizer_data = {
            'input_mean': self.input_mean.cpu(),
            'input_std': self.input_std.cpu(),
            'output_mean': self.output_mean.cpu(),
            'output_std': self.output_std.cpu(),
            'fitted': self.fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(normalizer_data, f)
    
    def load(self, filepath):
        """正規化パラメータを読み込み"""
        with open(filepath, 'rb') as f:
            normalizer_data = pickle.load(f)
        self.input_mean = normalizer_data['input_mean'].to(device)
        self.input_std = normalizer_data['input_std'].to(device)
        self.output_mean = normalizer_data['output_mean'].to(device)
        self.output_std = normalizer_data['output_std'].to(device)
        self.fitted = normalizer_data['fitted']
        return self

def normalize_dataset(dataset, normalizer):
    """
    データセットを正規化（1回のみ実行し、CPUに保存）
    バッチ読み込み時にGPUに転送する方式
    """
    normalized_dataset = []
    for data in dataset:
        input_tensor = data['input'] if torch.is_tensor(data['input']) else \
                       torch.tensor(data['input'], dtype=torch.float32)
        output_tensor = data['output'] if torch.is_tensor(data['output']) else \
                        torch.tensor(data['output'], dtype=torch.float32)
        
        # GPU上で正規化
        normalized_input = normalizer.normalize_input(input_tensor)
        normalized_output = normalizer.normalize_output(output_tensor)
        
        # CPUに保存（メモリ効率化）
        normalized_dataset.append({
            'input': normalized_input.cpu(),
            'output': normalized_output.cpu()
        })
    
    return normalized_dataset

# ==============================================================================
# Optunaハイパーパラメータ読み込み関数
# ==============================================================================

def load_best_hyperparameters(optuna_results_dir="./optuna_results"):
    """
    Optunaの最適化結果からハイパーパラメータを読み込む
    
    Returns:
        hyperparams_dict: {agent_name: best_params} の辞書、またはNone
    """
    results_dir = Path(optuna_results_dir)
    
    if not results_dir.exists():
        print(f"警告: Optunaの結果ディレクトリ '{optuna_results_dir}' が見つかりません。")
        return None
    
    # JSONファイルを検索
    result_files = list(results_dir.glob("*_results.json"))
    
    if not result_files:
        print(f"警告: Optunaの結果ファイルが見つかりません。")
        return None
    
    print(f"\n{'='*80}")
    print("=== Optuna最適化結果の読み込み ===")
    print(f"{'='*80}")
    print(f"利用可能な最適化結果:")
    
    for i, f in enumerate(result_files):
        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"{i+1}. {f.stem}")
        print(f"   Agent: {data.get('agent_type', 'Unknown')}")
        print(f"   Best Val Loss: {data.get('best_value', 'N/A'):.6f}")
        print(f"   Date: {data.get('datetime', 'Unknown')[:19]}")
    
    choice = input(f"\n使用する最適化結果を選択 (1-{len(result_files)}), または 0 でデフォルト: ").strip()
    
    if choice == '0':
        return None
    
    try:
        file_idx = int(choice) - 1
        selected_file = result_files[file_idx]
        
        with open(selected_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        agent_type = result_data['agent_type']
        best_params = result_data['best_params']
        
        print(f"\n✓ 読み込んだハイパーパラメータ ({agent_type}):")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        # 複数のエージェントに適用するか確認
        apply_to_both = input("\nこのハイパーパラメータを両方のエージェント（Agent1, Agent2）に適用しますか？ (yes/no): ").strip().lower()
        
        if apply_to_both == 'yes':
            return {
                'Agent1': best_params,
                'Agent2': best_params
            }
        else:
            return {agent_type: best_params}
            
    except (ValueError, IndexError):
        print("無効な選択です。デフォルトのハイパーパラメータを使用します。")
        return None

def get_default_hyperparameters():
    """デフォルトのハイパーパラメータを返す"""
    return {
        'hidden_dim': 512,
        'num_layers': 2,
        'dropout': 0.0,
        'learning_rate': 1e-3,
        'batch_size': 32
    }

def merge_hyperparameters(agents_config, optuna_params):
    """
    既存の設定とOptunaのハイパーパラメータをマージ
    
    Args:
        agents_config: 既存のエージェント設定
        optuna_params: Optunaから読み込んだハイパーパラメータ
    
    Returns:
        merged_config: マージされた設定
    """
    merged_config = {}
    
    for agent_name, config in agents_config.items():
        # 基本設定をコピー
        merged_config[agent_name] = config.copy()
        
        # Optunaのパラメータがあれば上書き
        if optuna_params and agent_name in optuna_params:
            params = optuna_params[agent_name]
            merged_config[agent_name].update({
                'hidden_dim': params.get('hidden_dim', config['hidden_dim']),
                'num_layers': params.get('num_layers', config['num_layers']),
                'dropout': params.get('dropout', 0.0),
                'learning_rate': params.get('learning_rate', 1e-3),
                'batch_size': params.get('batch_size', 32)
            })
        else:
            # デフォルト値を追加
            default_params = get_default_hyperparameters()
            merged_config[agent_name].update({
                'learning_rate': default_params['learning_rate'],
                'batch_size': default_params['batch_size'],
                'dropout': default_params['dropout']
            })
    
    return merged_config

# ==============================================================================
# 学習関数
# ==============================================================================

def train_agent_model_optimal(model, dataset_training, dataset_validation, normalizer,
                             epochs=40, batch_size=32, lr=1e-3, agent_name="Agent",
                             prediction_mode="last_step", seed=42, verbose=True):
    """
    最適化された学習関数
    """
    # 1. シード設定
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 2. データの正規化（1回のみ）
    if verbose:
        print(f"\n{agent_name} (seed={seed}) 学習開始:")
        print(f"  訓練データ: {len(dataset_training)} samples")
        print(f"  検証データ: {len(dataset_validation)} samples")
        print(f"  学習率: {lr:.2e}")
        print(f"  バッチサイズ: {batch_size}")

    train_dataset = normalize_dataset(dataset_training, normalizer)
    val_dataset = normalize_dataset(dataset_validation, normalizer)
    
    # 3. オプティマイザとスケジューラの設定
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=15, verbose=False
    )
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # 4. 学習ループ
    for epoch in range(epochs):
        # === 訓練フェーズ ===
        model.train()
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
        num_batches = 0

        train_indices = torch.randperm(len(train_dataset))

        for i in range(0, len(train_dataset), batch_size):
            batch_indices = train_indices[i:i+batch_size]
            batch_data = [train_dataset[idx] for idx in batch_indices]
            batch_inputs = torch.stack([data['input'] for data in batch_data]).to(device)
            batch_outputs = torch.stack([data['output'] for data in batch_data]).to(device)
            
            optimizer.zero_grad()

            model_output = model(batch_inputs)
            predictions = model_output[0] if isinstance(model_output, tuple) else model_output

            if prediction_mode == "last_step":
                target_outputs = batch_outputs[:, -1, :] if batch_outputs.dim() == 3 else batch_outputs
                if predictions.dim() == 3:
                    predictions = predictions[:, -1, :]
            else:
                target_outputs = batch_outputs
                if predictions.dim() == 2 and batch_outputs.dim() == 3:
                    seq_len = batch_outputs.size(1)
                    predictions = predictions.unsqueeze(1).expand(-1, seq_len, -1)

            loss = criterion(predictions, target_outputs)
            
            mae = torch.mean(torch.abs(predictions - target_outputs))
            output_magnitude = torch.mean(torch.abs(target_outputs))
            accuracy = torch.clamp(1 - (mae / (output_magnitude + 1e-8)), 0, 1)
            
            l1_lambda = 1e-5
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            total_loss = loss + l1_lambda * l1_norm
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_train_accuracy += accuracy.item()
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        avg_train_accuracy = epoch_train_accuracy / num_batches
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        # === 検証フェーズ ===
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_accuracy = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for i in range(0, len(val_dataset), batch_size):
                batch_data = val_dataset[i:i+batch_size]
                batch_inputs = torch.stack([data['input'] for data in batch_data]).to(device)
                batch_outputs = torch.stack([data['output'] for data in batch_data]).to(device)
                
                model_output = model(batch_inputs)
                predictions = model_output[0] if isinstance(model_output, tuple) else model_output
                
                if prediction_mode == "last_step":
                    target_outputs = batch_outputs[:, -1, :] if batch_outputs.dim() == 3 else batch_outputs
                    if predictions.dim() == 3:
                        predictions = predictions[:, -1, :]
                else:
                    target_outputs = batch_outputs
                    if predictions.dim() == 2 and batch_outputs.dim() == 3:
                        seq_len = batch_outputs.size(1)
                        predictions = predictions.unsqueeze(1).expand(-1, seq_len, -1)
                
                loss = criterion(predictions, target_outputs)
                
                mae = torch.mean(torch.abs(predictions - target_outputs))
                output_magnitude = torch.mean(torch.abs(target_outputs))
                accuracy = torch.clamp(1 - (mae / (output_magnitude + 1e-8)), 0, 1)
                
                epoch_val_loss += loss.item()
                epoch_val_accuracy += accuracy.item()
                num_val_batches += 1
        
        avg_val_loss = epoch_val_loss / num_val_batches
        avg_val_accuracy = epoch_val_accuracy / num_val_batches
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)
        
        scheduler.step(avg_val_loss)
        
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}, "
                  f"Train Acc={avg_train_accuracy:.4f}, Val Acc={avg_val_accuracy:.4f}, "
                  f"LR={current_lr:.2e}")
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# ==============================================================================
# データ読み込み関数
# ==============================================================================

def load_dataset_from_directory(save_dir, agent_name, show_source_stats=False):
    """
    指定されたディレクトリからデータセットを読み込む
    
    Args:
        save_dir: データディレクトリ
        agent_name: エージェント名
        show_source_stats: 混合データセットのソース統計を表示するか
    """
    if not os.path.exists(save_dir):
        print(f"エラー: ディレクトリ '{save_dir}' が存在しません。")
        return None, None
    
    pkl_files = glob.glob(os.path.join(save_dir, "*.pkl"))
    
    if not pkl_files:
        print(f"エラー: '{save_dir}' にpklファイルが見つかりません。")
        return None, None
    
    print(f"\n=== {agent_name} 用データセット選択 ({save_dir}) ===")
    for i, file_path in enumerate(pkl_files):
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"{i+1}: {file_name} ({file_size:.2f} MB)")
    
    while True:
        try:
            choice = input(f"\n{agent_name} の学習に使用するファイルを選択してください (1-{len(pkl_files)}): ")
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(pkl_files):
                selected_file = pkl_files[choice_idx]
                break
            else:
                print(f"1から{len(pkl_files)}の間で選択してください。")
        except ValueError:
            print("数字を入力してください。")
    
    print(f"選択されたファイル: {os.path.basename(selected_file)}")
    
    try:
        with open(selected_file, 'rb') as f:
            dataset = pickle.load(f)
        
        if isinstance(dataset, dict):
            if 'training' in dataset and 'validation' in dataset:
                dataset_training = dataset['training']
                dataset_validation = dataset['validation']
            elif 'train' in dataset and 'val' in dataset:
                dataset_training = dataset['train']
                dataset_validation = dataset['val']
            else:
                keys = list(dataset.keys())
                print(f"\n訓練データと検証データのキーを指定してください:")
                for i, key in enumerate(keys):
                    print(f"{i+1}: {key}")
                
                train_choice = int(input("訓練データのキー番号: ")) - 1
                val_choice = int(input("検証データのキー番号: ")) - 1
                
                dataset_training = dataset[keys[train_choice]]
                dataset_validation = dataset[keys[val_choice]]
        
        elif isinstance(dataset, list):
            split_ratio = 0.8
            split_idx = int(len(dataset) * split_ratio)
            dataset_training = dataset[:split_idx]
            dataset_validation = dataset[split_idx:]
        
        else:
            print(f"エラー: 未対応のデータセット形式: {type(dataset)}")
            return None, None
        
        if len(dataset_training) > 0:
            sample = dataset_training[0]
            print(f"データ形式: input {sample['input'].shape}, output {sample['output'].shape}")
            
            # 混合データセットの場合、ソース統計を表示
            if show_source_stats and 'source' in sample:
                train_sources = {}
                for data in dataset_training:
                    source = data.get('source', 'unknown')
                    train_sources[source] = train_sources.get(source, 0) + 1
                
                val_sources = {}
                for data in dataset_validation:
                    source = data.get('source', 'unknown')
                    val_sources[source] = val_sources.get(source, 0) + 1
                
                print(f"\n混合データセット構成:")
                print(f"  訓練データ:")
                for source, count in train_sources.items():
                    ratio = count / len(dataset_training) * 100
                    print(f"    {source}: {count} samples ({ratio:.1f}%)")
                print(f"  検証データ:")
                for source, count in val_sources.items():
                    ratio = count / len(dataset_validation) * 100
                    print(f"    {source}: {count} samples ({ratio:.1f}%)")
        
        print(f"訓練データ: {len(dataset_training)} samples, 検証データ: {len(dataset_validation)} samples")
        
        return dataset_training, dataset_validation
        
    except Exception as e:
        print(f"エラー: データセットの読み込みに失敗しました: {e}")
        return None, None

# ==============================================================================
# モデル・結果保存関数
# ==============================================================================

def save_model_and_results(model, normalizer, train_losses, val_losses, train_acc, val_acc,
                          agent_name, save_dir, seed, env_type="solo"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join(save_dir, f"{env_type}_{agent_name}_model_seed{seed}")
    os.makedirs(model_save_dir, exist_ok=True)

    model_path = os.path.join(model_save_dir, f"{agent_name}_model_final.pth")
    torch.save(model.state_dict(), model_path)
    
    normalizer_path = os.path.join(model_save_dir, f"{agent_name}_normalizer.pkl")
    normalizer.save(normalizer_path)
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_acc,
        'val_accuracies': val_acc,
        'agent_name': agent_name,
        'env_type': env_type,
        'seed': seed,
        'timestamp': timestamp,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'final_train_acc': train_acc[-1] if train_acc else None,
        'final_val_acc': val_acc[-1] if val_acc else None,
    }
    
    history_path = os.path.join(model_save_dir, f"{agent_name}_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    print(f"✓ モデル保存: {model_save_dir}")
    
    return model_save_dir

# ==============================================================================
# プロット関数（Solo vs dyad比較版）
# ==============================================================================

def plot_solo_vs_dyad_comparison(solo_results, dyad_results, save_dir, epochs, 
                                  save_filename):
    """
    Solo環境と混合データセットの学習結果を比較プロット
    Learning CurveとAccuracyを別々のファイルに保存
    
    Args:
        solo_results: Solo環境の結果
        dyad_results: 混合データセットの結果
        save_dir: 保存先ディレクトリ
        epochs: エポック数
        save_filename: 保存ファイル名（ベース名、拡張子は自動付与）
    """
    plt.style.use('default')
    
    # 色設定
    colors_solo = {'Agent1': '#2E86AB', 'Agent2': '#A23B72'}
    colors_dyad = {'Agent1': '#F18F01', 'Agent2': '#C73E1D'}
    
    epoch_list = np.arange(1, epochs + 1)
    
    # ============================================================
    # 1. Learning Curve（損失）のプロット
    # ============================================================
    fig_loss, axes_loss = plt.subplots(2, 1, figsize=(16, 12))
    
    # 全体の損失範囲を計算（Agent1とAgent2で統一するため）
    all_loss_values = []
    for agent_name in ['Agent1', 'Agent2']:
        for results in [solo_results, dyad_results]:
            if agent_name in results and results[agent_name]['train_losses']:
                data = results[agent_name]
                all_loss_values.extend(np.array(data['train_losses']).flatten())
                all_loss_values.extend(np.array(data['val_losses']).flatten())
    
    if all_loss_values:
        loss_min = np.min(all_loss_values)
        loss_max = np.max(all_loss_values)
    else:
        loss_min, loss_max = 1e-6, 1.0
    
    for idx, agent_name in enumerate(['Agent1', 'Agent2']):
        ax_loss = axes_loss[idx]
        
        # Solo環境の損失
        if agent_name in solo_results and solo_results[agent_name]['train_losses']:
            solo_data = solo_results[agent_name]
            train_losses_array = np.array(solo_data['train_losses'])
            val_losses_array = np.array(solo_data['val_losses'])
            
            train_mean = np.mean(train_losses_array, axis=0) if train_losses_array.ndim > 1 else train_losses_array
            train_std = np.std(train_losses_array, axis=0) if train_losses_array.ndim > 1 else np.zeros_like(train_mean)
            val_mean = np.mean(val_losses_array, axis=0) if val_losses_array.ndim > 1 else val_losses_array
            val_std = np.std(val_losses_array, axis=0) if val_losses_array.ndim > 1 else np.zeros_like(val_mean)
            
            ax_loss.plot(epoch_list, train_mean, color=colors_solo[agent_name], linestyle='-',
                        label=f'Solo Train', linewidth=2.5, alpha=0.9)
            ax_loss.fill_between(epoch_list, train_mean - train_std, train_mean + train_std,
                                color=colors_solo[agent_name], alpha=0.2)
            ax_loss.plot(epoch_list, val_mean, color=colors_solo[agent_name], linestyle='--',
                        label=f'Solo Val', linewidth=2.5, alpha=0.9)
            ax_loss.fill_between(epoch_list, val_mean - val_std, val_mean + val_std,
                                color=colors_solo[agent_name], alpha=0.15)
        
        # 混合データセットの損失
        if agent_name in dyad_results and dyad_results[agent_name]['train_losses']:
            dyad_data = dyad_results[agent_name]
            train_losses_array = np.array(dyad_data['train_losses'])
            val_losses_array = np.array(dyad_data['val_losses'])
            
            train_mean = np.mean(train_losses_array, axis=0) if train_losses_array.ndim > 1 else train_losses_array
            train_std = np.std(train_losses_array, axis=0) if train_losses_array.ndim > 1 else np.zeros_like(train_mean)
            val_mean = np.mean(val_losses_array, axis=0) if val_losses_array.ndim > 1 else val_losses_array
            val_std = np.std(val_losses_array, axis=0) if val_losses_array.ndim > 1 else np.zeros_like(val_mean)
            
            ax_loss.plot(epoch_list, train_mean, color=colors_dyad[agent_name], linestyle='-',
                        label=f'dyad Train', linewidth=2.5, alpha=0.9)
            ax_loss.fill_between(epoch_list, train_mean - train_std, train_mean + train_std,
                                color=colors_dyad[agent_name], alpha=0.2)
            ax_loss.plot(epoch_list, val_mean, color=colors_dyad[agent_name], linestyle='--',
                        label=f'dyad Val', linewidth=2.5, alpha=0.9)
            ax_loss.fill_between(epoch_list, val_mean - val_std, val_mean + val_std,
                                color=colors_dyad[agent_name], alpha=0.15)
        
        ax_loss.set_xlabel('Epoch', fontsize=18)
        ax_loss.set_ylabel('Loss (MSE)', fontsize=18)
        ax_loss.set_title(f'{agent_name} - Loss (Solo vs dyad)', fontsize=20, fontweight='bold')
        ax_loss.legend(loc='best', fontsize=16)
        ax_loss.grid(True, alpha=0.3, linestyle='--')
        ax_loss.set_yscale('log')
        ax_loss.set_ylim(loss_min * 0.8, loss_max * 1.2)  # 統一された軸範囲
        ax_loss.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    
    # Learning Curveを保存
    base_name = save_filename.replace('.png', '')
    loss_plot_path = os.path.join(save_dir, f"{base_name}_loss.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Learning Curve保存: {loss_plot_path}")
    plt.close()
    
    # ============================================================
    # 2. Accuracy（精度）のプロット
    # ============================================================
    fig_acc, axes_acc = plt.subplots(2, 1, figsize=(16, 12))
    
    # 全体の精度範囲を計算（Agent1とAgent2で統一、かつデータに寄せる）
    all_acc_values = []
    for agent_name in ['Agent1', 'Agent2']:
        for results in [solo_results, dyad_results]:
            if agent_name in results and results[agent_name]['train_acc']:
                data = results[agent_name]
                all_acc_values.extend(np.array(data['train_acc']).flatten())
                all_acc_values.extend(np.array(data['val_acc']).flatten())
    
    if all_acc_values:
        acc_min = np.min(all_acc_values)
        acc_max = np.max(all_acc_values)
        # データに寄せた軸範囲（マージンを少なく）
        acc_margin = (acc_max - acc_min) * 0.1
        acc_ylim_min = max(0, acc_min - acc_margin)
        acc_ylim_max = min(1, acc_max + acc_margin)
    else:
        acc_ylim_min, acc_ylim_max = 0, 1
    
    for idx, agent_name in enumerate(['Agent1', 'Agent2']):
        ax_acc = axes_acc[idx]
        
        # Solo環境の精度
        if agent_name in solo_results and solo_results[agent_name]['train_acc']:
            solo_data = solo_results[agent_name]
            train_acc_array = np.array(solo_data['train_acc'])
            val_acc_array = np.array(solo_data['val_acc'])
            
            train_acc_mean = np.mean(train_acc_array, axis=0) if train_acc_array.ndim > 1 else train_acc_array
            train_acc_std = np.std(train_acc_array, axis=0) if train_acc_array.ndim > 1 else np.zeros_like(train_acc_mean)
            val_acc_mean = np.mean(val_acc_array, axis=0) if val_acc_array.ndim > 1 else val_acc_array
            val_acc_std = np.std(val_acc_array, axis=0) if val_acc_array.ndim > 1 else np.zeros_like(val_acc_mean)
            
            ax_acc.plot(epoch_list, train_acc_mean, color=colors_solo[agent_name], linestyle='-',
                       label=f'Solo Train', linewidth=2.5, alpha=0.9)
            ax_acc.fill_between(epoch_list, train_acc_mean - train_acc_std, train_acc_mean + train_acc_std,
                               color=colors_solo[agent_name], alpha=0.2)
            ax_acc.plot(epoch_list, val_acc_mean, color=colors_solo[agent_name], linestyle='--',
                       label=f'Solo Val', linewidth=2.5, alpha=0.9)
            ax_acc.fill_between(epoch_list, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std,
                               color=colors_solo[agent_name], alpha=0.15)
        
        # 混合データセットの精度
        if agent_name in dyad_results and dyad_results[agent_name]['train_acc']:
            dyad_data = dyad_results[agent_name]
            train_acc_array = np.array(dyad_data['train_acc'])
            val_acc_array = np.array(dyad_data['val_acc'])
            
            train_acc_mean = np.mean(train_acc_array, axis=0) if train_acc_array.ndim > 1 else train_acc_array
            train_acc_std = np.std(train_acc_array, axis=0) if train_acc_array.ndim > 1 else np.zeros_like(train_acc_mean)
            val_acc_mean = np.mean(val_acc_array, axis=0) if val_acc_array.ndim > 1 else val_acc_array
            val_acc_std = np.std(val_acc_array, axis=0) if val_acc_array.ndim > 1 else np.zeros_like(val_acc_mean)
            
            ax_acc.plot(epoch_list, train_acc_mean, color=colors_dyad[agent_name], linestyle='-',
                       label=f'dyad Train', linewidth=2.5, alpha=0.9)
            ax_acc.fill_between(epoch_list, train_acc_mean - train_acc_std, train_acc_mean + train_acc_std,
                               color=colors_dyad[agent_name], alpha=0.2)
            ax_acc.plot(epoch_list, val_acc_mean, color=colors_dyad[agent_name], linestyle='--',
                       label=f'dyad Val', linewidth=2.5, alpha=0.9)
            ax_acc.fill_between(epoch_list, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std,
                               color=colors_dyad[agent_name], alpha=0.15)
        
        ax_acc.set_xlabel('Epoch', fontsize=18)
        ax_acc.set_ylabel('Accuracy (1 - Relative MAE)', fontsize=18)
        ax_acc.set_title(f'{agent_name} - Accuracy (Solo vs dyad)', fontsize=20, fontweight='bold')
        ax_acc.legend(loc='best', fontsize=16)
        ax_acc.grid(True, alpha=0.3, linestyle='--')
        ax_acc.set_ylim(acc_ylim_min, acc_ylim_max)  # データに寄せた統一された軸範囲
        ax_acc.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    
    # Accuracyを保存
    acc_plot_path = os.path.join(save_dir, f"{base_name}_accuracy.png")
    plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Accuracy保存: {acc_plot_path}")
    plt.close()

# ==============================================================================
# メイン関数
# ==============================================================================

def main():
    """
    Solo環境と混合データセット（Optuna最適化済み）を比較する学習実験
    """
    
    # ============================================================
    # 学習設定
    # ============================================================
    # データディレクトリ（Solo環境と混合データセット）
    solo_data_dir = "./Data_Training_ForwardModel/solo_tracking_data_0120"
    dyad_data_dir = "./Data_Training_ForwardModel/dyad_tracking_data_kp60_cd3"

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"Experiment_Results_Solo_vs_dyad_kp60_cd3"
    os.makedirs(results_dir, exist_ok=True)
    
    # Agent設定（デフォルト値 - Optunaで上書きされる可能性あり）
    agents_config = {
        "Agent1": {
            "input_dim": 14,
            "output_dim": 6,
            "hidden_dim": 512,  # デフォルト
            "num_layers": 2      # デフォルト
        },
        "Agent2": {
            "input_dim": 14,
            "output_dim": 6,
            "hidden_dim": 512,
            "num_layers": 2
        }
    }
    
    # ============================================================
    # Optunaの最適化結果を読み込み
    # ============================================================
    optuna_params = load_best_hyperparameters()
    
    # ハイパーパラメータをマージ
    agents_config = merge_hyperparameters(agents_config, optuna_params)
    
    # ハイパーパラメータ
    epochs = 40
    num_seeds = 1
    seeds = [42]
    
    # ============================================================
    # グローバルシード設定
    # ============================================================
    global_seed = 42
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(global_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"{'='*100}")
    print(f"Solo vs 混合データセット 比較学習（Optuna最適化済み）")
    print(f"{'='*100}")
    print(f"実行タイムスタンプ: {run_timestamp}")
    print(f"結果保存先: {results_dir}")
    print(f"Solo データディレクトリ: {solo_data_dir}")
    print(f"混合データディレクトリ: {dyad_data_dir}")
    print(f"エージェント数: {len(agents_config)}")
    print(f"各エージェントのシード数: {num_seeds}")
    print(f"環境数: 2 (Solo, dyad)")
    print(f"総学習回数: {len(agents_config) * num_seeds * 2}")
    print(f"エポック数: {epochs}")
    print(f"使用デバイス: {device}")
    
    # エージェント設定を表示
    print(f"\n=== エージェント設定（Optuna最適化済み） ===")
    for agent_name, config in agents_config.items():
        print(f"{agent_name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print(f"{'='*100}\n")
    
    # ============================================================
    # データセット読み込み（Solo環境と混合データセット）
    # ============================================================
    environments = {
        'solo': {'dir': solo_data_dir, 'datasets': {}, 'show_source': False},
        'dyad': {'dir': dyad_data_dir, 'datasets': {}, 'show_source': True}  # 混合データセットはソース統計を表示
    }
    
    for env_type, env_config in environments.items():
        print(f"\n{'='*100}")
        print(f"=== {env_type.upper()} {'混合データセット' if env_type == 'dyad' else '環境'} のデータセット読み込み ===")
        print(f"{'='*100}")
        
        for agent_name in agents_config.keys():
            train_data, val_data = load_dataset_from_directory(
                env_config['dir'], 
                agent_name,
                show_source_stats=env_config['show_source']
            )
            
            if train_data is None or val_data is None:
                print(f"❌ {env_type.upper()} - {agent_name} のデータセット読み込みに失敗しました。")
                return
            
            env_config['datasets'][agent_name] = {
                'train': train_data,
                'val': val_data
            }
            print(f"✓ {env_type.upper()} - {agent_name} データセット読み込み完了")
    
    # ============================================================
    # 学習実行（Solo環境と混合データセット）
    # ============================================================
    all_results = {
        'solo': {},
        'dyad': {}
    }
    
    for env_type in ['solo', 'dyad']:
        env_datasets = environments[env_type]['datasets']
        
        print(f"\n{'='*100}")
        print(f"=== {env_type.upper()} {'混合データセット' if env_type == 'dyad' else '環境'} の学習開始 ===")
        print(f"{'='*100}")
        
        for agent_name, config in agents_config.items():
            print(f"\n{'='*100}")
            print(f"=== {env_type.upper()} - {agent_name} の学習開始 ===")
            print(f"{'='*100}")
            
            # 正規化器を作成
            normalizer = DataNormalizer()
            normalizer.fit(env_datasets[agent_name]['train'])
            
            # 結果格納用
            agent_results = {
                'seeds': seeds[:num_seeds],
                'train_losses': [],
                'val_losses': [],
                'train_acc': [],
                'val_acc': []
            }
            
            # 各シードで学習
            for seed_idx, seed in enumerate(seeds[:num_seeds]):
                print(f"\n{'─'*100}")
                print(f"--- {env_type.upper()} - {agent_name} Seed {seed} ({seed_idx+1}/{num_seeds}) ---")
                print(f"{'─'*100}")
                
                # モデル初期化（Optuna最適化済みハイパーパラメータを使用）
                model = ForwardDynamicsLSTM(
                    input_dim=config['input_dim'],
                    output_dim=config['output_dim'],
                    hidden_dim=config['hidden_dim'],
                    num_layers=config['num_layers'],
                    dropout=config.get('dropout', 0.0),
                    seed=seed
                ).to(device)
                
                # 学習実行（Optunaのハイパーパラメータを使用）
                train_losses, val_losses, train_acc, val_acc = train_agent_model_optimal(
                    model,
                    env_datasets[agent_name]['train'],
                    env_datasets[agent_name]['val'],
                    normalizer,
                    epochs=epochs,
                    batch_size=config.get('batch_size', 32),
                    lr=config.get('learning_rate', 1e-3),
                    agent_name=f"{env_type.upper()}-{agent_name}",
                    prediction_mode="last_step",
                    seed=seed,
                    verbose=(seed_idx == 0)  # 最初のシードのみ詳細表示
                )
                
                # モデルと結果を保存
                model_save_dir = save_model_and_results(
                    model, normalizer, train_losses, val_losses, train_acc, val_acc,
                    agent_name, results_dir, seed, env_type=env_type
                )
                
                # 結果を追加
                agent_results['train_losses'].append(train_losses)
                agent_results['val_losses'].append(val_losses)
                agent_results['train_acc'].append(train_acc)
                agent_results['val_acc'].append(val_acc)
                
                print(f"\n✓ Seed {seed} 完了:")
                print(f"  最終訓練損失: {train_losses[-1]:.6f}")
                print(f"  最終検証損失: {val_losses[-1]:.6f}")
                print(f"  最終訓練精度: {train_acc[-1]:.4f}")
                print(f"  最終検証精度: {val_acc[-1]:.4f}")
            
            all_results[env_type][agent_name] = agent_results
            print(f"\n✓ {env_type.upper()} - {agent_name} の全シード学習が完了しました。")
    
    # ============================================================
    # 結果の可視化（Solo vs dyad比較）
    # ============================================================
    print(f"\n{'='*100}")
    print("=== Solo vs dyad 比較プロット作成 ===")
    print(f"{'='*100}")
    
    plot_solo_vs_dyad_comparison(
        all_results['solo'],
        all_results['dyad'],
        results_dir,
        epochs,
        save_filename="solo_vs_dyad_comparison.png"
    )
    
    # ============================================================
    # 完了メッセージ
    # ============================================================
    print(f"\n{'='*100}")
    print(f"✓ 全ての学習が完了しました！")
    print(f"{'='*100}")
    print(f"実行タイムスタンプ: {run_timestamp}")
    print(f"結果保存先: {results_dir}")
    print(f"学習済みモデル数: {len(agents_config) * num_seeds * 2}")
    print(f"比較プロット: {os.path.join(results_dir, 'solo_vs_dyad_comparison.png')}")
    print(f"{'='*100}\n")

if __name__ == "__main__":
    main()
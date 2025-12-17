"""
NFL Player Trajectory Prediction Model
=====================================
Based on the specified architecture:
- Play-level model handling up to 22 players
- Dynamic + Static feature encoding
- Transformer-based inter-player interaction
- GaussianNLLLoss with auxiliary losses
"""

import os
import math
import copy
import warnings
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import RAdam
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

pd.options.display.max_columns = None
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data config
    MAX_PLAYERS = 22
    INPUT_FRAMES = 20  # 20 frames before the pass
    OUTPUT_FRAMES = 48  # Predict 48 frames ahead
    
    # Feature dimensions
    DYNAMIC_FEATURES = 10  # x, y, sin(o), cos(o), sin(dir)*s, cos(dir)*s, dx_ball, dy_ball, dx_recv, dy_recv
    STATIC_FEATURES = 12   # role(4) + num_pred_frames(1) + num_input_frames(1) + passer_xy(2) + ball_land_xy(2) + final_xy(2)
    
    # Model config
    DYNAMIC_OUT_DIM = 64  # Per-feature after depthwise conv
    DYNAMIC_TOTAL_DIM = 640  # 10 features * 64 = 640
    STATIC_OUT_DIM = 64
    PLAYER_HIDDEN_DIM = 256
    
    # Transformer config
    NUM_TRANSFORMER_LAYERS = 3
    NUM_HEADS = 8
    DROPOUT = 0.1
    
    # Decoder config
    DECODER_HIDDEN = 1536
    DECODER_CHANNELS = 32
    
    # Training config
    N_FOLDS = 2
    N_REPEATS = 1
    EPOCHS = 210
    EVAL_EVERY = 6
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EMA_DECAY = 0.9995
    
    # Augmentation config
    ROTATION_PROB = 0.5
    MAX_EARLY_FRAMES = 20  # Up to 20 frames earlier for prediction
    FLIP_PROB = 0.5
    
    # Player roles (4 categories)
    PLAYER_ROLES = ['Defensive Coverage', 'Other Route Runner', 'Passer',
       'Targeted Receiver']
    
    # Outlier plays to remove
    OUTLIER_PLAYS = {
        (2023091100, 3167),  # too long
        (2023122100, 1450),  # too long
        (2023091001, 3216),  # no passer
        (2023112606, 4180),  # no passer
        (2023121009, 3594),  # no passer
    }

config = Config()

# %%
# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outlier plays from dataframe using vectorized operations"""
    if df.empty:
        return df
    # Create tuple column for fast lookup
    play_keys = list(zip(df['game_id'], df['play_id']))
    mask = [key not in config.OUTLIER_PLAYS for key in play_keys]
    return df[mask].reset_index(drop=True)


def normalize_play_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Rotate plays where play_direction == 'left' by 180 degrees"""
    df = df.copy()
    left_mask = df['play_direction'] == 'left'
    
    if left_mask.any():
        # Rotate coordinates (flip x around field center, flip y around field center)
        df.loc[left_mask, 'x'] = 120 - df.loc[left_mask, 'x']
        df.loc[left_mask, 'y'] = 53.3 - df.loc[left_mask, 'y']
        
        # Rotate angles by 180 degrees
        for angle_col in ['dir', 'o']:
            if angle_col in df.columns:
                df.loc[left_mask, angle_col] = (df.loc[left_mask, angle_col] + 180) % 360
        
        # Flip ball landing coordinates
        if 'ball_land_x' in df.columns:
            df.loc[left_mask, 'ball_land_x'] = 120 - df.loc[left_mask, 'ball_land_x']
        if 'ball_land_y' in df.columns:
            df.loc[left_mask, 'ball_land_y'] = 53.3 - df.loc[left_mask, 'ball_land_y']
    
    return df


def load_all_data(weeks: range = range(1, 19)) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load all weekly data files and combine"""
    input_dfs = []
    output_dfs = []
    
    for week in weeks:
        week_str = f"{week:02d}"
        input_file = f"train/input_2023_w{week_str}.csv"
        output_file = f"train/output_2023_w{week_str}.csv"
        
        try:
            input_df = pd.read_csv(input_file)
            output_df = pd.read_csv(output_file)
            input_dfs.append(input_df)
            output_dfs.append(output_df)
            print(f"Week {week:2d}: input={len(input_df):,} rows, output={len(output_df):,} rows")
        except FileNotFoundError as e:
            print(f"Week {week:2d}: File not found")
    
    train_data = pd.concat(input_dfs, ignore_index=True) if input_dfs else pd.DataFrame()
    output_data = pd.concat(output_dfs, ignore_index=True) if output_dfs else pd.DataFrame()
    
    print(f"\nTotal before filtering: input={len(train_data):,} rows, output={len(output_data):,} rows")
    
    # Filter outliers
    train_data = filter_outliers(train_data)
    output_data = filter_outliers(output_data)
    
    # Normalize play direction
    train_data = normalize_play_direction(train_data)
    output_data_with_dir = output_data.merge(
        train_data[['game_id', 'play_id', 'nfl_id', 'play_direction']].drop_duplicates(),
        on=['game_id', 'play_id', 'nfl_id'],
        how='left'
    )
    output_data_normalized = normalize_play_direction(output_data_with_dir)
    output_data = output_data_normalized[['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y']]
    
    print(f"Total after filtering: input={len(train_data):,} rows, output={len(output_data):,} rows")
    
    return train_data, output_data

# Load data (just week 1 for initial testing)
print("Loading week 1 data for testing...")
train_data, output_data = load_all_data(weeks=range(1, 2))

# %%
# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureExtractor:
    """
    Extract dynamic and static features for the model.
    
    Dynamic Features (10-dim per frame, 20 frames):
        - x, y coordinates
        - sin(o), cos(o) - orientation
        - sin(dir) * s, cos(dir) * s - velocity direction weighted by speed
        - x - ball_land_x, y - ball_land_y - relative to ball landing
        - x - receiver_x, y - receiver_y - relative to receiver (target)
    
    Static Features (12-dim):
        - One-hot player_role (4 dims)
        - Number of prediction frames (1)
        - Number of input frames used (1)
        - Passer's final-frame coordinates (2)
        - Ball landing coordinates (2)
        - Player's final-frame coordinates (2)
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.role_to_idx = {role: i for i, role in enumerate(config.PLAYER_ROLES)}
    
    def _deg_to_rad(self, degrees: np.ndarray) -> np.ndarray:
        """Convert degrees to radians"""
        return np.deg2rad(degrees)
    
    def extract_dynamic_features(
        self,
        player_frames: pd.DataFrame,
        ball_land_x: float,
        ball_land_y: float,
        receiver_x: float,
        receiver_y: float,
        num_input_frames: int = None
    ) -> np.ndarray:
        """
        Extract dynamic features for a single player.
        
        Args:
            player_frames: DataFrame with player tracking data (sorted by frame_id)
            ball_land_x, ball_land_y: Ball landing coordinates
            receiver_x, receiver_y: Receiver final position
            num_input_frames: Number of frames to use (for earlier frame augmentation)
        
        Returns:
            np.ndarray of shape (10, INPUT_FRAMES)
        """
        if num_input_frames is None:
            num_input_frames = self.config.INPUT_FRAMES
        
        # Ensure at least 1 frame
        num_input_frames = max(1, num_input_frames)
        
        # Get last N frames (or pad if fewer available)
        frames = player_frames.tail(num_input_frames).copy()
        n_frames = len(frames)
        
        # Handle empty frames case
        if n_frames == 0:
            frames = player_frames.tail(1).copy()
            n_frames = len(frames)
        
        # Initialize output array
        features = np.zeros((10, self.config.INPUT_FRAMES), dtype=np.float32)
        
        # Extract raw values
        x = frames['x'].values.astype(np.float32)
        y = frames['y'].values.astype(np.float32)
        s = frames['s'].values.astype(np.float32)  # speed
        o = self._deg_to_rad(frames['o'].values.astype(np.float32))  # orientation in radians
        dir_rad = self._deg_to_rad(frames['dir'].values.astype(np.float32))  # direction in radians
        
        # Compute features
        sin_o = np.sin(o)
        cos_o = np.cos(o)
        sin_dir_s = np.sin(dir_rad) * s
        cos_dir_s = np.cos(dir_rad) * s
        dx_ball = x - ball_land_x
        dy_ball = y - ball_land_y
        dx_recv = x - receiver_x
        dy_recv = y - receiver_y
        
        # Stack features: (10, n_frames)
        frame_features = np.stack([
            x, y,
            sin_o, cos_o,
            sin_dir_s, cos_dir_s,
            dx_ball, dy_ball,
            dx_recv, dy_recv
        ], axis=0)
        
        # Pad to INPUT_FRAMES (pad at the beginning with first frame values)
        if n_frames < self.config.INPUT_FRAMES:
            pad_width = self.config.INPUT_FRAMES - n_frames
            # Pad with the first available value (earliest frame)
            pad_values = frame_features[:, :1]  # Shape (10, 1)
            padding = np.repeat(pad_values, pad_width, axis=1)
            features = np.concatenate([padding, frame_features], axis=1)
        else:
            features = frame_features[:, -self.config.INPUT_FRAMES:]
        
        return features
    
    def extract_static_features(
        self,
        player_role: str,
        num_output_frames: int,
        num_input_frames: int,
        passer_x: float,
        passer_y: float,
        ball_land_x: float,
        ball_land_y: float,
        player_final_x: float,
        player_final_y: float
    ) -> np.ndarray:
        """
        Extract static features for a single player.
        
        Returns:
            np.ndarray of shape (12,)
        """
        features = np.zeros(12, dtype=np.float32)
        
        # One-hot encoding of player role (4 dims)
        role_idx = self.role_to_idx.get(player_role, 0)
        features[role_idx] = 1.0
        
        # Number of prediction frames (normalized)
        features[4] = num_output_frames / self.config.OUTPUT_FRAMES
        
        # Number of input frames used (for earlier frame augmentation)
        features[5] = num_input_frames / self.config.INPUT_FRAMES
        
        # Passer's final-frame coordinates (normalized)
        features[6] = passer_x / 120.0
        features[7] = passer_y / 53.3
        
        # Ball landing coordinates (normalized)
        features[8] = ball_land_x / 120.0
        features[9] = ball_land_y / 53.3
        
        # Player's final-frame coordinates (normalized)
        features[10] = player_final_x / 120.0
        features[11] = player_final_y / 53.3
        
        return features


# Test feature extractor
extractor = FeatureExtractor(config)
print("Feature extractor initialized.")
print(f"Role mapping: {extractor.role_to_idx}")

# %%
# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class DataAugmentation:
    """
    Data augmentation techniques:
    1. Rotation augmentation (50% chance, uniform 0-360°)
    2. Predicting from earlier frames (up to 20 frames earlier)
    3. Vertical flip (flip along X-axis)
    """
    
    def __init__(self, config: Config):
        self.config = config
    
    def rotate_play(
        self,
        dynamic_features: np.ndarray,  # (players, 10, frames)
        static_features: np.ndarray,   # (players, 12)
        targets: np.ndarray,           # (players, 2, output_frames)
        center_x: float,
        center_y: float,
        angle_deg: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Rotate the entire play around the mean player position.
        
        Args:
            dynamic_features: (players, 10, frames) - features
            static_features: (players, 12) - static features
            targets: (players, 2, output_frames) - target displacements
            center_x, center_y: Center of rotation
            angle_deg: Rotation angle in degrees (random if None)
        """
        if angle_deg is None:
            angle_deg = np.random.uniform(0, 360)
        
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Rotation matrix
        def rotate_point(x, y):
            x_centered = x - center_x
            y_centered = y - center_y
            x_rot = x_centered * cos_a - y_centered * sin_a + center_x
            y_rot = x_centered * sin_a + y_centered * cos_a + center_y
            return x_rot, y_rot
        
        # Rotate dynamic features
        dynamic_out = dynamic_features.copy()
        # x, y at indices 0, 1
        x_vals = dynamic_out[:, 0, :]
        y_vals = dynamic_out[:, 1, :]
        x_rot, y_rot = rotate_point(x_vals, y_vals)
        dynamic_out[:, 0, :] = x_rot
        dynamic_out[:, 1, :] = y_rot
        
        # Rotate orientation: indices 2, 3 are sin(o), cos(o)
        old_sin_o = dynamic_out[:, 2, :]
        old_cos_o = dynamic_out[:, 3, :]
        # Rotate angle by adding rotation angle
        dynamic_out[:, 2, :] = old_sin_o * cos_a + old_cos_o * sin_a
        dynamic_out[:, 3, :] = old_cos_o * cos_a - old_sin_o * sin_a
        
        # Rotate velocity direction: indices 4, 5 are sin(dir)*s, cos(dir)*s
        old_sin_dir_s = dynamic_out[:, 4, :]
        old_cos_dir_s = dynamic_out[:, 5, :]
        dynamic_out[:, 4, :] = old_sin_dir_s * cos_a + old_cos_dir_s * sin_a
        dynamic_out[:, 5, :] = old_cos_dir_s * cos_a - old_sin_dir_s * sin_a
        
        # Relative positions (indices 6-9) need to be rotated too
        # dx_ball, dy_ball at 6, 7
        dx_ball = dynamic_out[:, 6, :]
        dy_ball = dynamic_out[:, 7, :]
        dynamic_out[:, 6, :] = dx_ball * cos_a - dy_ball * sin_a
        dynamic_out[:, 7, :] = dx_ball * sin_a + dy_ball * cos_a
        
        # dx_recv, dy_recv at 8, 9
        dx_recv = dynamic_out[:, 8, :]
        dy_recv = dynamic_out[:, 9, :]
        dynamic_out[:, 8, :] = dx_recv * cos_a - dy_recv * sin_a
        dynamic_out[:, 9, :] = dx_recv * sin_a + dy_recv * cos_a
        
        # Rotate static features (coordinates)
        static_out = static_features.copy()
        # Passer coords at 6, 7
        px, py = static_out[:, 6] * 120, static_out[:, 7] * 53.3
        px_rot, py_rot = rotate_point(px, py)
        static_out[:, 6] = px_rot / 120.0
        static_out[:, 7] = py_rot / 53.3
        
        # Ball land coords at 8, 9
        bx, by = static_out[:, 8] * 120, static_out[:, 9] * 53.3
        bx_rot, by_rot = rotate_point(bx, by)
        static_out[:, 8] = bx_rot / 120.0
        static_out[:, 9] = by_rot / 53.3
        
        # Player final coords at 10, 11
        fx, fy = static_out[:, 10] * 120, static_out[:, 11] * 53.3
        fx_rot, fy_rot = rotate_point(fx, fy)
        static_out[:, 10] = fx_rot / 120.0
        static_out[:, 11] = fy_rot / 53.3
        
        # Rotate targets (displacements)
        targets_out = targets.copy()
        dx = targets_out[:, 0, :]
        dy = targets_out[:, 1, :]
        targets_out[:, 0, :] = dx * cos_a - dy * sin_a
        targets_out[:, 1, :] = dx * sin_a + dy * cos_a
        
        return dynamic_out, static_out, targets_out
    
    def vertical_flip(
        self,
        dynamic_features: np.ndarray,
        static_features: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Flip the play along the X-axis (vertical flip).
        y -> 53.3 - y, and negate y-components of direction/velocity
        """
        dynamic_out = dynamic_features.copy()
        static_out = static_features.copy()
        targets_out = targets.copy()
        
        # Flip y coordinates (index 1)
        dynamic_out[:, 1, :] = 53.3 - dynamic_out[:, 1, :]
        
        # Negate y-components
        dynamic_out[:, 2, :] = -dynamic_out[:, 2, :]  # sin(o) -> -sin(o)
        dynamic_out[:, 4, :] = -dynamic_out[:, 4, :]  # sin(dir)*s
        dynamic_out[:, 7, :] = -dynamic_out[:, 7, :]  # dy_ball
        dynamic_out[:, 9, :] = -dynamic_out[:, 9, :]  # dy_recv
        
        # Flip static y coordinates
        static_out[:, 7] = 1.0 - static_out[:, 7]   # passer_y
        static_out[:, 9] = 1.0 - static_out[:, 9]   # ball_land_y
        static_out[:, 11] = 1.0 - static_out[:, 11] # player_final_y
        
        # Negate target dy
        targets_out[:, 1, :] = -targets_out[:, 1, :]
        
        return dynamic_out, static_out, targets_out
    
    def apply_earlier_frame_augmentation(
        self,
        dynamic_features: np.ndarray,  # (players, 10, INPUT_FRAMES)
        targets: np.ndarray,           # (players, 2, OUTPUT_FRAMES)
        num_earlier_frames: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Shift the prediction start point earlier by num_earlier_frames.
        This means we use fewer input frames and need to predict more output frames.
        
        Returns:
            dynamic_features: Modified dynamic features
            targets: Extended targets
            actual_output_frames: New number of output frames
        """
        if num_earlier_frames <= 0:
            return dynamic_features, targets, targets.shape[2]
        
        # Remove last N input frames
        dynamic_out = dynamic_features.copy()
        # Shift features: the last frame becomes earlier
        # We need to pad the end with the new "last frame"
        if num_earlier_frames < self.config.INPUT_FRAMES:
            # Use frames up to (INPUT_FRAMES - num_earlier_frames)
            effective_frames = self.config.INPUT_FRAMES - num_earlier_frames
            # Pad from the front to maintain INPUT_FRAMES
            dynamic_out = np.concatenate([
                np.repeat(dynamic_features[:, :, :1], num_earlier_frames, axis=2),
                dynamic_features[:, :, :effective_frames]
            ], axis=2)
        
        # Extend targets with the "transition" frames
        # In reality, we'd need the actual positions for those frames
        # For simplicity, we'll just note this extends the prediction window
        actual_output_frames = targets.shape[2] + num_earlier_frames
        
        return dynamic_out, targets, actual_output_frames


augmenter = DataAugmentation(config)
print("Data augmentation initialized.")

# %%
# ============================================================================
# DATASET CLASS
# ============================================================================

"""
Modified NFLPlayDataset with Random Frame Offset Augmentation
==============================================================
Changes the frame offset randomly for each epoch/sample during training.
"""

class NFLPlayDataset(Dataset):
    """
    Dataset for NFL play trajectory prediction.
    Each sample is a single play with up to 22 players.
    
    AUGMENTATION: Random frame offset applied per sample during training.
    This means each epoch will see different temporal windows of the same play.
    """
    
    def __init__(
        self,
        input_data: pd.DataFrame,
        output_data: pd.DataFrame,
        config: Config,
        training: bool = True,
        fixed_offset: int = None  # For validation/test: use fixed offset
    ):
        self.config = config
        self.training = training
        self.fixed_offset = fixed_offset  # None for training (random), set for val/test
        self.extractor = FeatureExtractor(config)
        self.augmenter = DataAugmentation(config)
        
        # Group by play
        self.plays = self._prepare_plays(input_data, output_data)
        print(f"Prepared {len(self.plays)} plays (training={training})")
    
    def _prepare_plays(
        self,
        input_data: pd.DataFrame,
        output_data: pd.DataFrame
    ) -> List[Dict]:
        """Prepare play-level data structures"""
        plays = []
        
        # Get unique plays
        play_keys = input_data[['game_id', 'play_id']].drop_duplicates()
        
        for _, row in tqdm(play_keys.iterrows(), total=len(play_keys), desc="Preparing plays"):
            game_id, play_id = row['game_id'], row['play_id']
            
            # Get all players in this play
            play_input = input_data[
                (input_data['game_id'] == game_id) & 
                (input_data['play_id'] == play_id)
            ].copy()
            
            play_output = output_data[
                (output_data['game_id'] == game_id) & 
                (output_data['play_id'] == play_id)
            ].copy()
            
            if play_input.empty:
                continue
            
            # Get play-level info
            ball_land_x = play_input['ball_land_x'].iloc[0]
            ball_land_y = play_input['ball_land_y'].iloc[0]
            num_output_frames = int(play_input['num_frames_output'].iloc[0])
            
            # Find passer and receiver
            passer_data = play_input[play_input['player_role'] == 'Passer']
            receiver_data = play_input[play_input['player_role'] == 'Targeted Receiver']
            
            if passer_data.empty:
                continue  # Skip plays without passer
            
            passer_final = passer_data.sort_values('frame_id').iloc[-1]
            passer_x, passer_y = passer_final['x'], passer_final['y']
            
            # Get receiver final position (use ball landing if no receiver)
            if not receiver_data.empty:
                receiver_final = receiver_data.sort_values('frame_id').iloc[-1]
                receiver_x, receiver_y = receiver_final['x'], receiver_final['y']
            else:
                receiver_x, receiver_y = ball_land_x, ball_land_y
            
            # Get unique players
            player_ids = play_input['nfl_id'].unique()
            
            plays.append({
                'game_id': game_id,
                'play_id': play_id,
                'player_ids': player_ids,
                'input_data': play_input,
                'output_data': play_output,
                'ball_land_x': ball_land_x,
                'ball_land_y': ball_land_y,
                'passer_x': passer_x,
                'passer_y': passer_y,
                'receiver_x': receiver_x,
                'receiver_y': receiver_y,
                'num_output_frames': num_output_frames
            })
        
        return plays
    
    def __len__(self):
        return len(self.plays)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        play = self.plays[idx]
        
        # Initialize tensors for all players
        n_players = len(play['player_ids'])
        
        # Dynamic features: (max_players, 10, INPUT_FRAMES)
        dynamic_features = np.zeros(
            (self.config.MAX_PLAYERS, 10, self.config.INPUT_FRAMES),
            dtype=np.float32
        )
        
        # Static features: (max_players, 12)
        static_features = np.zeros(
            (self.config.MAX_PLAYERS, 12),
            dtype=np.float32
        )
        
        # Targets: (max_players, 2, OUTPUT_FRAMES) - displacement from final input frame
        targets = np.zeros(
            (self.config.MAX_PLAYERS, 2, self.config.OUTPUT_FRAMES),
            dtype=np.float32
        )
        
        # Player mask: which players are valid
        player_mask = np.zeros(self.config.MAX_PLAYERS, dtype=np.float32)
        
        # Final positions (for converting displacement to absolute)
        final_positions = np.zeros((self.config.MAX_PLAYERS, 2), dtype=np.float32)
        
        # ============================================================
        # KEY CHANGE: Determine frame offset for this sample
        # ============================================================
        if self.training and self.fixed_offset is None:
            # Random offset for each sample during training
            # Can shift back up to MAX_EARLY_FRAMES
            max_offset = min(self.config.MAX_EARLY_FRAMES, self.config.INPUT_FRAMES - 1)
            num_earlier_frames = np.random.randint(0, max_offset + 1)
        else:
            # Fixed offset for validation/test (default: 0, no shift)
            num_earlier_frames = self.fixed_offset if self.fixed_offset is not None else 0
        
        # Calculate effective number of input frames to use
        effective_input_frames = self.config.INPUT_FRAMES - num_earlier_frames
        
        for i, nfl_id in enumerate(play['player_ids'][:self.config.MAX_PLAYERS]):
            player_input = play['input_data'][
                play['input_data']['nfl_id'] == nfl_id
            ].sort_values('frame_id')
            
            player_output = play['output_data'][
                play['output_data']['nfl_id'] == nfl_id
            ].sort_values('frame_id')
            
            if player_input.empty:
                continue
            
            # Get player info
            player_role = player_input['player_role'].iloc[0]
            
            # ============================================================
            # Apply frame offset: use frames up to (last - num_earlier_frames)
            # ============================================================
            if num_earlier_frames > 0 and len(player_input) > num_earlier_frames:
                # Remove the last N frames from input
                player_input_shifted = player_input.iloc[:-num_earlier_frames]
                
                # The "final frame" is now earlier
                if len(player_input_shifted) > 0:
                    final_frame = player_input_shifted.iloc[-1]
                else:
                    # Fallback if we removed too much
                    final_frame = player_input.iloc[-1]
                    player_input_shifted = player_input
            else:
                # No shift or not enough frames
                player_input_shifted = player_input
                final_frame = player_input.iloc[-1]
            
            final_x, final_y = final_frame['x'], final_frame['y']
            
            # Extract dynamic features with effective frame count
            dynamic_features[i] = self.extractor.extract_dynamic_features(
                player_input_shifted,
                play['ball_land_x'],
                play['ball_land_y'],
                play['receiver_x'],
                play['receiver_y'],
                num_input_frames=effective_input_frames
            )
            
            # Extract static features
            static_features[i] = self.extractor.extract_static_features(
                player_role=player_role,
                num_output_frames=play['num_output_frames'] + num_earlier_frames,  # Extended prediction
                num_input_frames=effective_input_frames,
                passer_x=play['passer_x'],
                passer_y=play['passer_y'],
                ball_land_x=play['ball_land_x'],
                ball_land_y=play['ball_land_y'],
                player_final_x=final_x,
                player_final_y=final_y
            )
            
            # Extract targets (displacement from final input position)
            if not player_output.empty:
                # ============================================================
                # Adjust target frames: if we shifted earlier, we need to predict
                # the frames that were originally in the input + all output frames
                # ============================================================
                if num_earlier_frames > 0:
                    # Get the frames that were removed from input
                    original_input_last_frames = player_input.iloc[-num_earlier_frames:]
                    
                    # Combine: removed input frames + output frames
                    combined_x = np.concatenate([
                        original_input_last_frames['x'].values,
                        player_output['x'].values
                    ])
                    combined_y = np.concatenate([
                        original_input_last_frames['y'].values,
                        player_output['y'].values
                    ])
                    
                    # Take up to OUTPUT_FRAMES
                    output_x = combined_x[:self.config.OUTPUT_FRAMES]
                    output_y = combined_y[:self.config.OUTPUT_FRAMES]
                else:
                    # Normal case: just use output frames
                    output_x = player_output['x'].values[:self.config.OUTPUT_FRAMES]
                    output_y = player_output['y'].values[:self.config.OUTPUT_FRAMES]
                
                n_out = len(output_x)
                
                # Target is displacement from final input position
                targets[i, 0, :n_out] = output_x - final_x
                targets[i, 1, :n_out] = output_y - final_y
                
                # Pad with last displacement if needed
                if n_out < self.config.OUTPUT_FRAMES:
                    if n_out > 0:
                        targets[i, 0, n_out:] = targets[i, 0, n_out-1]
                        targets[i, 1, n_out:] = targets[i, 1, n_out-1]
            
            player_mask[i] = 1.0
            final_positions[i] = [final_x, final_y]
        
        # Apply spatial augmentations during training
        if self.training:
            # Calculate center for rotation
            valid_mask = player_mask > 0
            if valid_mask.sum() > 0:
                center_x = dynamic_features[valid_mask, 0, -1].mean()
                center_y = dynamic_features[valid_mask, 1, -1].mean()
            else:
                center_x, center_y = 60.0, 26.65
            
            # Rotation augmentation (50% chance)
            if np.random.random() < self.config.ROTATION_PROB:
                dynamic_features, static_features, targets = self.augmenter.rotate_play(
                    dynamic_features, static_features, targets,
                    center_x, center_y
                )
            
            # Vertical flip (50% chance)
            if np.random.random() < self.config.FLIP_PROB:
                dynamic_features, static_features, targets = self.augmenter.vertical_flip(
                    dynamic_features, static_features, targets
                )
        
        return {
            'dynamic_features': torch.from_numpy(dynamic_features),  # (MAX_PLAYERS, 10, INPUT_FRAMES)
            'static_features': torch.from_numpy(static_features),    # (MAX_PLAYERS, 12)
            'targets': torch.from_numpy(targets),                    # (MAX_PLAYERS, 2, OUTPUT_FRAMES)
            'player_mask': torch.from_numpy(player_mask),            # (MAX_PLAYERS,)
            'final_positions': torch.from_numpy(final_positions),    # (MAX_PLAYERS, 2)
            'game_id': play['game_id'],
            'play_id': play['play_id'],
            'frame_offset': num_earlier_frames  # For debugging/analysis
        }


# Create dataset for testing
print("\nCreating dataset...")
dataset = NFLPlayDataset(train_data, output_data, config, training=True)
print(f"Dataset size: {len(dataset)}")

# Test one sample
sample = dataset[0]
print(f"\nSample shapes:")
for key, value in sample.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: {value.shape}")

# %%
# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class DepthwiseConv1dBlock(nn.Module):
    """
    Depthwise separable Conv1d block.
    Each feature channel is processed independently.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            groups=in_channels, padding=0  # No padding to emphasize last frame
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        # x: (batch, channels, frames)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DynamicFeatureEncoder(nn.Module):
    """
    Encode dynamic features using depthwise Conv1d.
    
    Input: (batch, 10 features, 20 frames)
    Output: (batch, 640, 1 frame)
    
    Each of the 10 features is processed by 7 depthwise Conv1d layers
    to produce 64-dim output, then concatenated to 640-dim.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # 7 depthwise conv layers: 20 -> 18 -> 16 -> 14 -> 12 -> 10 -> 8 -> 6
        # Then we take only the last frame: 6 -> 1
        # Actually: 20 -> 18 -> 16 -> 14 -> 12 -> 10 -> 8 (7 layers of kernel=3)
        # We want to end up with 1 frame, so:
        # After 7 conv with kernel=3: 20 - 7*2 = 6 frames, then take last
        
        # Process each feature independently then project to 64 dims
        self.feature_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=0),
                nn.BatchNorm1d(16),
                nn.SiLU(),
                nn.Conv1d(16, 32, kernel_size=3, padding=0),
                nn.BatchNorm1d(32),
                nn.SiLU(),
                nn.Conv1d(32, 32, kernel_size=3, padding=0),
                nn.BatchNorm1d(32),
                nn.SiLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=0),
                nn.BatchNorm1d(64),
                nn.SiLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=0),
                nn.BatchNorm1d(64),
                nn.SiLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=0),
                nn.BatchNorm1d(64),
                nn.SiLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=0),
                nn.BatchNorm1d(64),
                nn.SiLU(),
            )
            for _ in range(config.DYNAMIC_FEATURES)
        ])
        
        # After 7 conv layers with kernel=3: 20 - 7*2 = 6 frames remaining
        # We take only the last frame
    
    def forward(self, x):
        """
        Args:
            x: (batch * players, 10, 20)
        Returns:
            (batch * players, 640, 1)
        """
        # Process each feature independently
        feature_outputs = []
        for i, encoder in enumerate(self.feature_encoders):
            feat = x[:, i:i+1, :]  # (B, 1, 20)
            encoded = encoder(feat)  # (B, 64, 6)
            # Take only the last frame
            encoded = encoded[:, :, -1:]  # (B, 64, 1)
            feature_outputs.append(encoded)
        
        # Concatenate all features
        out = torch.cat(feature_outputs, dim=1)  # (B, 640, 1)
        return out


class StaticFeatureEncoder(nn.Module):
    """
    Encode static features using Conv1d.
    
    Input: (batch * players, 12)
    Output: (batch * players, 64)
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.STATIC_FEATURES, 64),
            nn.BatchNorm1d(64),
            nn.SiLU()
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch * players, 12)
        Returns:
            (batch * players, 64)
        """
        return self.encoder(x)


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with SwiGLU FFN.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, d_model * 4, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, players, d_model)
            mask: (batch, players) - True for valid players
        Returns:
            (batch, players, d_model)
        """
        # Create key_padding_mask (True means ignore/pad)
        key_padding_mask = None
        if mask is not None:
            # mask is (batch, players) with True for valid players
            # key_padding_mask needs True for positions to IGNORE
            key_padding_mask = ~mask  # Invert: True for invalid players
        
        # Self-attention
        x2 = self.norm1(x)
        x2, _ = self.self_attn(x2, x2, x2, key_padding_mask=key_padding_mask)
        x = x + self.dropout(x2)
        
        # FFN
        x2 = self.norm2(x)
        x2 = self.ffn(x2)
        x = x + x2
        
        return x


class InterPlayerInteraction(nn.Module):
    """
    Transformer encoder for inter-player interaction.
    
    Input: (batch, players, 256)
    Output: (batch, players, 256)
    """
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.PLAYER_HIDDEN_DIM,
                config.NUM_HEADS,
                config.DROPOUT
            )
            for _ in range(config.NUM_TRANSFORMER_LAYERS)
        ])
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, players, 256)
            mask: (batch, players) - True for valid players
        Returns:
            (batch, players, 256)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


print("Model components defined.")

# %%
# ============================================================================
# DECODER AND MAIN MODEL
# ============================================================================

class TrajectoryDecoder(nn.Module):
    """
    Decode player features to trajectory predictions.
    
    Input: (batch * players, 256)
    Output: 
        - trajectory: (batch * players, 2, 48) - mean xy for each frame
        - variance: (batch * players, 2, 48) - variance for GaussianNLL
        - auxiliary: (batch * players, 4, 48) - velocity and acceleration variances
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # 256 -> 1536 -> reshape to (32, 48)
        self.expand = nn.Sequential(
            nn.Linear(config.PLAYER_HIDDEN_DIM, config.DECODER_HIDDEN),
            nn.SiLU()
        )
        
        # Reshape: 1536 -> (32, 48)
        # 1536 = 32 * 48
        
        # Conv1d refinement layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )
        
        # Output heads
        # Main output: 2 (xy mean) + 2 (xy variance)
        # Auxiliary: 2 (velocity variance) + 2 (acceleration variance)
        self.output_head = nn.Conv1d(32, 2 + 2 + 4, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: (batch * players, 256)
        Returns:
            mean: (batch * players, 2, 48)
            var: (batch * players, 2, 48)
            aux: (batch * players, 4, 48)
        """
        batch_size = x.size(0)
        
        # Expand
        x = self.expand(x)  # (B, 1536)
        
        # Reshape to (B, 32, 48)
        x = x.view(batch_size, 32, self.config.OUTPUT_FRAMES)
        
        # Conv refinement
        x = self.conv_layers(x)  # (B, 32, 48)
        
        # Output
        out = self.output_head(x)  # (B, 8, 48)
        
        # Split outputs
        mean = out[:, :2, :]  # (B, 2, 48) - xy displacement
        var_raw = out[:, 2:4, :]  # (B, 2, 48) - xy variance
        aux = out[:, 4:, :]  # (B, 4, 48) - auxiliary (velocity/accel variance)
        
        # Constrain variance to positive: softplus(var) + 1e-3
        var = F.softplus(var_raw) + 1e-3
        aux_var = F.softplus(aux) + 1e-3
        
        return mean, var, aux_var


class NFLTrajectoryModel(nn.Module):
    """
    Main model for NFL player trajectory prediction.
    
    Architecture:
        1. Dynamic Feature Encoder: (10, 20) -> (640, 1)
        2. Static Feature Encoder: (12,) -> (64,)
        3. Merge: (640 + 64) -> 256
        4. Inter-player Interaction: Transformer x 3
        5. Decoder: 256 -> (2, 48) + auxiliary
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Encoders
        self.dynamic_encoder = DynamicFeatureEncoder(config)
        self.static_encoder = StaticFeatureEncoder(config)
        
        # Merge layer
        self.merge = nn.Sequential(
            nn.Linear(config.DYNAMIC_TOTAL_DIM + config.STATIC_OUT_DIM, config.PLAYER_HIDDEN_DIM),
            nn.LayerNorm(config.PLAYER_HIDDEN_DIM),
            nn.SiLU()
        )
        
        # Inter-player interaction
        self.interaction = InterPlayerInteraction(config)
        
        # Decoder
        self.decoder = TrajectoryDecoder(config)
    
    def forward(
        self,
        dynamic_features: torch.Tensor,
        static_features: torch.Tensor,
        player_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            dynamic_features: (batch, players, 10, 20)
            static_features: (batch, players, 12)
            player_mask: (batch, players) - 1 for valid players
        
        Returns:
            mean: (batch, players, 2, 48) - predicted xy displacement
            var: (batch, players, 2, 48) - variance
            aux_var: (batch, players, 4, 48) - auxiliary variance
        """
        batch_size = dynamic_features.size(0)
        n_players = dynamic_features.size(1)
        
        # Reshape for encoding: (batch * players, ...)
        dynamic_flat = dynamic_features.view(batch_size * n_players, 10, self.config.INPUT_FRAMES)
        static_flat = static_features.view(batch_size * n_players, -1)
        
        # Encode features
        dynamic_encoded = self.dynamic_encoder(dynamic_flat)  # (B*P, 640, 1)
        dynamic_encoded = dynamic_encoded.squeeze(-1)  # (B*P, 640)
        
        static_encoded = self.static_encoder(static_flat)  # (B*P, 64)
        
        # Merge
        merged = torch.cat([dynamic_encoded, static_encoded], dim=1)  # (B*P, 704)
        player_features = self.merge(merged)  # (B*P, 256)
        
        # Reshape for transformer: (batch, players, 256)
        player_features = player_features.view(batch_size, n_players, -1)
        
        # Inter-player interaction
        player_features = self.interaction(
            player_features, 
            mask=player_mask.bool()
        )  # (B, P, 256)
        
        # Reshape for decoder: (batch * players, 256)
        player_features = player_features.view(batch_size * n_players, -1)
        
        # Decode
        mean, var, aux_var = self.decoder(player_features)
        
        # Reshape outputs: (batch, players, ...)
        mean = mean.view(batch_size, n_players, 2, self.config.OUTPUT_FRAMES)
        var = var.view(batch_size, n_players, 2, self.config.OUTPUT_FRAMES)
        aux_var = aux_var.view(batch_size, n_players, 4, self.config.OUTPUT_FRAMES)
        
        return mean, var, aux_var


# Test model
print("Testing model...")
model = NFLTrajectoryModel(config).to(device)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

# Test forward pass
test_dynamic = torch.randn(2, config.MAX_PLAYERS, 10, config.INPUT_FRAMES).to(device)
test_static = torch.randn(2, config.MAX_PLAYERS, 12).to(device)
test_mask = torch.ones(2, config.MAX_PLAYERS).to(device)
test_mask[:, 15:] = 0  # Only 15 players valid

with torch.no_grad():
    mean, var, aux_var = model(test_dynamic, test_static, test_mask)

print(f"Output shapes:")
print(f"  mean: {mean.shape}")
print(f"  var: {var.shape}")
print(f"  aux_var: {aux_var.shape}")

# %%
# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class TrajectoryLoss(nn.Module):
    """
    Combined loss for trajectory prediction.
    
    Main loss: GaussianNLLLoss for trajectory prediction
    Auxiliary loss: GaussianNLLLoss for velocity and acceleration
    """
    
    def __init__(self, config: Config, aux_weight: float = 0.1):
        super().__init__()
        self.config = config
        self.aux_weight = aux_weight
        self.gaussian_nll = nn.GaussianNLLLoss(reduction='none')
    
    def compute_derivatives(self, trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute velocity (1st difference) and acceleration (2nd difference).
        
        Args:
            trajectory: (batch, players, 2, frames)
        
        Returns:
            velocity: (batch, players, 2, frames-1)
            acceleration: (batch, players, 2, frames-2)
        """
        velocity = trajectory[..., 1:] - trajectory[..., :-1]
        acceleration = velocity[..., 1:] - velocity[..., :-1]
        return velocity, acceleration
    
    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_var: torch.Tensor,
        pred_aux_var: torch.Tensor,
        target: torch.Tensor,
        player_mask: torch.Tensor,
        num_valid_frames: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_mean: (batch, players, 2, 48) - predicted displacement
            pred_var: (batch, players, 2, 48) - variance
            pred_aux_var: (batch, players, 4, 48) - aux variance (vel_var, accel_var)
            target: (batch, players, 2, 48) - target displacement
            player_mask: (batch, players) - 1 for valid players
            num_valid_frames: number of valid output frames (default: all)
        
        Returns:
            Dictionary with loss components
        """
        if num_valid_frames is None:
            num_valid_frames = self.config.OUTPUT_FRAMES
        
        # Expand player mask for broadcasting
        mask = player_mask.unsqueeze(-1).unsqueeze(-1)  # (B, P, 1, 1)
        
        # Main trajectory loss (GaussianNLL)
        main_loss = self.gaussian_nll(
            pred_mean[..., :num_valid_frames],
            target[..., :num_valid_frames],
            pred_var[..., :num_valid_frames]
        )
        main_loss = (main_loss * mask).sum() / (mask.sum() * 2 * num_valid_frames + 1e-8)
        
        # Compute velocity and acceleration from predictions and targets
        pred_vel, pred_accel = self.compute_derivatives(pred_mean)
        target_vel, target_accel = self.compute_derivatives(target)
        
        # Velocity loss
        vel_var = pred_aux_var[:, :, :2, :-1]  # (B, P, 2, 47)
        vel_frames = min(num_valid_frames - 1, vel_var.size(-1))
        vel_loss = self.gaussian_nll(
            pred_vel[..., :vel_frames],
            target_vel[..., :vel_frames],
            vel_var[..., :vel_frames]
        )
        vel_loss = (vel_loss * mask).sum() / (mask.sum() * 2 * vel_frames + 1e-8)
        
        # Acceleration loss
        accel_var = pred_aux_var[:, :, 2:, :-2]  # (B, P, 2, 46)
        accel_frames = min(num_valid_frames - 2, accel_var.size(-1))
        if accel_frames > 0:
            accel_loss = self.gaussian_nll(
                pred_accel[..., :accel_frames],
                target_accel[..., :accel_frames],
                accel_var[..., :accel_frames]
            )
            accel_loss = (accel_loss * mask).sum() / (mask.sum() * 2 * accel_frames + 1e-8)
        else:
            accel_loss = torch.tensor(0.0, device=pred_mean.device)
        
        # Combined auxiliary loss
        aux_loss = vel_loss + accel_loss
        
        # Total loss
        total_loss = main_loss + self.aux_weight * aux_loss
        
        return {
            'total': total_loss,
            'main': main_loss,
            'velocity': vel_loss,
            'acceleration': accel_loss,
            'auxiliary': aux_loss
        }


# ============================================================================
# EXPONENTIAL MOVING AVERAGE
# ============================================================================

class EMA:
    """
    Exponential Moving Average for model parameters.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters with current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply shadow parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# Test loss function
loss_fn = TrajectoryLoss(config)
print("Loss function initialized.")

# Test EMA
ema = EMA(model)
print(f"EMA initialized with decay={config.EMA_DECAY}")

# %%
# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DataLoader."""
    return {
        'dynamic_features': torch.stack([b['dynamic_features'] for b in batch]),
        'static_features': torch.stack([b['static_features'] for b in batch]),
        'targets': torch.stack([b['targets'] for b in batch]),
        'player_mask': torch.stack([b['player_mask'] for b in batch]),
        'final_positions': torch.stack([b['final_positions'] for b in batch]),
        'game_ids': [b['game_id'] for b in batch],
        'play_ids': [b['play_id'] for b in batch]
    }


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: TrajectoryLoss,
    ema: EMA,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move to device
        dynamic = batch['dynamic_features'].to(device)
        static = batch['static_features'].to(device)
        targets = batch['targets'].to(device)
        mask = batch['player_mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_mean, pred_var, pred_aux_var = model(dynamic, static, mask)
        
        # Compute loss
        losses = loss_fn(pred_mean, pred_var, pred_aux_var, targets, mask)
        
        # Backward pass
        losses['total'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update EMA
        ema.update()
        
        # Accumulate losses
        total_loss += losses['total'].item()
        total_main_loss += losses['main'].item()
        total_aux_loss += losses['auxiliary'].item()
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'main': f"{losses['main'].item():.4f}",
            'aux': f"{losses['auxiliary'].item():.4f}"
        })
    
    return {
        'loss': total_loss / n_batches,
        'main_loss': total_main_loss / n_batches,
        'aux_loss': total_aux_loss / n_batches
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: TrajectoryLoss,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0.0
    total_main_loss = 0.0
    total_mse = 0.0  # Changed from total_mae
    n_batches = 0
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        # Move to device
        dynamic = batch['dynamic_features'].to(device)
        static = batch['static_features'].to(device)
        targets = batch['targets'].to(device)
        mask = batch['player_mask'].to(device)
        final_pos = batch['final_positions'].to(device)
        
        # Forward pass
        pred_mean, pred_var, pred_aux_var = model(dynamic, static, mask)
        
        # Compute loss
        losses = loss_fn(pred_mean, pred_var, pred_aux_var, targets, mask)
        
        # Compute MSE (Mean Squared Error) for trajectory
        # pred_mean is displacement, targets is displacement
        squared_error = (pred_mean - targets) ** 2
        mse = (squared_error * mask.unsqueeze(-1).unsqueeze(-1)).sum() / (mask.sum() * 2 * config.OUTPUT_FRAMES + 1e-8)
        
        total_loss += losses['total'].item()
        total_main_loss += losses['main'].item()
        total_mse += mse.item()
        n_batches += 1
    
    # Compute RMSE from average MSE
    avg_mse = total_mse / n_batches
    rmse = avg_mse ** 0.5
    
    return {
        'loss': total_loss / n_batches,
        'main_loss': total_main_loss / n_batches,
        'rmse': rmse,
        'mse': avg_mse  # Also return MSE for reference
    }

print("Training utilities defined with RMSE evaluation.")

# %%
# %%
# ============================================================================
# LOAD WEIGHTS AND COLLECT CV PREDICTIONS
# ============================================================================

def collect_cv_results_from_checkpoints(
    dataset: NFLPlayDataset,
    config: Config,
    device: torch.device,
    checkpoint_dir: str = "checkpoints",
    checkpoint_pattern: str = "model_repeat*_fold*.pt"
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load saved model checkpoints and collect CV predictions without retraining.
    
    Args:
        dataset: Dataset containing all data
        config: Model configuration
        device: Device for inference
        checkpoint_dir: Directory containing checkpoint files
        checkpoint_pattern: Pattern to match checkpoint files (glob pattern)
    
    Returns:
        Dictionary with fold names as keys and (pred_xy, actual_xy) tuples
    """
    import glob
    import re
    
    cv_predictions = {}
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, checkpoint_pattern))
    
    if not checkpoint_files:
        print(f"No checkpoint files found matching pattern: {checkpoint_pattern}")
        return cv_predictions
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Get game_ids for creating fold splits
    game_ids = np.array([play['game_id'] for play in dataset.plays])
    
    for checkpoint_path in sorted(checkpoint_files):
        # Extract fold info from filename
        filename = os.path.basename(checkpoint_path)
        
        # Parse: model_repeat1_fold2_valloss0.1234.pt
        match = re.search(r'repeat(\d+)_fold(\d+)', filename)
        if not match:
            print(f"Skipping {filename} - couldn't parse fold info")
            continue
        
        repeat_num = int(match.group(1))
        fold_num = int(match.group(2))
        fold_name = f"repeat{repeat_num}_fold{fold_num}"
        
        print(f"\nProcessing {fold_name}...")
        print(f"  Loading checkpoint: {filename}")
        
        # Initialize model
        model = NFLTrajectoryModel(config).to(device)
        
        # Load checkpoint
        try:
            model, ema, epoch, val_loss = load_model(
                model=model,
                checkpoint_path=checkpoint_path,
                device=device,
                load_optimizer=False
            )
        except Exception as e:
            print(f"  Error loading checkpoint: {e}")
            continue
        
        # Recreate the same fold split
        # Note: This assumes the same split logic as training
        n_folds = config.N_FOLDS
        
        # Use GroupKFold with the same seed
        np.random.seed(42 + repeat_num - 1)
        unique_games = np.unique(game_ids)
        shuffled_games = unique_games.copy()
        np.random.shuffle(shuffled_games)
        
        kfold = GroupKFold(n_splits=n_folds)
        
        # Get the specific fold split
        for current_fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)), groups=game_ids), 1):
            if current_fold == fold_num:
                # Create validation subset
                val_subset = torch.utils.data.Subset(dataset, val_idx)
                val_loader = DataLoader(
                    val_subset,
                    batch_size=config.BATCH_SIZE,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=0,
                    pin_memory=True
                )
                
                print(f"  Validation samples: {len(val_idx)}")
                
                # Apply EMA weights and collect predictions
                ema.apply_shadow()
                pred_xy, actual_xy, _ = collect_cv_predictions(model, val_loader, device)
                ema.restore()
                
                cv_predictions[fold_name] = (pred_xy, actual_xy)
                print(f"  Collected {len(pred_xy)} valid predictions")
                
                break
    
    return cv_predictions


def evaluate_single_checkpoint(
    checkpoint_path: str,
    dataset: NFLPlayDataset,
    config: Config,
    device: torch.device,
    val_indices: List[int] = None
) -> Dict[str, float]:
    """
    Evaluate a single checkpoint on specified validation indices.
    
    Args:
        checkpoint_path: Path to checkpoint file
        dataset: Full dataset
        config: Model configuration
        device: Device for inference
        val_indices: Validation indices (if None, uses all data)
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating checkpoint: {os.path.basename(checkpoint_path)}")
    
    # Initialize model
    model = NFLTrajectoryModel(config).to(device)
    
    # Load checkpoint
    model, ema, epoch, val_loss = load_model(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        load_optimizer=False
    )
    
    # Create dataloader
    if val_indices is not None:
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        val_loader = DataLoader(
            val_subset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )
    else:
        val_loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )
    
    # Evaluate with EMA weights
    loss_fn = TrajectoryLoss(config)
    ema.apply_shadow()
    metrics = evaluate(model, val_loader, loss_fn, device)
    ema.restore()
    
    # Collect predictions
    ema.apply_shadow()
    pred_xy, actual_xy, _ = collect_cv_predictions(model, val_loader, device)
    ema.restore()
    
    # Calculate additional metrics
    mae_x = np.mean(np.abs(pred_xy[:, 0] - actual_xy[:, 0]))
    mae_y = np.mean(np.abs(pred_xy[:, 1] - actual_xy[:, 1]))
    overall_mae = (mae_x + mae_y) / 2
    
    print(f"Checkpoint Epoch: {epoch}")
    print(f"Val Loss: {metrics['loss']:.4f}")
    print(f"Val RMSE: {metrics['rmse']:.4f}")
    print(f"Overall MAE: {overall_mae:.4f} (X: {mae_x:.4f}, Y: {mae_y:.4f})")
    
    return {
        'epoch': epoch,
        'loss': metrics['loss'],
        'rmse': metrics['rmse'],
        'mae': overall_mae,
        'mae_x': mae_x,
        'mae_y': mae_y,
        'pred_xy': pred_xy,
        'actual_xy': actual_xy
    }


print("Checkpoint loading and evaluation functions defined.")

# %%
# ============================================================================
# CV PREDICTION COLLECTION AND PLOTTING
# ============================================================================

@torch.no_grad()
def collect_cv_predictions(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect predictions and actual values from validation set.
    
    Returns:
        pred_xy: (N, 2) array of predicted (x, y) positions
        actual_xy: (N, 2) array of actual (x, y) positions
        mask: (N,) boolean array indicating valid samples
    """
    model.eval()
    
    all_pred_x = []
    all_pred_y = []
    all_actual_x = []
    all_actual_y = []
    all_masks = []
    
    for batch in tqdm(val_loader, desc="Collecting predictions"):
        dynamic = batch['dynamic_features'].to(device)
        static = batch['static_features'].to(device)
        targets = batch['targets'].to(device)
        mask = batch['player_mask'].to(device)
        final_pos = batch['final_positions'].to(device)
        
        # Forward pass
        pred_mean, _, _ = model(dynamic, static, mask)  # (B, P, 2, 48)
        
        # Convert displacement to absolute positions
        final_pos_expanded = final_pos.unsqueeze(-1)  # (B, P, 2, 1)
        pred_abs = pred_mean + final_pos_expanded  # (B, P, 2, 48)
        target_abs = targets + final_pos_expanded  # (B, P, 2, 48)
        
        # Take the last frame prediction (frame 48)
        pred_last = pred_abs[:, :, :, -1]  # (B, P, 2)
        target_last = target_abs[:, :, :, -1]  # (B, P, 2)
        
        # Flatten and collect
        B, P = mask.shape
        mask_flat = mask.cpu().numpy().flatten()  # (B*P,)
        pred_x_flat = pred_last[:, :, 0].cpu().numpy().flatten()  # (B*P,)
        pred_y_flat = pred_last[:, :, 1].cpu().numpy().flatten()  # (B*P,)
        actual_x_flat = target_last[:, :, 0].cpu().numpy().flatten()  # (B*P,)
        actual_y_flat = target_last[:, :, 1].cpu().numpy().flatten()  # (B*P,)
        
        all_masks.append(mask_flat)
        all_pred_x.append(pred_x_flat)
        all_pred_y.append(pred_y_flat)
        all_actual_x.append(actual_x_flat)
        all_actual_y.append(actual_y_flat)
    
    # Concatenate all batches
    mask_all = np.concatenate(all_masks)
    pred_x_all = np.concatenate(all_pred_x)
    pred_y_all = np.concatenate(all_pred_y)
    actual_x_all = np.concatenate(all_actual_x)
    actual_y_all = np.concatenate(all_actual_y)
    
    # Filter by mask
    valid_mask = mask_all > 0.5
    pred_xy = np.stack([pred_x_all[valid_mask], pred_y_all[valid_mask]], axis=1)
    actual_xy = np.stack([actual_x_all[valid_mask], actual_y_all[valid_mask]], axis=1)
    
    return pred_xy, actual_xy, valid_mask


def plot_cv_results(
    cv_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str = "cv_results_plot.png"
):
    """
    Plot CV results: predicted vs actual XY positions.
    
    Args:
        cv_predictions: Dictionary with fold names as keys and (pred_xy, actual_xy) tuples as values
        save_path: Path to save the plot
    """
    n_folds = len(cv_predictions)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, n_folds, figsize=(5*n_folds, 10))
    if n_folds == 1:
        axes = axes.reshape(2, 1)
    
    fold_names = sorted(cv_predictions.keys())
    
    for idx, fold_name in enumerate(fold_names):
        pred_xy, actual_xy = cv_predictions[fold_name]
        
        # X coordinate subplot
        ax_x = axes[0, idx]
        ax_x.scatter(actual_xy[:, 0], pred_xy[:, 0], alpha=0.3, s=1)
        
        # Add diagonal line (perfect prediction)
        min_val = min(actual_xy[:, 0].min(), pred_xy[:, 0].min())
        max_val = max(actual_xy[:, 0].max(), pred_xy[:, 0].max())
        ax_x.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        # Calculate metrics
        mae_x = np.mean(np.abs(pred_xy[:, 0] - actual_xy[:, 0]))
        rmse_x = np.sqrt(np.mean((pred_xy[:, 0] - actual_xy[:, 0])**2))
        
        ax_x.set_xlabel('Actual X', fontsize=12)
        ax_x.set_ylabel('Predicted X', fontsize=12)
        ax_x.set_title(f'{fold_name}\nMAE: {mae_x:.3f}, RMSE: {rmse_x:.3f}', fontsize=12)
        ax_x.legend()
        ax_x.grid(True, alpha=0.3)
        
        # Y coordinate subplot
        ax_y = axes[1, idx]
        ax_y.scatter(actual_xy[:, 1], pred_xy[:, 1], alpha=0.3, s=1)
        
        # Add diagonal line
        min_val = min(actual_xy[:, 1].min(), pred_xy[:, 1].min())
        max_val = max(actual_xy[:, 1].max(), pred_xy[:, 1].max())
        ax_y.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        # Calculate metrics
        mae_y = np.mean(np.abs(pred_xy[:, 1] - actual_xy[:, 1]))
        rmse_y = np.sqrt(np.mean((pred_xy[:, 1] - actual_xy[:, 1])**2))
        
        ax_y.set_xlabel('Actual Y', fontsize=12)
        ax_y.set_ylabel('Predicted Y', fontsize=12)
        ax_y.set_title(f'{fold_name}\nMAE: {mae_y:.3f}, RMSE: {rmse_y:.3f}', fontsize=12)
        ax_y.legend()
        ax_y.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()
    
    # Print overall statistics
    print("\n" + "="*50)
    print("Overall CV Statistics")
    print("="*50)
    
    all_pred_x = []
    all_pred_y = []
    all_actual_x = []
    all_actual_y = []
    
    for fold_name in fold_names:
        pred_xy, actual_xy = cv_predictions[fold_name]
        all_pred_x.append(pred_xy[:, 0])
        all_pred_y.append(pred_xy[:, 1])
        all_actual_x.append(actual_xy[:, 0])
        all_actual_y.append(actual_xy[:, 1])
    
    all_pred_x = np.concatenate(all_pred_x)
    all_pred_y = np.concatenate(all_pred_y)
    all_actual_x = np.concatenate(all_actual_x)
    all_actual_y = np.concatenate(all_actual_y)
    
    overall_mae_x = np.mean(np.abs(all_pred_x - all_actual_x))
    overall_mae_y = np.mean(np.abs(all_pred_y - all_actual_y))
    overall_rmse_x = np.sqrt(np.mean((all_pred_x - all_actual_x)**2))
    overall_rmse_y = np.sqrt(np.mean((all_pred_y - all_actual_y)**2))
    overall_mae = (overall_mae_x + overall_mae_y) / 2
    overall_rmse = np.sqrt((overall_rmse_x**2 + overall_rmse_y**2) / 2)
    
    print(f"X - MAE: {overall_mae_x:.3f}, RMSE: {overall_rmse_x:.3f}")
    print(f"Y - MAE: {overall_mae_y:.3f}, RMSE: {overall_rmse_y:.3f}")
    print(f"Overall - MAE: {overall_mae:.3f}, RMSE: {overall_rmse:.3f}")


print("CV prediction collection and plotting functions defined.")

# %%
# ============================================================================
# MODEL SAVING UTILITIES
# ============================================================================

def save_model(
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    save_dir: str = "checkpoints",
    filename: str = None
):
    """
    Save model checkpoint with EMA weights.
    
    Args:
        model: The model to save
        ema: EMA object containing shadow weights
        optimizer: Optimizer state
        epoch: Current epoch
        val_loss: Validation loss
        save_dir: Directory to save checkpoint
        filename: Checkpoint filename (optional)
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        filename = f"model_epoch{epoch}_valloss{val_loss:.4f}.pt"
    
    # Full path
    save_path = os.path.join(save_dir, filename)
    
    # Prepare checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_shadow': ema.shadow,
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': {
            'max_players': config.MAX_PLAYERS,
            'input_frames': config.INPUT_FRAMES,
            'output_frames': config.OUTPUT_FRAMES,
            'dynamic_features': config.DYNAMIC_FEATURES,
            'static_features': config.STATIC_FEATURES,
        }
    }
    
    # Save checkpoint
    torch.save(checkpoint, save_path)


def load_model(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    load_optimizer: bool = False,
    optimizer: torch.optim.Optimizer = None
) -> Tuple[nn.Module, Optional[EMA], int, float]:
    """
    Load model checkpoint.
    
    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        load_optimizer: Whether to load optimizer state
        optimizer: Optimizer instance (required if load_optimizer=True)
    
    Returns:
        model: Model with loaded weights
        ema: EMA object with loaded shadow weights
        epoch: Epoch number from checkpoint
        val_loss: Validation loss from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Recreate EMA with shadow weights
    ema = EMA(model)
    ema.shadow = checkpoint['ema_shadow']
    
    # Load optimizer if requested
    if load_optimizer and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {epoch}, Val Loss: {val_loss:.4f}")
    
    return model, ema, epoch, val_loss


print("Model saving/loading utilities defined.")

# %%

# ============================================================================
# CROSS-VALIDATION TRAINING
# ============================================================================

def run_cross_validation(
    dataset: NFLPlayDataset,
    config: Config,
    device: torch.device,
    n_folds: int = 5,
    n_repeats: int = 1,
    epochs: int = 210,
    eval_every: int = 6
) -> Dict[str, List[float]]:
    """
    Run group k-fold cross-validation.
    
    Args:
        dataset: Training dataset
        config: Model configuration
        device: Device to train on
        n_folds: Number of CV folds
        n_repeats: Number of times to repeat CV with different splits
        epochs: Total training epochs
        eval_every: Evaluate every N epochs
    
    Returns:
        Dictionary with validation metrics
    """
    # Get game_ids for grouping
    game_ids = np.array([play['game_id'] for play in dataset.plays])
    unique_games = np.unique(game_ids)
    
    all_val_losses = []
    all_val_rmses = []  # Changed from all_val_maes
    cv_predictions = {}  # Store predictions for each fold
    
    for repeat in range(n_repeats):
        print(f"\n{'='*50}")
        print(f"Cross-Validation Repeat {repeat + 1}/{n_repeats}")
        print(f"{'='*50}")
        
        # Shuffle games for different splits
        np.random.seed(42 + repeat)
        shuffled_games = unique_games.copy()
        np.random.shuffle(shuffled_games)
        
        # Create GroupKFold splits
        kfold = GroupKFold(n_splits=n_folds)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)), groups=game_ids)):
            print(f"\n--- Fold {fold + 1}/{n_folds} ---")
            print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
            
            # Create data subsets
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            
            train_loader = DataLoader(
                train_subset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=True
            )
            
            # Initialize model, optimizer, loss, EMA
            model = NFLTrajectoryModel(config).to(device)
            optimizer = RAdam(model.parameters(), lr=config.LEARNING_RATE)
            loss_fn = TrajectoryLoss(config)
            ema = EMA(model, decay=config.EMA_DECAY)
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )
            
            best_val_loss = float('inf')
            best_val_rmse = float('inf')  # Changed from best_val_mae
            
            # Training loop
            for epoch in range(1, epochs + 1):
                # Train
                train_metrics = train_one_epoch(
                    model, train_loader, optimizer, loss_fn, ema, device, epoch
                )
                
                scheduler.step()
                
                # Evaluate every N epochs
                if epoch % eval_every == 0 or epoch == epochs:
                    # Apply EMA for evaluation
                    ema.apply_shadow()
                    
                    val_metrics = evaluate(model, val_loader, loss_fn, device)
                    
                    # Restore original weights
                    ema.restore()
                    
                    print(f"  Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                          f"Val Loss={val_metrics['loss']:.4f}, Val RMSE={val_metrics['rmse']:.4f}")
                    
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        best_val_rmse = val_metrics['rmse']  # Changed from best_val_mae
            
            print(f"  Best Val Loss: {best_val_loss:.4f}, Best Val RMSE: {best_val_rmse:.4f}")
            all_val_losses.append(best_val_loss)
            all_val_rmses.append(best_val_rmse)  # Changed from all_val_maes
            
            # Save model for this fold
            fold_name = f"repeat{repeat+1}_fold{fold+1}"
            save_model(
                model=model,
                ema=ema,
                optimizer=optimizer,
                epoch=epochs,
                val_loss=best_val_loss,
                save_dir="checkpoints",
                filename=f"model_{fold_name}_valloss{best_val_loss:.4f}.pt"
            )
            print(f"  Saved model: checkpoints/model_{fold_name}_valloss{best_val_loss:.4f}.pt")
            
            # Collect predictions for this fold
            print(f"  Collecting predictions for {fold_name}...")
            ema.apply_shadow()  # Use EMA weights for predictions
            pred_xy, actual_xy, _ = collect_cv_predictions(model, val_loader, device)
            ema.restore()
            cv_predictions[fold_name] = (pred_xy, actual_xy)
            print(f"  Collected {len(pred_xy)} valid predictions")
    
    # Summary
    print(f"\n{'='*50}")
    print("Cross-Validation Summary")
    print(f"{'='*50}")
    print(f"Mean Val Loss: {np.mean(all_val_losses):.4f} ± {np.std(all_val_losses):.4f}")
    print(f"Mean Val RMSE: {np.mean(all_val_rmses):.4f} ± {np.std(all_val_rmses):.4f}")
    
    return {
        'val_losses': all_val_losses,
        'val_rmses': all_val_rmses,  # Changed from val_maes
        'cv_predictions': cv_predictions
    }


print("Cross-validation training function defined with RMSE.")

# %%
# ============================================================================
# INFERENCE UTILITIES
# ============================================================================

@torch.no_grad()
def predict_trajectories(
    model: nn.Module,
    dataset: NFLPlayDataset,
    device: torch.device,
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Generate trajectory predictions for submission.
    
    Returns:
        DataFrame with columns: game_id, play_id, nfl_id, frame_id, x, y
    """
    model.eval()
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    all_predictions = []
    
    for batch in tqdm(loader, desc="Predicting"):
        dynamic = batch['dynamic_features'].to(device)
        static = batch['static_features'].to(device)
        mask = batch['player_mask'].to(device)
        final_pos = batch['final_positions'].to(device)
        game_ids = batch['game_ids']
        play_ids = batch['play_ids']
        
        # Forward pass
        pred_mean, _, _ = model(dynamic, static, mask)  # (B, P, 2, 48)
        
        # Convert displacement to absolute positions
        # final_pos: (B, P, 2)
        final_pos_expanded = final_pos.unsqueeze(-1)  # (B, P, 2, 1)
        pred_abs = pred_mean + final_pos_expanded  # (B, P, 2, 48)
        
        # Convert to numpy
        pred_abs = pred_abs.cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        # Extract predictions for each sample
        for b_idx in range(len(game_ids)):
            game_id = game_ids[b_idx]
            play_id = play_ids[b_idx]
            
            play_info = dataset.plays[b_idx]
            player_ids = play_info['player_ids']
            
            for p_idx, nfl_id in enumerate(player_ids[:config.MAX_PLAYERS]):
                if mask_np[b_idx, p_idx] < 0.5:
                    continue
                
                for frame_idx in range(config.OUTPUT_FRAMES):
                    all_predictions.append({
                        'game_id': game_id,
                        'play_id': play_id,
                        'nfl_id': nfl_id,
                        'frame_id': frame_idx + 1,
                        'x': pred_abs[b_idx, p_idx, 0, frame_idx],
                        'y': pred_abs[b_idx, p_idx, 1, frame_idx]
                    })
    
    return pd.DataFrame(all_predictions)


def train_single_model(
    train_data: pd.DataFrame,
    output_data: pd.DataFrame,
    config: Config,
    device: torch.device,
    epochs: int = 210,
    eval_split: float = 0.1
) -> Tuple[nn.Module, EMA]:
    """
    Train a single model on all data.
    
    Returns:
        Trained model and EMA
    """
    # Create dataset
    dataset = NFLPlayDataset(train_data, output_data, config, training=True)
    
    # Split for validation
    n_val = int(len(dataset) * eval_split)
    n_train = len(dataset) - n_val
    
    # Random split (not grouped for simplicity)
    indices = np.random.permutation(len(dataset))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    # Initialize
    model = NFLTrajectoryModel(config).to(device)
    optimizer = RAdam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = TrajectoryLoss(config)
    ema = EMA(model, decay=config.EMA_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, ema, device, epoch)
        scheduler.step()
        
        if epoch % config.EVAL_EVERY == 0 or epoch == epochs:
            ema.apply_shadow()
            val_metrics = evaluate(model, val_loader, loss_fn, device)
            ema.restore()
            
            print(f"Epoch {epoch}: Train={train_metrics['loss']:.4f}, Val={val_metrics['loss']:.4f}")
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                # Save best EMA state
                best_state = {k: v.clone() for k, v in ema.shadow.items()}
    
    # Load best EMA state
    if best_state is not None:
        for name, param in model.named_parameters():
            if name in best_state:
                param.data = best_state[name]
    
    return model, ema


print("Inference utilities defined.")
# %%
# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Quick test with small dataset (week 1 only)
print("=" * 60)
print("NFL Player Trajectory Prediction Model")
print("=" * 60)
print(f"\nConfiguration:")
print(f"  Max players: {config.MAX_PLAYERS}")
print(f"  Input frames: {config.INPUT_FRAMES}")
print(f"  Output frames: {config.OUTPUT_FRAMES}")
print(f"  Dynamic features: {config.DYNAMIC_FEATURES}")
print(f"  Static features: {config.STATIC_FEATURES}")
print(f"  Hidden dim: {config.PLAYER_HIDDEN_DIM}")
print(f"  Transformer layers: {config.NUM_TRANSFORMER_LAYERS}")
print(f"  Batch size: {config.BATCH_SIZE}")
print(f"  Learning rate: {config.LEARNING_RATE}")
print(f"  EMA decay: {config.EMA_DECAY}")
print(f"\nDevice: {device}")

# For quick testing, just run a few epochs
QUICK_TEST = False

if QUICK_TEST:
    print("\n" + "=" * 60)
    print("QUICK TEST MODE - Running short training")
    print("=" * 60)
    
    # Create small dataset
    test_dataset = dataset  # Use the dataset created earlier
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create model
    model = NFLTrajectoryModel(config).to(device)
    optimizer = RAdam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = TrajectoryLoss(config)
    ema = EMA(model, decay=config.EMA_DECAY)
    
    # Train for a few epochs
    print("\nTraining for 3 epochs...")
    best_loss = float('inf')
    for epoch in range(1, 4):
        train_metrics = train_one_epoch(model, test_loader, optimizer, loss_fn, ema, device, epoch)
        print(f"Epoch {epoch}: Loss={train_metrics['loss']:.4f}, Main={train_metrics['main_loss']:.4f}")
        
        # Save best model
        if train_metrics['loss'] < best_loss:
            best_loss = train_metrics['loss']
    
    # Save final model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'ema_shadow': ema.shadow,
        'epoch': epoch,
        'loss': best_loss,
    }, "checkpoints/trajectory_model_latest.pt")
    print(f"\nModel saved to checkpoints/trajectory_model_latest.pt")
    
    print("\nQuick test completed successfully!")
    
else:
    # Full cross-validation training
    print("\n" + "=" * 60)
    print("FULL TRAINING - Cross-Validation")
    print("=" * 60)
    
    # Load all data
    train_data_full, output_data_full = load_all_data(weeks=range(1, 19))
    
    # Create dataset
    full_dataset = NFLPlayDataset(train_data_full, output_data_full, config, training=True)
    
    # Run cross-validation
    cv_results = run_cross_validation(
        full_dataset,
        config,
        device,
        n_folds=config.N_FOLDS,
        n_repeats=config.N_REPEATS,
        epochs=config.EPOCHS,
        eval_every=config.EVAL_EVERY
    )

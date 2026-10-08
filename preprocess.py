import pandas as pd
import numpy as np

def extract_features(file_path, label, window_size=50):
    df = pd.read_csv(file_path)
    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()
    df['dt'] = df['t'].diff()
    
    df = df.dropna()
    df = df[df['dt'] > 0.001]
    
    df['dist'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['speed'] = df['dist'] / df['dt']
    df['accel'] = df['speed'].diff() / df['dt']
    df = df.dropna()
    
    features = []
    for i in range(0, len(df) - window_size, window_size):
        window = df.iloc[i : i + window_size]
        mean_speed = window['speed'].mean()
        std_speed = window['speed'].std()
        mean_accel = window['accel'].mean()
        std_accel = window['accel'].std()
        
        total_path_length = window['dist'].sum()
        direct_distance = np.sqrt(
            (window.iloc[-1]['x'] - window.iloc[0]['x'])**2 + 
            (window.iloc[-1]['y'] - window.iloc[0]['y'])**2
        )
        straightness = direct_distance / total_path_length if total_path_length > 0 else 1.0
        features.append([mean_speed, std_speed, mean_accel, std_accel, straightness, label])
        
    return pd.DataFrame(features, columns=['mean_speed', 'std_speed', 'mean_accel', 'std_accel', 'straightness', 'label'])

try:
    df_human = extract_features("mouse_human.csv", label=1)
    df_bot = extract_features("mouse_bot.csv", label=0)
    dataset = pd.concat([df_human, df_bot], ignore_index=True)
    dataset.to_csv("processed_dataset.csv", index=False)
    print(f"Human: {len(df_human)}, Bot: {len(df_bot)}")
except FileNotFoundError as e:
    print(f"Error: {e}")
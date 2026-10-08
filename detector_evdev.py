import evdev
import time
import numpy as np
import torch
import torch.nn as nn
import joblib

class AntiBotNet(nn.Module):
    def __init__(self, input_dim):
        super(AntiBotNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

try:
    scaler = joblib.load("scaler.pkl")
    model = AntiBotNet(input_dim=5)
    model.load_state_dict(torch.load("antibot_model.pth"))
    model.eval()
except Exception as e:
    print(f"Error: {e}")
    exit()

devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
mouse_devices = []
for device in devices:
    try:
        capabilities = device.capabilities()
        if evdev.ecodes.EV_REL in capabilities:
            rel_codes = capabilities[evdev.ecodes.EV_REL]
            if evdev.ecodes.REL_X in rel_codes and evdev.ecodes.REL_Y in rel_codes:
                mouse_devices.append(device)
    except Exception:
        pass

if not mouse_devices:
    print("Device not found.")
    exit()

mouse_device = None
for d in mouse_devices:
    if "mouse" in d.name.lower():
        mouse_device = d
        break

if not mouse_device:
    mouse_device = mouse_devices[0]

print(f"Device: {mouse_device.name} ({mouse_device.path})")

buffer = []
WINDOW_SIZE = 50
x, y = 0, 0

history = []
HISTORY_MAX_SIZE = 15

try:
    for event in mouse_device.read_loop():
        if event.type == evdev.ecodes.EV_REL:
            if event.code == evdev.ecodes.REL_X:
                x += event.value
            elif event.code == evdev.ecodes.REL_Y:
                y += event.value
        elif event.type == evdev.ecodes.EV_SYN:
            t = time.time()
            buffer.append([x, y, t])
            
            if len(buffer) >= WINDOW_SIZE:
                points = np.array(buffer)
                buffer.clear()
                
                dx = np.diff(points[:, 0])
                dy = np.diff(points[:, 1])
                dt = np.diff(points[:, 2])
                dt[dt <= 0.001] = 0.001
                
                dist = np.sqrt(dx**2 + dy**2)
                speed = dist / dt
                accel = np.diff(speed) / dt[:-1]
                
                mean_speed = np.mean(speed)
                std_speed = np.std(speed)
                mean_accel = np.mean(accel) if len(accel) > 0 else 0
                std_accel = np.std(accel) if len(accel) > 0 else 0
                
                total_path = np.sum(dist)
                direct_dist = np.sqrt((points[-1, 0] - points[0, 0])**2 + (points[-1, 1] - points[0, 1])**2)
                straightness = direct_dist / total_path if total_path > 0 else 1.0
                
                features = np.array([[mean_speed, std_speed, mean_accel, std_accel, straightness]])
                features_scaled = scaler.transform(features)
                features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
                
                with torch.no_grad():
                    probability = model(features_tensor).item()
                
                history.append(probability)
                if len(history) > HISTORY_MAX_SIZE:
                    history.pop(0)
                    
                avg_prob = np.mean(history)
                stats = f"[Str: {straightness:.2f} | StdS: {std_speed:.1f}]"
                
                if avg_prob > 0.5:
                    confidence = avg_prob * 100
                    print(f"{stats} Вердикт: ЧЕЛОВЕК (Уверенность: {confidence:.1f}%)          ", end="\r")
                else:
                    confidence = (1 - avg_prob) * 100
                    print(f"{stats} Вердикт: ОБНАРУЖЕН БОТ! (Уверенность: {confidence:.1f}%)  ", end="\r")
except KeyboardInterrupt:
    print("\nОстановлено.")
    if history:
        final_prob = np.mean(history)
        print("\n" + "="*30)
        print("    ИТОГОВЫЙ ОТЧЕТ ЗА СЕССИЮ")
        print("="*30)
        if final_prob > 0.5:
            print(f"ИТОГ: ЧЕЛОВЕК (Уверенность: {final_prob*100:.1f}%)")
        else:
            print(f"ИТОГ: БОТ (Уверенность: {(1-final_prob)*100:.1f}%)")
        print("="*30)
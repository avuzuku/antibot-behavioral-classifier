import evdev
import time
import csv

devices = []
try:
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
except PermissionError:
    print("Ошибка прав! Запустите скрипт через sudo.")
    exit()

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
    print("Устройства с осями перемещения X/Y не найдены.")
    exit()

mouse_device = None
for d in mouse_devices:
    if "mouse" in d.name.lower():
        mouse_device = d
        break

if not mouse_device:
    mouse_device = mouse_devices[0]

print(f"Device: {mouse_device.name} ({mouse_device.path})")
OUTPUT_FILE = "mouse_human.csv"

x, y = 0, 0
data = []

print(f"Recording to {OUTPUT_FILE}...")
try:
    for event in mouse_device.read_loop():
        if event.type == evdev.ecodes.EV_REL:
            if event.code == evdev.ecodes.REL_X:
                x += event.value
            elif event.code == evdev.ecodes.REL_Y:
                y += event.value
        elif event.type == evdev.ecodes.EV_SYN:
            t = time.time()
            data.append([x, y, t])
except KeyboardInterrupt:
    with open(OUTPUT_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "t"])
        writer.writerows(data)
    print(f"\nSaved {len(data)} points.")
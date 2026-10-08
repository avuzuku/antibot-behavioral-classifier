import evdev
from evdev import UInput, ecodes as e
import time
import random

cap = {
    e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT],
    e.EV_REL: [e.REL_X, e.REL_Y]
}

print("Инициализация виртуальной мыши...")
try:
    with UInput(cap, name="Virtual Bot Mouse") as ui:
        print("Устройство 'Virtual Bot Mouse' успешно создано в ядре!")
        print("Бот начнет симуляцию движения через 5 секунд...")
        print("Для остановки нажмите Ctrl+C.")
        time.sleep(5)
        
        while True:
            dx = random.choice([-3, -1, 1, 3])
            dy = random.choice([-3, -1, 1, 3])
            
            for _ in range(60):
                ui.write(e.EV_REL, e.REL_X, dx)
                ui.write(e.EV_REL, e.REL_Y, dy)
                ui.syn()
                time.sleep(0.01)
                
            time.sleep(random.uniform(0.5, 1.0))
except PermissionError:
    print("Ошибка! Для создания uinput-устройства требуются права sudo.")
except KeyboardInterrupt:
    print("\nБот остановлен.")
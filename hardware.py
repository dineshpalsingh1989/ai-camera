"""Optional peripherals for the Cytron IRIV EdgeAI CM5 (Raspberry Pi CM5 + Hailo AI HAT+).

Everything here is a safe no-op unless a real CM5 *and* the Hailo HAT are detected, so the dashboard
and the demo still run unchanged on an ordinary PC. Pin map is from the IRIV EdgeAI CM5 datasheet.
GPIO/OLED code only runs on the board and must be verified there; it never affects monitoring.
"""
import socket
import threading
import time

# IRIV EdgeAI CM5 pin map (BCM numbering), from the datasheet "Pin & Interface Mapping".
BUZZER, LED0, LED1 = 19, 20, 21   # active buzzer + user LEDs
DO0, DO1 = 23, 24                 # isolated digital outputs — drive an external siren / strobe / relay
OLED_ADDR = 0x3C                  # SSD1315 0.96" OLED on I2C1


def board_present(hailo_ready):
    """True only on a Raspberry Pi Compute Module 5 with the Hailo HAT active (not CPU fallback)."""
    if not hailo_ready:
        return False
    try:
        with open("/proc/device-tree/model") as f:
            return "Compute Module 5" in f.read()
    except OSError:
        return False  # not a Pi (e.g. a Windows/Linux dev PC) -> peripherals stay off


class Peripherals:
    """Buzzer, LEDs, isolated digital outputs and OLED on the IRIV board. Every method is a no-op when
    the board is absent or the gpiozero/luma libraries are missing, so a plain PC runs the demo fine."""

    def __init__(self, hailo_ready):
        self.enabled = board_present(hailo_ready)
        self.oled = None
        self._out = {}
        self._lock = threading.Lock()
        if self.enabled:
            self._setup_gpio()
            self._setup_oled()

    def _setup_gpio(self):
        try:
            from gpiozero import LED, Buzzer  # uses lgpio on Pi 5 / CM5
            self._out = {"buzzer": Buzzer(BUZZER), "led0": LED(LED0), "led1": LED(LED1),
                         "do0": LED(DO0), "do1": LED(DO1)}
        except Exception as e:  # missing lib or pin busy -> skip GPIO, keep running
            print(f"[hardware] GPIO unavailable: {e}")
            self._out = {}

    def _setup_oled(self):
        try:
            from luma.core.interface.serial import i2c
            from luma.oled.device import ssd1306  # SSD1315 is SSD1306-compatible
            self.oled = ssd1306(i2c(port=1, address=OLED_ADDR))
        except Exception as e:
            print(f"[hardware] OLED unavailable: {e}")
            self.oled = None

    def on_alert(self, label, seconds=3):
        """Beep the buzzer and pulse digital output DO0 (+ LED0) so an external siren/light can react."""
        if not self._out:
            return
        names = [n for n in ("buzzer", "do0", "led0") if n in self._out]

        def pulse():
            with self._lock:  # one alert at a time; a burst of events won't stack relay pulses
                for n in names:
                    self._out[n].on()
                time.sleep(seconds)
                for n in names:
                    self._out[n].off()
        threading.Thread(target=pulse, daemon=True).start()

    def set_health(self, all_online):
        """LED1 steady when every camera is connected, off otherwise (a glance tells you the box is healthy)."""
        if "led1" in self._out:
            self._out["led1"].on() if all_online else self._out["led1"].off()

    def update_status(self, lines):
        if not self.oled:
            return
        try:
            from luma.core.render import canvas
            with canvas(self.oled) as draw:
                for i, line in enumerate(lines[:5]):
                    draw.text((0, i * 12), str(line)[:21], fill="white")
        except Exception:
            pass

    def cleanup(self):
        for out in self._out.values():
            try:
                out.off()
            except Exception:
                pass


def _local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "offline"


def status_loop(peripherals, cameras, interval=5):
    """Refresh the OLED and health LED with camera status. Returns immediately when the board is absent."""
    if not peripherals.enabled:
        return
    ip = _local_ip()
    while True:
        online = sum(1 for c in cameras.values() if getattr(c, "online", False))
        total = len(cameras)
        peripherals.set_health(total > 0 and online == total)
        peripherals.update_status(["AI Camera", f"IP {ip}", f"Cameras {online}/{total}"])
        time.sleep(interval)

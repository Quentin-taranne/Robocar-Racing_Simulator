import os

os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import pygame
from pynput import keyboard


def apply_deadzone(value: float, deadzone: float) -> float:
    if abs(value) < deadzone:
        return 0.0
    return value


class PygameKeyboardController:
    """Clavier via fenêtre pygame (nécessite le focus)."""

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((360, 120))
        pygame.display.set_caption("Robocar Controller (flèches ou WASD)")
        self.font = pygame.font.SysFont("Courier", 18)

    def poll(self) -> tuple[float, float, float]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        keys = pygame.key.get_pressed()
        steer = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steer -= 1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steer += 1.0

        throttle = 1.0 if keys[pygame.K_UP] or keys[pygame.K_w] else 0.0
        brake = 1.0 if keys[pygame.K_DOWN] or keys[pygame.K_s] else 0.0

        self._render(steer, throttle, brake)
        return steer, throttle, brake

    def _render(self, steer: float, throttle: float, brake: float) -> None:
        self.screen.fill((15, 17, 26))
        text = self.font.render(
            f"Steer {steer:+.1f}  Throttle {throttle:.1f}  Brake {brake:.1f}",
            True,
            (180, 220, 255),
        )
        self.screen.blit(text, (14, 40))
        pygame.display.flip()


class GlobalKeyboardController:
    """
    Clavier global (pas besoin de focus sur la fenêtre pygame).
    Nécessite l'autorisation "Accessibilité" sur macOS pour le terminal.
    """

    def __init__(self) -> None:
        self.state = {
            "left": False,
            "right": False,
            "up": False,
            "down": False,
            "a": False,
            "d": False,
            "w": False,
            "s": False,
        }
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False,
        )
        self.listener.start()

    def _on_press(self, key):
        try:
            k = key.char
        except AttributeError:
            k = key
        if k in (keyboard.Key.left, "a"):
            self.state["left"] = True
        if k in (keyboard.Key.right, "d"):
            self.state["right"] = True
        if k in (keyboard.Key.up, "w"):
            self.state["up"] = True
        if k in (keyboard.Key.down, "s"):
            self.state["down"] = True
        if k == keyboard.Key.esc:
            raise KeyboardInterrupt

    def _on_release(self, key):
        try:
            k = key.char
        except AttributeError:
            k = key
        if k in (keyboard.Key.left, "a"):
            self.state["left"] = False
        if k in (keyboard.Key.right, "d"):
            self.state["right"] = False
        if k in (keyboard.Key.up, "w"):
            self.state["up"] = False
        if k in (keyboard.Key.down, "s"):
            self.state["down"] = False

    def poll(self) -> tuple[float, float, float]:
        steer = 0.0
        if self.state["left"]:
            steer -= 1.0
        if self.state["right"]:
            steer += 1.0
        throttle = 1.0 if self.state["up"] else 0.0
        brake = 1.0 if self.state["down"] else 0.0
        return steer, throttle, brake


class GamepadController:
    """Manette via pygame joystick."""

    def __init__(
        self,
        *,
        index: int = 0,
        steer_axis: int = 0,
        throttle_axis: int = 5,
        brake_axis: int = 4,
        throttle_button: int = 0,
        brake_button: int = 1,
        deadzone: float = 0.12,
        invert_steer: bool = False,
        debug: bool = False,
    ) -> None:
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= index:
            raise RuntimeError(
                f"Aucune manette trouvee a l'index {index}. "
                f"Manettes detectees: {pygame.joystick.get_count()}."
            )

        self.joystick = pygame.joystick.Joystick(index)
        self.joystick.init()
        self.steer_axis = steer_axis
        self.throttle_axis = throttle_axis
        self.brake_axis = brake_axis
        self.throttle_button = throttle_button
        self.brake_button = brake_button
        self.deadzone = deadzone
        self.invert_steer = invert_steer
        self.debug = debug
        self.screen = pygame.display.set_mode((760, 220) if debug else (520, 140))
        pygame.display.set_caption(f"Robocar Gamepad ({self.joystick.get_name()})")
        self.font = pygame.font.SysFont("Courier", 16)

    def poll(self) -> tuple[float, float, float]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        steer = self._axis(self.steer_axis, default=0.0)
        steer = apply_deadzone(steer, self.deadzone)
        if self.invert_steer:
            steer *= -1.0

        throttle = self._trigger_or_button(self.throttle_axis, self.throttle_button)
        brake = self._trigger_or_button(self.brake_axis, self.brake_button)

        self._render(steer, throttle, brake)
        return steer, throttle, brake

    def _axis(self, axis: int, *, default: float) -> float:
        if axis < 0 or axis >= self.joystick.get_numaxes():
            return default
        return float(self.joystick.get_axis(axis))

    def _button(self, button: int) -> float:
        if button < 0 or button >= self.joystick.get_numbuttons():
            return 0.0
        return float(self.joystick.get_button(button))

    def _trigger_or_button(self, axis: int, button: int) -> float:
        if 0 <= axis < self.joystick.get_numaxes():
            value = self._axis(axis, default=-1.0)
            if value < -self.deadzone:
                return float(max(0.0, min(1.0, (value + 1.0) / 2.0)))
            return float(max(0.0, min(1.0, value)))
        return self._button(button)

    def _render(self, steer: float, throttle: float, brake: float) -> None:
        self.screen.fill((15, 17, 26))
        lines = [
            f"{self.joystick.get_name()}",
            f"Steer {steer:+.2f}  Throttle {throttle:.2f}  Brake {brake:.2f}",
            f"axes={self.joystick.get_numaxes()} buttons={self.joystick.get_numbuttons()}",
        ]
        if self.debug:
            axes = " ".join(
                f"{axis}:{self.joystick.get_axis(axis):+.2f}"
                for axis in range(self.joystick.get_numaxes())
            )
            buttons = " ".join(
                f"{button}:{self.joystick.get_button(button)}"
                for button in range(self.joystick.get_numbuttons())
            )
            lines.extend([f"axes {axes}", f"buttons {buttons}"])
        for index, line in enumerate(lines):
            text = self.font.render(line, True, (180, 220, 255))
            self.screen.blit(text, (14, 18 + index * 34))
        pygame.display.flip()


class AutoInputController:
    """Clavier pygame + manette optionnelle, sans permission globale macOS."""

    def __init__(
        self,
        *,
        index: int = 0,
        steer_axis: int = 0,
        throttle_axis: int = 5,
        brake_axis: int = 4,
        throttle_button: int = 0,
        brake_button: int = 1,
        deadzone: float = 0.12,
        invert_steer: bool = False,
        debug: bool = False,
    ) -> None:
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        if pygame.joystick.get_count() > index:
            self.joystick = pygame.joystick.Joystick(index)
            self.joystick.init()

        self.steer_axis = steer_axis
        self.throttle_axis = throttle_axis
        self.brake_axis = brake_axis
        self.throttle_button = throttle_button
        self.brake_button = brake_button
        self.deadzone = deadzone
        self.invert_steer = invert_steer
        self.debug = debug
        self.screen = pygame.display.set_mode((780, 240) if debug else (560, 150))
        pygame.display.set_caption("Robocar Input (keyboard + optional gamepad)")
        self.font = pygame.font.SysFont("Courier", 16)

    def poll(self) -> tuple[float, float, float]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        keys = pygame.key.get_pressed()
        keyboard_steer = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            keyboard_steer -= 1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            keyboard_steer += 1.0
        keyboard_throttle = 1.0 if keys[pygame.K_UP] or keys[pygame.K_w] else 0.0
        keyboard_brake = 1.0 if keys[pygame.K_DOWN] or keys[pygame.K_s] else 0.0

        gamepad_steer = 0.0
        gamepad_throttle = 0.0
        gamepad_brake = 0.0
        if self.joystick is not None:
            gamepad_steer = self._axis(self.steer_axis, default=0.0)
            gamepad_steer = apply_deadzone(gamepad_steer, self.deadzone)
            if self.invert_steer:
                gamepad_steer *= -1.0
            gamepad_throttle = self._trigger_or_button(self.throttle_axis, self.throttle_button)
            gamepad_brake = self._trigger_or_button(self.brake_axis, self.brake_button)

        steer = gamepad_steer if abs(gamepad_steer) > 0 else keyboard_steer
        throttle = max(keyboard_throttle, gamepad_throttle)
        brake = max(keyboard_brake, gamepad_brake)
        self._render(steer, throttle, brake)
        return steer, throttle, brake

    def _axis(self, axis: int, *, default: float) -> float:
        if self.joystick is None or axis < 0 or axis >= self.joystick.get_numaxes():
            return default
        return float(self.joystick.get_axis(axis))

    def _button(self, button: int) -> float:
        if self.joystick is None or button < 0 or button >= self.joystick.get_numbuttons():
            return 0.0
        return float(self.joystick.get_button(button))

    def _trigger_or_button(self, axis: int, button: int) -> float:
        if self.joystick is not None and 0 <= axis < self.joystick.get_numaxes():
            value = self._axis(axis, default=-1.0)
            if value < -self.deadzone:
                return float(max(0.0, min(1.0, (value + 1.0) / 2.0)))
            return float(max(0.0, min(1.0, value)))
        return self._button(button)

    def _render(self, steer: float, throttle: float, brake: float) -> None:
        self.screen.fill((15, 17, 26))
        joystick_name = self.joystick.get_name() if self.joystick is not None else "no gamepad"
        lines = [
            "Keyboard: WASD/fleches | Gamepad: stick/L2/R2",
            f"{joystick_name}",
            f"Steer {steer:+.2f}  Throttle {throttle:.2f}  Brake {brake:.2f}",
        ]
        if self.debug and self.joystick is not None:
            axes = " ".join(
                f"{axis}:{self.joystick.get_axis(axis):+.2f}"
                for axis in range(self.joystick.get_numaxes())
            )
            buttons = " ".join(
                f"{button}:{self.joystick.get_button(button)}"
                for button in range(self.joystick.get_numbuttons())
            )
            lines.extend([f"axes {axes}", f"buttons {buttons}"])
        for line_index, line in enumerate(lines):
            text = self.font.render(line, True, (180, 220, 255))
            self.screen.blit(text, (14, 18 + line_index * 34))
        pygame.display.flip()

import pygame
from pynput import keyboard


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

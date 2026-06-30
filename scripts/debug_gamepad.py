#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import pygame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect pygame gamepad axes/buttons")
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pygame.init()
    pygame.joystick.init()

    count = pygame.joystick.get_count()
    if count <= args.index:
        raise SystemExit(f"No gamepad at index {args.index}. Detected gamepads: {count}")

    joystick = pygame.joystick.Joystick(args.index)
    joystick.init()
    screen = pygame.display.set_mode((900, 260))
    pygame.display.set_caption(f"Gamepad debug: {joystick.get_name()}")
    font = pygame.font.SysFont("Courier", 16)

    print(f"name={joystick.get_name()}")
    print(f"axes={joystick.get_numaxes()} buttons={joystick.get_numbuttons()} hats={joystick.get_numhats()}")
    print("Move sticks/triggers and press buttons. Close the window or Ctrl+C to stop.")

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            axes = [
                f"{axis}:{joystick.get_axis(axis):+.2f}"
                for axis in range(joystick.get_numaxes())
            ]
            buttons = [
                f"{button}:{joystick.get_button(button)}"
                for button in range(joystick.get_numbuttons())
            ]
            hats = [
                f"{hat}:{joystick.get_hat(hat)}"
                for hat in range(joystick.get_numhats())
            ]

            screen.fill((15, 17, 26))
            lines = [
                joystick.get_name(),
                "axes    " + " ".join(axes),
                "buttons " + " ".join(buttons),
                "hats    " + " ".join(hats),
            ]
            for line_index, line in enumerate(lines):
                text = font.render(line, True, (180, 220, 255))
                screen.blit(text, (14, 18 + line_index * 42))
            pygame.display.flip()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()

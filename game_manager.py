# game_manager.py

import cv2
import numpy as np

from main_menu import MainMenu
from boxing_game import BoxingGame
from fruit_ninja_game import FruitNinjaGame
from input_module import InputModule

class GameManager:
    def __init__(self):
        self.games = {"Start Boxing Game": BoxingGame, "Start Fruit Ninja Game": FruitNinjaGame}
        self.current_game = MainMenu()
        self.input_module = InputModule()

    def select_game(self, game_name):
        if game_name == "Exit":
            return False
        elif game_name == "Instructions":
            self.current_game.instructions_mode = True
        elif game_name in self.games:
            self.current_game = self.games[game_name]()
        return True

    def run(self):
        while True:
            frame = self.input_module.get_frame()
            key = cv2.waitKey(5) & 0xFF
            
            if self.current_game.is_menu():
                # Handle menu input
                selected_option = self.current_game.handle_input(key)
                if selected_option:
                    if not self.select_game(selected_option):
                        break
                frame = self.current_game.render(frame if frame is not None else np.zeros((1080, 1920, 3), dtype=np.uint8))
            else:
                # Handle game input
                pose_data = self.input_module.process_pose(frame)
                self.current_game.update(pose_data, 0)  # dt handled internally
                frame = self.current_game.render(frame if frame is not None else np.zeros((1080, 1920, 3), dtype=np.uint8))
                
                if key == ord('q'):
                    self.current_game.reset()
                    self.current_game = MainMenu()

            cv2.imshow("Body Game Platform", frame)
            
            if key == 27:  # Esc key to exit
                break

        self.input_module.release()
        cv2.destroyAllWindows()
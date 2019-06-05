
import numpy as np
import time
import os
import datetime
from collections import deque
from PIL import Image

class StateGenerator:
    def __init__(self, frame_size, stack_size):
        self.stacked_frames = deque([np.zeros(frame_size)
                                     for i in range(stack_size)], maxlen=4)
        self.frame_size = frame_size
        self.stack_size = stack_size

    def get_stacked_frames(self, screenshot, is_new_episode = True):
        frame = self.process_screenshot(screenshot)
        if is_new_episode:
            for i in range(self.stack_size):
                self.stacked_frames.append(frame)
        else:
            self.stacked_frames.append(frame)

        stacked_state = np.stack(self.stacked_frames, axis=2)
        return stacked_state

    def process_screenshot(self, screenshot):
        croped_screenshot = Image.fromarray(screenshot).convert('L')
        currentDT = datetime.datetime.now()
        #croped_screenshot.save(f"./screenshots/screenshots_{currentDT.strftime('%H%M%S')}.png")
        resized_screenshot = croped_screenshot.resize(
            self.frame_size[::-1], Image.BICUBIC)
        normalized_screenshot = np.asarray(resized_screenshot) / 255.0
        return normalized_screenshot 
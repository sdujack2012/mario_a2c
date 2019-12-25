
import numpy as np
import time
import os
import datetime
from collections import deque
from PIL import Image
import imageio

class StateGenerator:
    def __init__(self, frame_size, stack_size):
        self.stacked_frames = deque([np.zeros(frame_size)
                                     for i in range(stack_size)], maxlen=stack_size)
        self.all_frames = []
        self.frame_size = frame_size
        self.stack_size = stack_size

    def get_stacked_frames(self, screenshot, is_new_episode=True, is_creating_video=False, rewards = 0):
        
        frame, screenshot = self.process_screenshot(screenshot)
        self.all_frames.append(screenshot)
        if is_creating_video:
            currentDT = datetime.datetime.now()
            imageio.mimsave(f"./screenshots/videos_{currentDT.strftime('%H%M%S')}_{round(rewards, 2)}.gif", self.all_frames)

        if is_new_episode:
            self.all_frames = []
            for i in range(self.stack_size):
                self.stacked_frames.append(frame)
        else:
            self.stacked_frames.append(frame)

        stacked_state = np.stack(self.stacked_frames, axis=2)
        return stacked_state

    def process_screenshot(self, screenshot):
        screenshot = Image.fromarray(screenshot)
        croped_screenshot = screenshot.convert('L')

        resized_screenshot = croped_screenshot.resize(
            self.frame_size[::-1], Image.BICUBIC)
        normalized_screenshot = np.asarray(resized_screenshot, np.uint8)
        return normalized_screenshot, screenshot

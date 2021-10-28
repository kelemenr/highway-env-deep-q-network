import base64
import os
import random
from pathlib import Path

import gym
import numpy as np
import torch
import torchvision.transforms as T
from IPython import display
from tqdm.notebook import trange


def show_video(path):
    video_html = []
    for video in Path(path).glob("*.mp4"):
        video_base64 = base64.b64encode(video.read_bytes())
        video_html.append('''<video alt="{}" autoplay 
                      loop controls style="height: 200px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(video, video_base64.decode('ascii')))
    display.display(display.HTML(data="<br>".join(video_html)))


def replay_videos(env_name, target_net, num_episodes):
    env = gym.make(env_name)
    env = gym.wrappers.Monitor(env, './video', force=True,
                               video_callable=lambda episode: True)
    for _ in trange(num_episodes, desc="Test episodes"):
        env.reset()
        done = False
        last_screen = get_pixels(env)
        current_screen = get_pixels(env)
        state = current_screen - last_screen
        while not done:
            action = target_net(state)
            action = action.cpu()
            action = action.detach()
            action = np.argmax(action)
            _, _, done, _ = env.step(action.item())
    env.close()
    show_video('./video')


def discretize_action_space(steering_step, acceleration_step):
    discretized = []
    for steering in np.linspace(-1.0, 1.0, steering_step):
        for acceleration in np.linspace(-1.0, 1.0, acceleration_step):
            discretized.append(torch.Tensor([acceleration, steering]))
    return discretized


def get_pixels(env, size=(32, 64)):
    pixels = env.render(mode='rgb_array')
    pixels = pixels.transpose((2, 0, 1))

    pixels = np.ascontiguousarray(pixels, dtype=np.float32) / 255
    pixels = torch.from_numpy(pixels)

    transform = T.Compose([T.ToPILImage(),
                           T.Resize(size),
                           T.ToTensor()])
    return transform(pixels).unsqueeze(0)


def save_model(model, dir, name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(model, f"{dir}{name}.pth")


def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

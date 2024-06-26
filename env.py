import gym


class Diffusion_env(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
        
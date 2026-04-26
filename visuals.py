import os
from meltingpot import substrate
import numpy as np
from PIL import Image

NUMBER_AGENTS = 2

all_chosen = ['running_with_scissors_in_the_matrix__repeated',
              'stag_hunt_in_the_matrix__repeated',
              'commons_harvest__open']

for substrate_name in all_chosen:
  config = substrate.get_config(substrate_name)
  action_set = config.action_set
  env = substrate.build(substrate_name, roles=('default',)*NUMBER_AGENTS)
  timestep = env.reset()

  frames = []
  for i in range(100):
    actions = [np.random.randint(len(action_set)) for _ in range(NUMBER_AGENTS)]
    timestep = env.step(actions)
    frames.append(timestep.observation[0]['WORLD.RGB'])

  imgs = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
  imgs[0].save(f'visuals/{substrate_name}.gif', save_all=True, append_images=imgs[1:], duration=100, loop=0)
  print(f"Visuals produced for {substrate_name}.")
  env.close()

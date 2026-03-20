import os
from meltingpot import substrate
import numpy as np
from PIL import Image

# substrates = [f[:-3] for f in os.listdir('meltingpot/configs/substrates') if f.endswith('.py') and f != '__init__.py']

all_possible = ['pure_coordination_in_the_matrix__repeated', 'stag_hunt_in_the_matrix__arena',
                'chemistry__two_metabolic_cycles', 'rationalizable_coordination_in_the_matrix__repeated',
                'chemistry__two_metabolic_cycles_with_distractors', 'territory__open',
                'paintball__king_of_the_hill', 'factory_commons__either_or', 'chemistry__three_metabolic_cycles',
                'gift_refinements', 'clean_up', 'commons_harvest__partnership', 'bach_or_stravinsky_in_the_matrix__repeated',
                'pure_coordination_in_the_matrix__arena', 'commons_harvest__closed', 'paintball__capture_the_flag',
                'coop_mining', 'prisoners_dilemma_in_the_matrix__repeated', 'running_with_scissors_in_the_matrix__repeated',
                'coins', 'running_with_scissors_in_the_matrix__arena', 'stag_hunt_in_the_matrix__repeated',
                'allelopathic_harvest__open', 'rationalizable_coordination_in_the_matrix__arena',
                'chemistry__three_metabolic_cycles_with_plentiful_distractors', 'chicken_in_the_matrix__arena',
                'bach_or_stravinsky_in_the_matrix__arena', 'prisoners_dilemma_in_the_matrix__arena',
                'externality_mushrooms__dense', 'territory__rooms', 'territory__inside_out', 'commons_harvest__open',
                'running_with_scissors_in_the_matrix__one_shot', 'chicken_in_the_matrix__repeated']

for substrate_name in all_possible:
  try:
    config = substrate.get_config(substrate_name)
    action_set = config.action_set
    env = substrate.build(substrate_name, roles=('default',)*2)
    timestep = env.reset()

    frames = []
    for i in range(100):
      actions = [np.random.randint(len(action_set)) for _ in range(2)]  # Changed to 2!
      timestep = env.step(actions)
      frames.append(timestep.observation[0]['WORLD.RGB'])

    imgs = [Image.fromarray(f.astype(np.uint8)) for f in frames]
    imgs[0].save(f'visuals/{substrate_name}.gif', save_all=True, append_images=imgs[1:], duration=100, loop=0)
    print(f"Saved {substrate_name}.gif!")
    env.close()
  except Exception as e:
    print(f"Skipped {substrate_name}: {e}")

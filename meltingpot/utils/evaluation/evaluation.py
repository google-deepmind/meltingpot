# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluation utilities."""

import collections
from collections.abc import Collection, Iterator, Mapping
import contextlib
from typing import Optional, TypeVar

from absl import logging
import meltingpot
from meltingpot.utils.evaluation import return_subject
from meltingpot.utils.evaluation import video_subject as video_subject_lib
from meltingpot.utils.policies import policy as policy_lib
from meltingpot.utils.policies import saved_model_policy
from meltingpot.utils.scenarios import population as population_lib
from meltingpot.utils.scenarios import scenario as scenario_lib
from meltingpot.utils.substrates import substrate as substrate_lib
import numpy as np
import pandas as pd
from reactivex import operators as ops

T = TypeVar('T')


def run_episode(
    population: population_lib.Population,
    substrate: substrate_lib.Substrate,
) -> None:
  """Runs a population on a substrate for one episode."""
  population.reset()
  timestep = substrate.reset()
  population.send_timestep(timestep)
  actions = population.await_action()
  while not timestep.step_type.last():
    timestep = substrate.step(actions)
    population.send_timestep(timestep)
    actions = population.await_action()


def run_and_observe_episodes(
    population: population_lib.Population,
    substrate: substrate_lib.Substrate,
    num_episodes: int,
    video_root: Optional[str] = None,
) -> pd.DataFrame:
  """Runs a population on a substrate and returns results.

  Args:
    population: the population to run.
    substrate: the substrate to run on.
    num_episodes: the number of episodes to gather data for.
    video_root: path to directory to save videos in.

  Returns:
    A dataframe of results. One row for each episode with columns:
      background_player_names: the names of each background player.
      background_player_returns: the episode returns for each background player.
      focal_player_names: the names of each focal player.
      focal_player_returns: the episode returns for each focal player.
      video_path: a path to a video of the episode.
  """
  focal_observables = population.observables()
  if isinstance(substrate, scenario_lib.Scenario):
    substrate_observables = substrate.observables().substrate
    background_observables = substrate.observables().background
  else:
    substrate_observables = substrate.observables()
    background_observables = population_lib.PopulationObservables(
        names=focal_observables.names.pipe(ops.map(lambda x: ())),
        action=focal_observables.action.pipe(ops.map(lambda x: ())),
        timestep=focal_observables.timestep.pipe(
            ops.map(lambda t: t._replace(observation=(), reward=()))))

  data = collections.defaultdict(list)
  with contextlib.ExitStack() as stack:

    def subscribe(observable, *args, **kwargs):
      disposable = observable.subscribe(*args, **kwargs)  # pytype: disable=wrong-keyword-args
      stack.callback(disposable.dispose)

    if video_root:
      video_subject = video_subject_lib.VideoSubject(video_root)
      subscribe(substrate_observables.timestep, video_subject)
      subscribe(video_subject, on_next=data['video_path'].append)

    focal_return_subject = return_subject.ReturnSubject()
    subscribe(focal_observables.timestep, focal_return_subject)
    subscribe(focal_return_subject, on_next=data['focal_player_returns'].append)
    subscribe(focal_return_subject.pipe(ops.map(np.mean)),
              on_next=data['focal_per_capita_return'].append)
    subscribe(focal_observables.names,
              on_next=data['focal_player_names'].append)

    background_return_subject = return_subject.ReturnSubject()
    subscribe(background_observables.timestep, background_return_subject)
    subscribe(background_return_subject,
              on_next=data['background_player_returns'].append)
    subscribe(background_return_subject.pipe(ops.map(np.mean)),
              on_next=data['background_per_capita_return'].append)
    subscribe(background_observables.names,
              on_next=data['background_player_names'].append)

    for n in range(num_episodes):
      run_episode(population, substrate)
      logging.info('%4d / %4d episodes completed...', n + 1, num_episodes)

  return pd.DataFrame(data).sort_index(axis=1)


def evaluate_population_on_scenario(
    population: Mapping[str, policy_lib.Policy],
    names_by_role: Mapping[str, Collection[str]],
    scenario: str,
    num_episodes: int = 100,
    video_root: Optional[str] = None,
) -> pd.DataFrame:
  """Evaluates a population on a scenario.

  Args:
    population: the population to evaluate.
    names_by_role: the names of the policies that support specific roles.
    scenario: the scenario to evaluate on.
    num_episodes: the number of episodes to run.
    video_root: path to directory to save videos in.

  Returns:
    A dataframe of results. One row for each episode with columns:
      background_player_names: the names of each background player.
      background_player_returns: the episode returns for each background player.
      focal_player_names: the names of each focal player.
      focal_player_returns: the episode returns for each focal player.
      video_path: a path to a video of the episode.
  """
  factory = meltingpot.scenario.get_factory(scenario)
  focal_population = population_lib.Population(
      policies=population,
      names_by_role=names_by_role,
      roles=factory.focal_player_roles())
  with factory.build() as env:
    return run_and_observe_episodes(
        population=focal_population,
        substrate=env,
        num_episodes=num_episodes,
        video_root=video_root)


def evaluate_population_on_substrate(
    population: Mapping[str, policy_lib.Policy],
    names_by_role: Mapping[str, Collection[str]],
    substrate: str,
    num_episodes: int = 100,
    video_root: Optional[str] = None,
) -> pd.DataFrame:
  """Evaluates a population on a substrate.

  Args:
    population: the population to evaluate.
    names_by_role: the names of the policies that support specific roles.
    substrate: the substrate to evaluate on.
    num_episodes: the number of episodes to run.
    video_root: path to directory to save videos in.

  Returns:
    A dataframe of results. One row for each episode with columns:
      background_player_names: the names of each background player.
      background_player_returns: the episode returns for each background player.
      focal_player_names: the names of each focal player.
      focal_player_returns: the episode returns for each focal player.
      video_path: a path to a video of the episode.
  """
  factory = meltingpot.substrate.get_factory(substrate)
  roles = factory.default_player_roles()
  focal_population = population_lib.Population(
      policies=population,
      names_by_role=names_by_role,
      roles=roles)
  with factory.build(roles) as env:
    return run_and_observe_episodes(
        population=focal_population,
        substrate=env,
        num_episodes=num_episodes,
        video_root=video_root)


def evaluate_population(
    population: Mapping[str, policy_lib.Policy],
    names_by_role: Mapping[str, Collection[str]],
    scenario: str,
    num_episodes: int = 100,
    video_root: Optional[str] = None,
) -> pd.DataFrame:
  """Evaluates a population on a scenario (or a substrate).

  Args:
    population: the population to evaluate.
    names_by_role: the names of the policies that support specific roles.
    scenario: the scenario (or substrate) to evaluate on.
    num_episodes: the number of episodes to run.
    video_root: path to directory to save videos under.

  Returns:
    A dataframe of results. One row for each episode with columns:
      background_player_names: the names of each background player.
      background_player_returns: the episode returns for each background player.
      focal_player_names: the names of each focal player.
      focal_player_returns: the episode returns for each focal player.
      video_path: a path to a video of the episode.
  """
  if scenario in meltingpot.scenario.SCENARIOS:
    return evaluate_population_on_scenario(
        population=population,
        names_by_role=names_by_role,
        scenario=scenario,
        num_episodes=num_episodes,
        video_root=video_root)
  elif scenario in meltingpot.substrate.SUBSTRATES:
    return evaluate_population_on_substrate(
        population=population,
        names_by_role=names_by_role,
        substrate=scenario,
        num_episodes=num_episodes,
        video_root=video_root)
  else:
    raise ValueError(f'Unknown substrate or scenario: {scenario!r}')


@contextlib.contextmanager
def build_saved_model_population(
    saved_models: Mapping[str, str],
) -> Iterator[Mapping[str, policy_lib.Policy]]:
  """Builds a population from the specified saved models.

  Args:
    saved_models: a mapping form name to saved model path.

  Yields:
    A mapping from name to policy.
  """
  with contextlib.ExitStack() as stack:
    yield {
        name: stack.enter_context(saved_model_policy.SavedModelPolicy(path))
        for name, path in saved_models.items()
    }


def evaluate_saved_models_on_scenario(
    saved_models: Mapping[str, str],
    names_by_role: Mapping[str, Collection[str]],
    scenario: str,
    num_episodes: int = 100,
    video_root: Optional[str] = None,
) -> pd.DataFrame:
  """Evaluates saved models on a scenario.

  Args:
    saved_models: names and paths of the saved_models to evaluate.
    names_by_role: the names of the policies that support specific roles.
    scenario: the scenario to evaluate on.
    num_episodes: the number of episodes to run.
    video_root: path to directory to save videos in.

  Returns:
    A dataframe of results. One row for each episode with columns:
      background_player_names: the names of each background player.
      background_player_returns: the episode returns for each background player.
      focal_player_names: the names of each focal player.
      focal_player_returns: the episode returns for each focal player.
      video_path: a path to a video of the episode.
  """
  with build_saved_model_population(saved_models) as population:
    return evaluate_population_on_scenario(
        population=population,
        names_by_role=names_by_role,
        scenario=scenario,
        num_episodes=num_episodes,
        video_root=video_root)


def evaluate_saved_models_on_substrate(
    saved_models: Mapping[str, str],
    names_by_role: Mapping[str, Collection[str]],
    substrate: str,
    num_episodes: int = 100,
    video_root: Optional[str] = None,
) -> pd.DataFrame:
  """Evaluates saved models on a substrate.

  Args:
    saved_models: names and paths of the saved_models to evaluate.
    names_by_role: the names of the policies that support specific roles.
    substrate: the substrate to evaluate on.
    num_episodes: the number of episodes to run.
    video_root: path to directory to save videos in.

  Returns:
    A dataframe of results. One row for each episode with columns:
      background_player_names: the names of each background player.
      background_player_returns: the episode returns for each background player.
      focal_player_names: the names of each focal player.
      focal_player_returns: the episode returns for each focal player.
      video_path: a path to a video of the episode.
  """
  with build_saved_model_population(saved_models) as population:
    return evaluate_population_on_substrate(
        population=population,
        names_by_role=names_by_role,
        substrate=substrate,
        num_episodes=num_episodes,
        video_root=video_root)


def evaluate_saved_models(
    saved_models: Mapping[str, str],
    names_by_role: Mapping[str, Collection[str]],
    scenario: str,
    num_episodes: int = 100,
    video_root: Optional[str] = None,
) -> pd.DataFrame:
  """Evaluates saved models on a substrate and it's scenarios.

  Args:
    saved_models: names and paths of the saved_models to evaluate.
    names_by_role: the names of the policies that support specific roles.
    scenario: the scenario (or substrate) to evaluate on.
    num_episodes: the number of episodes to run.
    video_root: path to directory to save videos under.

  Returns:
    A dataframe of results. One row for each episode with columns:
      background_player_names: the names of each background player.
      background_player_returns: the episode returns for each background player.
      focal_player_names: the names of each focal player.
      focal_player_returns: the episode returns for each focal player.
      video_path: a path to a video of the episode.
  """
  with build_saved_model_population(saved_models) as population:
    return evaluate_population(
        population=population,
        names_by_role=names_by_role,
        scenario=scenario,
        num_episodes=num_episodes,
        video_root=video_root)

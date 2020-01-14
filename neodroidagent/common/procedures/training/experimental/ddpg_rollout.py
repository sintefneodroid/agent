#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 10/12/2019
           """


@drop_unused_kws
def rollout(
    self,
    initial_state: EnvironmentSnapshot,
    environment: Environment,
    *,
    metric_writer: Writer = MockWriter(),
    render: bool = False,
    train: bool = True,
):
    state = initial_state.observables
    episode_signal = []
    episode_length = 0

    T = tqdm(count(1), f"Rollout #{self._update_i}", leave=False, disable=not render)
    for t in T:
        self._sample_i += 1

        action = self.sample(state, disallow_random_sample=not train)
        snapshot = environment.react(action)

        successor_state, signal, terminated = (
            snapshot.observables,
            snapshot.signal,
            snapshot.terminated,
        )

        if render:
            environment.render()

        # successor_state = None
        # if not terminated:  # If environment terminated then there is no successor state

        if train:
            self.remember(state, action, signal, successor_state, terminated)
        state = successor_state

        if train:
            self.update()
        episode_signal.append(signal)

        if numpy.array(terminated).all():
            episode_length = t
            break

    es = numpy.array(episode_signal).mean()
    el = episode_length

    if metric_writer:
        metric_writer.scalar("duration", el, self._update_i)
        metric_writer.scalar("signal", es, self._update_i)

    return es, el

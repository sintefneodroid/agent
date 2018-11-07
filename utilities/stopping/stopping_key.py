#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import draugr

__author__ = 'cnheider'

from pynput import keyboard

# import keyboard
import utilities as U

COMBINATIONS = [
  {keyboard.Key.shift, keyboard.Key.alt, keyboard.KeyCode(char='s')},
  {keyboard.Key.shift, keyboard.Key.alt, keyboard.KeyCode(char='S')},
  ]

CALLBACKS = []
# The currently active modifiers
current = set()


def add_early_stopping_key_combination(callback, key='ctrl+shift+s'):
  # keyboard.add_hotkey(key, callback)
  CALLBACKS.append(callback)
  draugr.sprint(f'\n\nPress any of:\n{COMBINATIONS}\n for early stopping\n', color='red', bold=True,
           highlight=True)
  print('')
  return keyboard.Listener(on_press=on_press, on_release=on_release)


def on_press(key):
  if any([key in COMBO for COMBO in COMBINATIONS]):
    current.add(key)
    if any(all(k in current for k in COMBO) for COMBO in COMBINATIONS):
      for callback in CALLBACKS:
        callback()


def on_release(key):
  if any([key in COMBO for COMBO in COMBINATIONS]):
    current.remove(key)

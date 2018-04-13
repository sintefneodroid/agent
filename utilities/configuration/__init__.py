def get_upper_vars_of(module):
  v = vars(module)
  if v:
    return {key: value for key, value in module.__dict__.items() if key.isupper()}
  return {}
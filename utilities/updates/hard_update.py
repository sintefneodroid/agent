
def hard_update(target, source):
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(param.data)
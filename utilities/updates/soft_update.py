def soft_update(target, source, tau):
  assert tau >= 0 and tau <= 1
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(
        target_param.data * (1.0 - tau) + param.data * tau
        )
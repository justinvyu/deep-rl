from VPG.vpg import VPG


class ActorCritic(VPG):
    def __init__(self, env):
        super(ActorCritic, self).__init__(env)
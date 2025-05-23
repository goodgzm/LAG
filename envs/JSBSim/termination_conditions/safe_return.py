from .termination_condition_base import BaseTerminationCondition


class SafeReturn(BaseTerminationCondition):
    """
    SafeReturn.
    End up the simulation if:
        - the current aircraft has been shot down.
        - all the enemy-aircrafts has been destroyed while current aircraft is not under attack.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.

        End up the simulation if:
            - the current aircraft has been shot down.
            - all the enemy-aircrafts has been destroyed while current aircraft is not under attack.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        # the current aircraft has crashed
        if env.agents[agent_id].is_shotdown:
            # print("!!!!!!!!!!!!!!!!!!is_shotdown")
            self.log(f'{agent_id} has been shot down! Total Steps={env.current_step}')
            return True, False, info

        elif env.agents[agent_id].is_crash:
            # print("!!!!!!!!!!!!!!!!!!has crashed")
            self.log(f'{agent_id} has crashed! Total Steps={env.current_step}')
            return True, False, info

        # all the enemy-aircrafts has been destroyed while current aircraft is not under attack
        elif all([not enemy.is_alive for enemy in env.agents[agent_id].enemies]) \
                and all([not missile.is_alive for missile in env.agents[agent_id].under_missiles]):
            with open("/data/gzm/mission_status.log", "a") as f:
                f.write("!!!!!!!!!!!!!!!!!! mission completed\n")
                f.write(f"!!!!!!!!!!!!!!!!!! enemy is dead: {[not enemy.is_alive for enemy in env.agents[agent_id].enemies]}\n")
                f.write(f"!!!!!!!!!!!!!!!!!! missile is dead: {[not missile.is_alive for missile in env.agents[agent_id].under_missiles]}\n")
                f.write("\n")  # 加个空行便于分隔
            self.log(f'{agent_id} mission completed! Total Steps={env.current_step}')
            return True, True, info

        else:
            return False, False, info

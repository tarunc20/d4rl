"""Environments using kitchen and Franka robot."""
from d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0 import KitchenV0


class KitchenMicrowaveKettleLightTopLeftBurnerV0(KitchenV0):
    TASKS = ["microwave", "kettle", "light switch", "top left burner"]
    REMOVE_TASKS_WHEN_COMPLETE = True


class KitchenHingeSlideBottomLeftBurnerLightV0(KitchenV0):
    TASKS = [
        "hinge cabinet",
        "slide cabinet",
        "bottom left burner",
        "light switch",
    ]
    REMOVE_TASKS_WHEN_COMPLETE = True


class KitchenMicrowaveV0(KitchenV0):
    TASKS = ["microwave"]


class KitchenKettleV0(KitchenV0):
    TASKS = ["kettle"]
    REWARD_THRESH = 0.15


class KitchenBottomLeftBurnerV0(KitchenV0):
    TASKS = ["bottom left burner"]


class KitchenTopLeftBurnerV0(KitchenV0):
    TASKS = ["top left burner"]


class KitchenSlideCabinetV0(KitchenV0):
    TASKS = ["slide cabinet"]


class KitchenHingeCabinetV0(KitchenV0):
    TASKS = ["hinge cabinet"]


class KitchenLightSwitchV0(KitchenV0):
    TASKS = ["light switch"]

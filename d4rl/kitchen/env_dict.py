from d4rl.kitchen.kitchen_envs import (
    KitchenHingeCabinetV0,
    KitchenHingeSlideBottomLeftBurnerLightV0,
    KitchenKettleV0,
    KitchenLightSwitchV0,
    KitchenMicrowaveKettleLightTopLeftBurnerV0,
    KitchenMicrowaveV0,
    KitchenSlideCabinetV0,
    KitchenTopLeftBurnerV0,
)

ALL_KITCHEN_ENVIRONMENTS = {
    "microwave-v0": KitchenMicrowaveV0,
    "kettle-v0": KitchenKettleV0,
    "slide-v0": KitchenSlideCabinetV0,
    "hinge-v0": KitchenHingeCabinetV0,
    "tlb-v0": KitchenTopLeftBurnerV0,
    "light-v0": KitchenLightSwitchV0,
    "microwave_kettle_light_top_left_burner": KitchenMicrowaveKettleLightTopLeftBurnerV0,
    "hinge_slide_bottom_left_burner_light": KitchenHingeSlideBottomLeftBurnerLightV0,
}

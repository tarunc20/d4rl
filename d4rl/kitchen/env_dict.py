from d4rl.kitchen.kitchen_envs import (
    KitchenHingeCabinetV0,
    KitchenHingeSlideBottomLeftBurnerLightV0,
    KitchenKettleV0,
    KitchenLightSwitchV0,
    KitchenMicrowaveKettleLightTopLeftBurnerV0,
    KitchenMicrowaveV0,
    KitchenSlideCabinetV0,
    KitchenTopLeftBurnerV0,
    KitchenMS5V0,
)

ALL_KITCHEN_ENVIRONMENTS = {
    "kitchen-microwave-v0": KitchenMicrowaveV0,
    "kitchen-kettle-v0": KitchenKettleV0,
    "kitchen-slide-v0": KitchenSlideCabinetV0,
    "kitchen-hinge-v0": KitchenHingeCabinetV0,
    "kitchen-tlb-v0": KitchenTopLeftBurnerV0,
    "kitchen-light-v0": KitchenLightSwitchV0,
    "microwave_kettle_light_top_left_burner": KitchenMicrowaveKettleLightTopLeftBurnerV0,
    "hinge_slide_bottom_left_burner_light": KitchenHingeSlideBottomLeftBurnerLightV0,
    "kitchen-ms5-v0": KitchenMS5V0
}

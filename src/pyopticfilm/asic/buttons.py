from enum import Enum


class Button(str, Enum):
    INTELLI_SCAN = "intelli_scan"
    QUICK_SCAN = "quick_scan"

BUTTON_IDS = [
    (0x08, Button.QUICK_SCAN),
    (0x40, Button.INTELLI_SCAN),
]

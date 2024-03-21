from gpiozero import LED
from time import sleep

led = LED(17)

led.blink(0.05,0.05,10,False)

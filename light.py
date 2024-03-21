from gpiozero import LED
from time import sleep

led = LED(17)


def blink_fast(t=0.1, d=10):
    for i in range(d):
        led.on()
        sleep(t)
        led.off()

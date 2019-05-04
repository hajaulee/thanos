from pystray import MenuItem as item
import pystray
import threading
import sys, os
from PIL import Image

is_showing = True
def action():
    os._exit(0)
    pass
def set_icon():
    image = Image.open("icon.png")
    menu = (item('Exit', action),)
    icon = pystray.Icon("name", image, "Thanos búng phát bay hết cửa sổ", menu)
    icon.run()

a = threading.Thread(target=set_icon).start()
print("Icon on Set")
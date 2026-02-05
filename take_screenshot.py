import mss
import mss.tools

with mss.mss() as sct:
    monitor = sct.monitors[1]
    sct_img = sct.grab(monitor)
    mss.tools.to_png(sct_img.rgb, sct_img.size, output='verification_screenshot.png')
    print("Screenshot saved to verification_screenshot.png")

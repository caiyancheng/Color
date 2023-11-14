import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def wavelength_to_rgb(wavelength):
    """
    将波长映射为RGB颜色。
    参考：https://en.wikipedia.org/wiki/HSL_and_HSV#From_wavelengths
    """
    gamma = 0.8
    intensity_max = 255
    factor = 0.0
    R = G = B = 0

    if 380 <= wavelength < 440:
        R = -(wavelength - 440) / (440 - 380)
        B = 1.0
    elif 440 <= wavelength < 490:
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wavelength < 510:
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength < 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
    elif 580 <= wavelength < 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
    elif 645 <= wavelength <= 750:
        R = 1.0
    else:
        factor = 0.0

    R = int((R * intensity_max) ** gamma)
    G = int((G * intensity_max) ** gamma)
    B = int((B * intensity_max) ** gamma)

    return (R, G, B)

def plot_spectrum(wavelengths):
    """
    显示光频谱对应的颜色。
    """
    colors = [wavelength_to_rgb(wavelength) for wavelength in wavelengths]
    rgb_colors = np.array(colors) / 255.0

    fig, ax = plt.subplots(figsize=(10, 1))
    ax.imshow([rgb_colors], aspect='auto', extent=(wavelengths[0], wavelengths[-1], 0, 1))
    ax.set_xlim(wavelengths[0], wavelengths[-1])
    ax.set_yticks([])

    plt.show()

# 示例：显示波长在400到700纳米范围内的颜色
wavelength_range = np.arange(400, 701, 10)
plot_spectrum(wavelength_range)

import matplotlib.pyplot as plt
import numpy as np
import sys
from skimage import morphology
from scipy import ndimage as ndi

np.random.seed(42)

horizon_raw = np.random.randint(-100, 100, size=(512, 512))
rows = np.arange(512)[:, np.newaxis]
cols = np.arange(512)[np.newaxis, :]
horizon_break = rows * 0.3 + cols * 2 < 512
#plt.imshow(horizon_break)
#plt.show(block=True)
d = morphology.disk(radius=27)
horizon = ndi.median_filter(horizon_raw, footprint=d)
horizon[horizon_break] += 40
M, m = np.max(horizon), np.min(horizon)
p = 200 + M - m
v0 = ndi.gaussian_filter(np.random.normal(loc=0.5, scale=0.3,
                                          size=p - M - m),
                         sigma=2)
v1 = np.random.poisson(45, size=p - M - m)
max_index = np.argmax(v0) - 5
ramp_length = max_index - M
v1[M:max_index] -= (50 * (np.arange(ramp_length) / 45) ** 0.5).astype(int)
np.clip(v1, 0, None, v1)
plt.plot(v1)
plt.show(block=True)

volume_noise = np.random.normal(scale=0.05, size=(p, 512, 512))
indices_pln = (horizon[np.newaxis, ...] +
               np.arange(m, p - M)[:, np.newaxis, np.newaxis])
indices_row = np.arange(512)[np.newaxis, :, np.newaxis]
indices_col = np.arange(512)[np.newaxis, np.newaxis, :]
index_vol = (indices_pln, indices_row, indices_col)
volume_noise[index_vol] += v0[:, np.newaxis, np.newaxis]
plt.imshow(volume_noise[:, 50, :])
plt.show(block=True)
plt.hist(volume_noise.ravel(), bins=256)
plt.show(block=True)

volume_noise2 = np.random.poisson(2, size=(p, 512, 512)) - 2
volume_noise2[index_vol] += v1[:, np.newaxis, np.newaxis]
np.clip(volume_noise2, 0, None, volume_noise2)
plt.imshow(volume_noise2[:, 50, :])
plt.show(block=True)

MM = max(abs(M), abs(m))
volume0 = volume_noise[MM:-MM, :, :]
_, bins = np.histogram(volume0, bins=255)
volume0 = np.clip(np.digitize(volume0, bins), 0, 255).astype(np.uint8)
volume1 = volume_noise2[MM:-MM, :, :]
np.savez_compressed('geo.npz', strata=volume0, density=volume1)

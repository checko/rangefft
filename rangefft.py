import numpy as np
import matplotlib.pyplot as plt

rawdata = np.load('framedata.npy')
rxdata = np.zeros(len(rawdata)//2, dtype=complex)

rxdata[0::2] = rawdata[0::4] + 1j * rawdata[2::4]
rxdata[1::2] = rawdata[1::4] + 1j * rawdata[3::4]

rxchirp = rxdata.reshape((16*4*2,256))

plt.plot(np.abs(rxchirp[0]))
plt.show()

for i in range(16*4*2):
	plt.plot(np.abs(np.fft.fft(rxchirp[i])))
	plt.show(block=False)
	plt.pause(0.01)
	plt.clf()

range_plot = np.fft.fft(rxchirp,axis=1)
plt.imshow(np.abs(range_plot))
plt.show()

range_doppler = np.fft.fft(range_plot,axis=0)
plt.imshow(np.abs(range_doppler))
plt.show()

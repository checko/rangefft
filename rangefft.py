import numpy as np
import matplotlib.pyplot as plt

rawdata = np.load('framedata.npy')
rxdata = np.zeros(len(rawdata)//2, dtype=complex)

rxdata[0::2] = rawdata[0::4] + 1j * rawdata[2::4]
rxdata[1::2] = rawdata[1::4] + 1j * rawdata[3::4]

rxchirp = rxdata.reshape((16*4*2,256))

for i in range(20):
	plt.plot(np.abs(np.fft.fft(rxchirp[i])))
	plt.show()

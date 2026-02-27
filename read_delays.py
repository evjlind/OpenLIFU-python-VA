from scipy.io import loadmat
delays = loadmat('delays.mat')
delays = delays['delays'].reshape(-1)
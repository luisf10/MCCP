#LINK: https://www.youtube.com/watch?v=O0Y8FChBaFU&t=782s
import numpy as np
import matplotlib.pyplot as plt 
Fs=2000
tstep=1/Fs 
fo=100 #signal freq
N=int(10*Fs/fo) #number of samples
t=np.linspace(0,(N-1)*tstep,N) #time steps
fstep=Fs/N #freq interval 
f=np.linspace(0,(N-1)*fstep,N)


y=1*np.sin(2*np.pi*fo*t) + 1*np.sin(2*np.pi*3*fo*t)
print("long de y={0}, N={1}".format(len(y),N))
#..Perform plot:...............
X=np.fft.fft(y)
X_mag=np.abs(X)/N

#...FFT for graphic:...
f_plot=f[0:int(N/2+1)]
X_mag_plot=2*X_mag[0:int(N/2+1)]
X_mag_plot[0]=X_mag_plot[0]/2 #Note:DC components does not need to multiply by 2
#.........Graphic
fig,[ax1,ax2] = plt.subplots(nrows=2,ncols=1)
ax1.plot(t,y,'.-')
ax2.plot(f_plot,X_mag_plot,'.-')
plt.grid('on')
plt.show()


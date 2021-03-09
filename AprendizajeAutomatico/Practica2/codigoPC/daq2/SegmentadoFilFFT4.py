import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter1d


data=pd.read_csv('report//fileHJsaltando.csv',header=None,skiprows=[0],index_col=False)
#Primera muestra
print(data.shape)
print(data.isnull().values.any()) #False:No hay valores nulos
#.............................................................
def segmentacion(data):
    data1=data.iloc[0:11]
    data2=data.iloc[8:19]
    data3=data.iloc[16:27]
    data4=data.iloc[24:35]
    data5=data.iloc[32:43]
    data6=data.iloc[40:51]

    return data1,data2,data3,data4,data5,data6
#........................................................
#Segmentando a la primera signal.
df1=data.iloc[:51]
data1,data2,data3,data4,data5,data6=segmentacion(df1)
#-----------------------------------------------------------
#Obtengo las 3 signalas de ax,ay,az:

ax=data1.iloc[:,0]
ay=data1.iloc[:,1]
az=data1.iloc[:,2]
#-----------------------------------------------------------
#Aplico Filtro y Posteriormente FFT:
#Filtro Gaussiano:
sigma=0.5 #sigma menor,no hace nada.
#sigma mayor, elimina demasiadas frecuencias.
axf=gaussian_filter1d(ax, sigma)
ayf=gaussian_filter1d(ay, sigma)
azf=gaussian_filter1d(az, sigma)


#.......................................................
#Aplicando FFT:


N = len(ax)

# Sample spacing
Fs = 10
Ts=1/Fs
fstep=Fs/N
domfreq=np.linspace(0,(N-1)*fstep,N)

axfft = np.fft.fft(axf)
axfft_mag=np.abs(axfft)/N

ayfft = np.fft.fft(ayf)
ayfft_mag=np.abs(ayfft)/N

azfft = np.fft.fft(azf)
azfft_mag=np.abs(azfft)/N
#...FFT for graphic:...
domfreq_plot=domfreq[0:int(N/2+1)]
#.......................................
axfft_mag_plot=2*axfft_mag[0:int(N/2+1)]
axfft_mag_plot[0]=axfft_mag_plot[0]/2 #Note:DC components does not need to multiply by 2

ayfft_mag_plot=2*ayfft_mag[0:int(N/2+1)]
ayfft_mag_plot[0]=ayfft_mag_plot[0]/2 #Note:DC components does not need to multiply by 2


azfft_mag_plot=2*azfft_mag[0:int(N/2+1)]
azfft_mag_plot[0]=azfft_mag_plot[0]/2 #Note:DC components does not need to multiply by 2

#.........Graphic

#...............................................................................
#Graficos:
#............Graficando signal sin filtrado........
plt.subplot(3,1,1)
plt.plot(ax,linestyle=":",color="g",label="ax")
plt.plot(ay,linestyle=":",color="r",label="ay")
plt.plot(az,linestyle=":",color="b",label="az")
plt.xlabel('i')
plt.ylabel('aceleraciones')
plt.title('raw signal')
plt.grid('on',linestyle='--')
plt.legend()

plt.ylim([0,40])
#............Graficando signal con filtrado........
plt.subplot(3,1,2)
plt.plot(axf,linestyle=":",color="g",label="ax")
plt.plot(ayf,linestyle=":",color="r",label="ay")
plt.plot(azf,linestyle=":",color="b",label="az")
plt.xlabel('i')
plt.ylabel('filter signal')
plt.title('filter signal')
plt.grid('on',linestyle='--')
plt.legend()
plt.ylim([0,40])

plt.subplot(3,1,3)
plt.plot(domfreq_plot, axfft_mag_plot,linestyle=":",color="g",label="ax")
plt.plot(domfreq_plot, ayfft_mag_plot,linestyle=":",color="r",label="ay")
plt.plot(domfreq_plot, azfft_mag_plot,linestyle=":",color="b",label="az")
plt.xlabel('f(hz)')
plt.ylabel('Amplitud')
plt.title('FFT')
plt.grid('on',linestyle='--')
plt.legend()
plt.show()
#..............................................................
#Creacion del vector que se agregara al dataset:
feature=np.zeros((12))
#Guardamos los maximos de las signals filtradas
#Puede ir el mean,qurtosis,intervalo interquartil
#desv.standard,rms 
feature[0]=max(axf)
feature[1]=max(ayf)
feature[2]=max(ayf)
feature[3]=max(axfft_mag_plot)
feature[4]=max(ayfft_mag_plot)
feature[5]=max(azfft_mag_plot)
feature[6]=domfreq_plot[axfft_mag_plot==max(axfft_mag_plot)]
feature[7]=domfreq_plot[ayfft_mag_plot==max(ayfft_mag_plot)]
feature[8]=domfreq_plot[azfft_mag_plot==max(azfft_mag_plot)]
feature[9]=0 #0=masculino, 1=femenino
feature[10]=0#0=joven,1=adulto
feature[11]=3 #0=nothing,1=parado,2=caminando,3=saltando
print(feature)
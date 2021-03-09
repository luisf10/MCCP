import androidhelper
import time
############################################################
#............Adquisicion de la signal......................
def readAcc():
    

    dt=100
    endTime=65000
    timeSensed=0
    droid.startSensingTimed(2,dt)
    
    while timeSensed <= endTime:
        senout=droid.sensorsReadAccelerometer().result
        time.sleep(dt/1000.0)
        timeSensed+=dt
        file.write(str(str(senout).strip('[]')+'\n'))
    file.close()
#..........................................................
droid=androidhelper.Android()

#La carpeta donde apunta el compilador es qpython a partir de ahi se hacen
#las referencias.

file=open('prog//report//fileHJparado1.csv','w')
time.sleep(5)
readAcc()
droid.stopSensing()
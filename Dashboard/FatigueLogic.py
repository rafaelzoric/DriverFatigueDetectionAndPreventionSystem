def FatigueLogic(fatigueThreshold, cameraScore, GPSScore, IMUScore = None):
    if IMUScore  is not None:
        cameraWeight = 0.7
        GPSWeight = 0.1
        IMUWeight = 0.2
    else:
        cameraWeight = 0.85
        GPSWeight = 0.15
        IMUWeight = 0;
        IMUScore = 0;
    
    fatigueScore = cameraWeight * cameraScore + GPSWeight * GPSScore + IMUWeight * IMUScore;
    print(fatigueScore);
    
    if fatigueScore >= fatigueThreshold:
        return True;
    else:
        return False
    
print(FatigueLogic(80, 70,80,60))
        
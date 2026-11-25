Gesture Recognition Code: 


Simulation Code: 
-> davinci-gym-lite-main folder - the robotic arm simulator downloaded from https://github.com/we66530/davinci-gym-lite/tree/main 
-> kinematics, transcriptions folders - downloaded from https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/?utm_source=chatgpt.com to create kinematics for each gesture 

-> AverageGestureBothArms.py - averaging each gesture from all kinematics to output the folder: averaged_gestures; this contains one excel file containing the averaged kinematics of each gesture 

-> SimulationCode.py - code using one robotic arm with build in gestures for testing
-> SimulationBothArms.py - code using two mirrored robotic sim arms and build in gestures for testing 


Connected Code: 
-> run_gesture_simulation.py (connection) 
-> gesture_recognixer.py
-> gesture_simulator.py
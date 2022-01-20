# AirBots
This repo is responsible for the creation of a multi agent system that will train an air pollution predictive model, 
using a list of sensor data provided by the user. The system will use the sensor locations in this file to find the 
bounding box of the provided sensors. This bounding box will then be used to create a hexagonal tile mesh 
(100m long diameter). The system will then classify all the tiles in the mesh (priority on sensor tiles and their 
neighbours), and use these classifications when a) determining an agents confidence factor in a prediction and b) 
determining how to transform an agents prediction to more accurately reflect the conditions of the target tile.

In order to begin the model process, the user begins by making a prediction for a specific sensor at a specific time. 
The system first updates the sensor list to reflect the agents that will be active for the purposes of this prediction. 
These two parameters are used to :

1. define the training and validation intervals and 
2. determine a unique cluster for each agent to utilize in the predictive process. 

The system must first:
1. remove the target sensor and 
2. remove any sensors whose data completeness violates the data completeness threshold. 
   
Any remaining sensors will be used to generate the Agent Layer. Once each agent is initialized with starting configs 
and a generated cluster, the model training may begin.

To install packages, run:
```
pip install requirements.txt
```

You need to change your database defaults in DbManager.py: createEngine()

Sensor and measurements data is supplied. Database also requires Maps, Tiles, Sensors, Measures tables.
Structure may be found in folder "db_struct".

Prediction system run configurations may be found in docs/Model2. The system is run through main/app.py.

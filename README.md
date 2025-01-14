### Kidnapped Vehicle Project

This project is a part of:  
 [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


# Overview
This repository contains source code in C++ of Kidnapped Vehickle project based on Udatity's course. Base for this project is [this repository](https://github.com/udacity/CarND-Kidnapped-Vehicle-Project).

## Project Introduction
The robot has been kidnapped and transported to a new location! Having the map of location, noisy GPS estimate of initial location and noisy sensor and control data the task is to localize the robot on the map. 

Approach using partice filter is implemented.


## Running the Code
This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases). The C++ application uses web sockets to communicate with the simulator.

The project contains scripts for running all necessary commands:

1. ./clean.sh
2. ./build.sh
3. ./run.sh

## The results
With usage of 30 particles, the code scored in simulator:
* x-error:  .128
* y-error: .121
* yaw: .004
* system time: 78.74 sec

![screenshot](./images/success.png)
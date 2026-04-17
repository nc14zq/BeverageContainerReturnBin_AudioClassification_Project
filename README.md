# BCR_AudioClassificationProject

## Overview
This project was developed as a small-scale prototype inspired by Singapore’s Beverage Container Return (BCR) Scheme, with the goal of supporting recycling automation on the edge through a **low-cost, low-power, and low-latency** solution.

The system classifies whether an inserted beverage container is a **plastic bottle** or an **aluminium can** using **audio-based machine learning** on an embedded device, then actuates a servo motor to sort it into the correct bin.

## Demonstration
A short 1-minute demonstration of the smart BCR bin can be viewed here:

[Project Demo Video](https://youtu.be/jP9-McRUhSU)

## Objective
The objective of this project is to prototype a smart recycling bin that can:
- detect when a beverage container is dropped into the bin,
- capture the resulting impact sound,
- classify the container type on-device,
- and sort it automatically using a servo mechanism.

This demonstrates how embedded machine learning can be applied to edge-based recycling automation.

## Hardware Used
- Arduino Nano 33 BLE Sense
- SG90 Servo Motor
- 4 x AA Battery Pack

## Intended System Workflow
1. User finishes a beverage
2. User throws the container into the BCR bin
3. IMU detects the impact event
4. Audio signal is checked against an amplitude threshold
5. The machine learning model runs inference
6. Servo motor sorts the container into the correct bin

## Notes
The full report documents the development journey, from initial ideation to final implementation, including the challenges encountered throughout the project.

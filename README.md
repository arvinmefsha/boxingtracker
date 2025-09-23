# Kinemo

![alt text](assets/logo.png)

Kinemo is a motion-controlled gaming console that was created under 36 hours at [PennApps](https://pennapps.com/) 2025. It took first place overall in the competition.  

It leverages everyday laptop cameras(as opposed to custom hardware like the Xbox Kinect and the Wii) and modern libraries such as OpenCV and Mediapipe to accurately track body movement. 

Our makeshift team is incredibly proud of what we accomplished under 36 hours, and we look forward to becoming great friends and participating at future hackathons together.

# Building

Firstly, you need atleast Python 3.10 for this project.

Make sure to install all of the dependencies listed in `requirements.txt` for this project.

We recommend setting up a virtual environment before running the following commands. 

You can run the following commands to do so:

```bash
python -m venv venv
```

MacOS/Linux
```bash
source venv/bin/activate
```
Windows:
```bash
venv\Scripts\activate
```


To build, first clone the project:
```bash
git clone https://github.com/arvinmefsha/kinemo
```

```bash
# Then install all dependencies with:

pip install -r requirements.txt

# Finally, to run:
python main.py
```

# Controls

Stand a few feet away from the center of the camera. To control the cursor, use your hand. You might have to put one hand behind your back while you control the cursor with the other. 

## Inspiration

We started off making a boxing motion tracker with the intention of creating real time AI feedback. It slowly evolved into feeling more like a game, so we decided to work towards that instead.

Eventually, we realized we could turn this into something a bigger. Then the idea for a whole console-like experience with multiple games was born. 

## Description

Our project is a motion-controlled gaming platform that only needs a webcam. Two players can compete in three different active games. The main menu is controlled with hand gestures, letting players choose between Boxing, Flying Fruits, or Reaction Time.

In Boxing, players can punch, kick, or weave. Each player starts with 12 hearts. A punch takes away 1 heart, and a kick takes away 3. If the opponent weaves within 0.75 seconds, they avoid damage and the attacker is stunned for 1.25 seconds. If the weave happens between 0.75 and 1.25 seconds, no one takes damage or gets stunned. To stop spamming, each attack has a 0.5 second cooldown.

In Flying Fruits, players have 30 seconds to slice as many fruits as possible. Moving your hand quickly through a fruit earns +1 point, while slicing a bomb loses 5 points. The angle of the slice is also shown in the animation.

In Reaction Time, players race to raise both hands after a green light appears. If a player moves too early or during the red light, they “cheat,” and the other player gets a point.

## Overview

Our gaming platform was built in VS Code using Python along with OpenCV, MediaPipe, Pygame, and NumPy. Python manages the overall application, while OpenCV handles real-time video processing from the webcam. MediaPipe provides pose estimation and body tracking, and NumPy is used for mathematical operations like position, vectors, and motion calculations. Pygame is responsible for rendering visuals and handling audio effects.

## Challenges

* Picking up different movements without interference from background noise.
* Problems with our original hand tracking system until we found a great [library](https://github.com/small-cactus/handTrack) for hand tracking. Since it was more of a general purpose module that controlled the computer's cursor, we had to integrate it to work within our window and user interface. 

## Future Goals

* More games! Some ideas we had were a game like Just Dance or basketball. 
* Adding hand control priority. Basically, we want to give one player the "magic touch" for controlling the user interface.
* Revamping the UI.


# Kinemo

## Inspiration

Our inspiration was wanting to have fun with friends! We thought doing augmented reality with pose estimation would be a great way to make games interactive and physical. We also wanted some intellectual stimulation so we invented our own boxing game as well!

## What it does

Our project is a motion-controlled gaming platform that only requires a webcam. Two players can have fun with three different physically active games. The main menu can be navigated using hand gestures to select either Boxing, Flying Fruits, or Reaction Time.

For the Boxing game, each player can either punch, kick, or weave. Each player starts with twelve hearts. A landed punch deals 1 damage and a landed kick deals 3 damage. However, if the other player manages to weave in less than 0.75 seconds, they take no damage and the attacker gets stunned for 1.25 seconds. If the other player manages to weave between 0.75 and 1.25 seconds, no damage is taken and nobody is stunned. Finally, to prevent players from spamming attacks, there is a 0.5 second cooldown on each attack.

For Flying Fruits, the players compete to see who can get the highest score in 30 seconds. In order to chop a fruit, your hand must slice through it at a decently vast velocity. Chopping a fruit gives +1 point while chopping a bomb gives -5 points. The angle you slice it at is also incorporated into the animation.

For Reaction Time, the players compete to see who can lift both of their hands up the fastest after seeing a green light. If a player raises their hand during the red light or prematurely, they have "cheated" and the other player gets a point.

## How we built it

Our gaming platform was developed in VS Code with Python with OpenCV, MediaPipe, Pygame, and NumPy. Python is used on the backend to handle application management. OpenCV is used to process the video webcams real time. Next, Google's MediaPipe framework helped us with the pose estimation and body tracking for all of the games. Afterwards, we used NumPy for position analysis, vector modelling, and velocity/acceleration calculations. Finally, Pygame helped us with the visuals, in particular for fruit ninja for chopping the fruits based on the angle we chop it at. It also is used to play all of our sound effects.

## Challenges we ran into

The biggest challenge we had was detecting various movements without detecting noise of the background. While detecting punches, we modeled the arm as a segmented kinematic chain. The punch was only detected if the elbow angle was greater than 150 degrees and the shoulder joint had a somewhat open angle. This multi-joint approach makes waving your arms around or other arm movement not trigger a punch. For kick detection, we tracked the player's ankle as it moved across the hip's horizontal plane to capture the arc of the leg swing upwards. Finally, weave detection made us calculate core stability and rotational inertia. We also angular deviation from vertical axis and started a weave when this was more than the 45 degree figure. Hysteresis is avoided with a control system that has the neutral, upright position.

## Accomplishments that we're proud of

We started off using mouse and keyboard navigation to go between the games in the menu navigation. However, we decided to change it to make the experience more immersive so we used hand tracking navigation. We made the hand tracking smoother using linear interpolation using the NumPy library. Finally, we made it robust by making it resistance to environmental motions.

## What we learned

This project helped us learn about performance optimization strategies. Our main approach was using a state-machine conditional processing in our GameManager class which chooses the least demanding motion-tracking model for the task. The various states reduced the computational requirements and standardized all incoming video that are compatible with multiple device cameras. This allowed a stable, high frame rate across all games for a responsive motion-controlled experience!

## What's next for kinemo

In the future, we would first like to add real-time matchmaking across the internet so you can play with someone anywhere. Next, we want to add a text-to-speech feature so our boxing coach can give feedback on punch placement and kick timing. Finally, we want to add other games like Simon Says and Just Dance for physical and intellectual stimulation!

## Inspiration
Healthcare professionals often work long consecutive hours, which can be detrimental to their ability to provide quality care. This can be concerning when performing invasive and precise procedures like cardiac catheterization, in which a thin tube is guided through a blood vessel for diagnostic and therapeutic purposes.

## What it does
CatheterDir is a self-guiding catheter that automates the cardiac catheterization procedure. The stability and consistency of our prototype can not only eliminate a major source of human error, but potentially increase efficiency. Additionally, the versatility and simplicity of our system allows it to be seamlessly integrated into existing hospital systems for a wide range of patients.

## How we built it
The robot consists of two parts. First, a path-finding program evaluates a live image feed, finds the blood vessels, traces out an ideal path for the catheter using OpenCV and Matplotlib, and communicates a series of instructions to the robot. The robot’s guide consists of an Arduino with three motors. Two stepper motors feed and guide the catheter, and one servo motor tilts the entire device. The feeding and retracting is controlled by two rubber wheels, and the guiding is done by rotating the wire the same way a doctor would when performing the procedure.

## Challenges we ran into
In the beginning, we ran into several issues with our Arduinos. Our first Arduino was fried, and it took us a while to obtain our second Arduino, which ended up being dysfunctional. We wrestled with the pathfinding algorithm for a long time due to poor photo quality, which made the blood vessel volume difficult to map. Once the program was finished, translating the plotted path into the Arduino instructions took a long time as well. Finding a name was very difficult too (we named him Buck).

## Accomplishments that we're proud of
We were able to successfully integrate multiple simultaneously functioning mechanical systems into our design, creating a device capable of independently navigating complex paths.

## What we learned
UTRA Hacks was the first hackathon for 3 of us, so it was a big learning curve. We learned a lot about Arduino, image processing, robotics in general, and "effective time management".

## What's next for CatheterDir
Our prototype can only navigate a 2D space due to the limitations of the image capture technology available to us (smartphone camera). The robot can be adapted to accommodate a third dimension in the future if a 3d model of blood vessels could be constructed. There is also room for improvement in the general function of the robot.

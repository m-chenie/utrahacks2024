//Includes the Arduino Stepper Library
#include <Stepper.h>
#include <Servo.h>

// Defines the number of steps per rotation
const int stepsPerRevolution = 2038;
const float scale = 20;
bool readyToProceed = false;

// Creates an instance of stepper class
// Pins entered in sequence IN1-IN3-IN2-IN4 for proper step sequence
Stepper feedStepper = Stepper(stepsPerRevolution, 8, 10, 9, 11);
Stepper turnStepper = Stepper(stepsPerRevolution, 2, 4, 3, 5);
Servo myServo;

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(1);

  myServo.attach(6);
  feedStepper.setSpeed(15);
  turnStepper.setSpeed(15);
}

void loop() {
  if (Serial.available() > 0) {
    char firstChar = Serial.peek();
    String command = Serial.readString();

    if (isdigit(firstChar)) { //checks if first character is a number
      float commandFloat = command.toFloat(); //can't do numbers greater than 9?? but thats probably fine bc 9 cm is very long anyways
      Serial.print(commandFloat);
      feedStepper.step(stepsPerRevolution * (commandFloat / scale));
      Serial.print("feed");
      delay(100);
      readyToProceed = true;
    }


    else if (firstChar == 'L') {
      turnStepper.step(-stepsPerRevolution * 0.1);
      Serial.print("left");
      delay(100);
      readyToProceed = true;
    } else if (firstChar == 'R') {
      turnStepper.step(stepsPerRevolution * 0.1);
      Serial.print("right");
      delay(100);
      readyToProceed = true;
    }

    else if (firstChar == 'z') {
      for (int i = 0; i <= 180; i += 5) {
        myServo.write(i);
        delay(50);
      }
      Serial.print("spin");
      readyToProceed = true;
    } else if (firstChar == 'x') {
      for (int i = 180; i >= 0; i -= 5) {
        myServo.write(i);
        delay(50);
      }
      Serial.print("spin");
      readyToProceed = true;
    }
  }

  if (readyToProceed) { //Prints G to serial monitor to let python code know to proceed
    Serial.println("G");
    readyToProceed = false;
    delay(250);
  }
}
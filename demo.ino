#include <Stepper.h>
#include <Servo.h>

const int stepsPerRevolution = 2038;
const float scale = 20;

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
    char typeChar = Serial.read();

    if (typeChar == 'q') { 
      Serial.println("feed");
      feedStepper.step(stepsPerRevolution);
      delay(100);
    }

    else if (typeChar == 'w') {  
      Serial.println("retract");
      feedStepper.step(-stepsPerRevolution);
      delay(100);
    }

    else if (typeChar == 'a') {
      turnStepper.step(-stepsPerRevolution * 0.1);
      Serial.print("turn left");
      delay(100);
    } else if (typeChar == 's') {
      turnStepper.step(stepsPerRevolution * 0.1);
      Serial.print("turn right");
      delay(100);
    }

    else if (typeChar == 'z') {
      for (int i = 0; i <= 180; i += 5) {
        myServo.write(i);
        delay(50);
      }
      Serial.print("spin");
    } else if (typeChar == 'x') {
      for (int i = 180; i >= 0; i -= 5) {
        myServo.write(i);
        delay(50);
      }
      Serial.print("spin");
    }
  }
  
}
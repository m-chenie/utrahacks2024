//Includes the Arduino Stepper Library
#include <Stepper.h>
#include <Servo.h>

// Defines the number of steps per rotation
const int stepsPerRevolution = 2038;
const float scale = 20;

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
    //String inputString = Serial.readString;
    String inputString = "5,R,5,L,5,R,5,L,";
    String token;

    for (char c : inputString) {
      // Check if the character is not a comma
      if (c != ',') {
        // Append the character to the token String
        token += c;
      } else {
        // Print the token when a comma is encountered
        Serial.println(token);

        char firstDigit = token.charAt(0);
        if (isdigit(firstDigit)) {
          float tokenFloat = token.toFloat();  //can't do numbers greater than 9?? but thats probably fine bc 9 cm is very long anyways
          feedStepper.step(stepsPerRevolution * (tokenFloat / scale));
          Serial.println("feed");
          delay(1000);
        } else if (token == "L") {  //Left turn
          turnStepper.step(-stepsPerRevolution * 0.05);
          Serial.println("left");
          delay(100);
        } else if (token == "R") {  //Right turn
          turnStepper.step(stepsPerRevolution * 0.05);
          Serial.println("right");
          delay(100);
        } else if (firstDigit == 'z') {  //Spin servo
          for (int i = 0; i <= 180; i += 5) {
            myServo.write(i);
            delay(50);
          }
          Serial.print("spin");
        } else if (firstDigit == 'x') {  //Spin servo other way
          for (int i = 180; i >= 0; i -= 5) {
            myServo.write(i);
            delay(50);
          }
          Serial.println("spin");
        }

        delay(1000);
        // Clear the token for the next one
        token = "";
      }
    }
    // Print the last token
    Serial.println(token);
  }
}

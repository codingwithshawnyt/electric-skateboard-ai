import time
import RPi.GPIO as GPIO

# --- Configuration Parameters ---
PWM_PIN = 18              # GPIO pin for PWM output
PWM_FREQUENCY = 50       # PWM frequency in Hz
MAX_SPEED = 100          # Maximum speed (adjust according to your motor and ESC)
MIN_SPEED = 0            # Minimum speed
ACCELERATION_RATE = 10   # Acceleration rate (units per second squared)
BRAKING_RATE = 20        # Braking rate (units per second squared)
SPEED_DEADZONE = 5       # Deadzone for speed control (to prevent motor jitter)

# --- GPIO Setup ---
GPIO.setmode(GPIO.BCM)
GPIO.setup(PWM_PIN, GPIO.OUT)
pwm = GPIO.PWM(PWM_PIN, PWM_FREQUENCY)
pwm.start(0)  # Start with 0% duty cycle

# --- Speed Control Functions ---
def set_motor_speed(speed):
    """
    Sets the motor speed by adjusting the PWM duty cycle.

    Args:
        speed (float): Desired speed (0 to MAX_SPEED).
    """
    # Clamp speed to the allowed range
    speed = max(MIN_SPEED, min(speed, MAX_SPEED))

    # Calculate duty cycle
    duty_cycle = speed / MAX_SPEED * 100

    # Set PWM duty cycle
    pwm.ChangeDutyCycle(duty_cycle)

def accelerate(current_speed, target_speed, dt):
    """
    Accelerates the motor smoothly towards the target speed.

    Args:
        current_speed (float): Current motor speed.
        target_speed (float): Target motor speed.
        dt (float): Time elapsed since the last speed update.

    Returns:
        float: New motor speed after acceleration.
    """
    speed_diff = target_speed - current_speed
    max_increment = ACCELERATION_RATE * dt

    if abs(speed_diff) <= SPEED_DEADZONE:
        return target_speed  # Within deadzone, set to target speed

    if speed_diff > 0:
        # Accelerate
        new_speed = current_speed + min(speed_diff, max_increment)
    else:
        # Decelerate (brake)
        new_speed = current_speed - min(abs(speed_diff), max_increment)

    return new_speed

def brake(current_speed, dt):
    """
    Applies braking to the motor.

    Args:
        current_speed (float): Current motor speed.
        dt (float): Time elapsed since the last speed update.

    Returns:
        float: New motor speed after braking.
    """
    if current_speed > 0:
        new_speed = max(0, current_speed - BRAKING_RATE * dt)
    else:
        new_speed = min(0, current_speed + BRAKING_RATE * dt)
    return new_speed

# --- Safety Mechanisms ---
def emergency_stop():
    """
    Stops the motor immediately.
    """
    set_motor_speed(0)
    print("Emergency stop activated!")

# --- Main Control Loop (Example) ---
current_speed = 0.0
previous_time = time.time()

while True:
    # Get the target speed from the AI prediction (replace with your prediction function)
    target_speed = get_predicted_speed()

    # Calculate elapsed time
    current_time = time.time()
    dt = current_time - previous_time
    previous_time = current_time

    # --- Safety Check (Example) ---
    if is_emergency_situation():  # Replace with your emergency situation detection
        emergency_stop()
        break

    # --- Speed Control ---
    if target_speed > current_speed:
        # Accelerate towards target speed
        current_speed = accelerate(current_speed, target_speed, dt)
    else:
        # Brake towards target speed
        current_speed = accelerate(current_speed, target_speed, dt)
        # Or apply more aggressive braking:
        # current_speed = brake(current_speed, dt)

    # Set the motor speed
    set_motor_speed(current_speed)

    # --- Add a small delay ---
    time.sleep(0.01)

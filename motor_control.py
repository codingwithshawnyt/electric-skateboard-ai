import time
import RPi.GPIO as GPIO
from main import config # Import config from main.py

# --- GPIO Setup ---
GPIO.setmode(GPIO.BCM)
GPIO.setup(config.PWM_PIN, GPIO.OUT)
pwm = GPIO.PWM(config.PWM_PIN, config.PWM_FREQUENCY)
pwm.start(0)  # Start with 0% duty cycle

# --- Global state for Jerk Limitation ---
previous_speed_change_for_jerk_limit = 0.0

# --- Speed Control Functions ---
def set_motor_speed(speed):
    """
    Sets the motor speed by adjusting the PWM duty cycle.

    Args:
        speed (float): Desired speed (0 to config.MAX_SPEED).
    """
    # Clamp speed to the allowed range
    speed = max(config.MIN_SPEED, min(speed, config.MAX_SPEED))

    # Calculate duty cycle
    duty_cycle = speed / config.MAX_SPEED * 100

    # Set PWM duty cycle
    pwm.ChangeDutyCycle(duty_cycle)

def accelerate(current_speed, target_speed, dt, current_config): # Added current_config
    """
    Accelerates the motor smoothly towards the target speed, with jerk limitation.

    Args:
        current_speed (float): Current motor speed.
        target_speed (float): Target motor speed.
        dt (float): Time elapsed since the last speed update.
        current_config (MasterConfig): The application's configuration object.

    Returns:
        float: New motor speed after acceleration.
    """
    global previous_speed_change_for_jerk_limit

    desired_speed_change = target_speed - current_speed
    
    # Determine direction
    direction = 1.0 if desired_speed_change > 0 else -1.0 if desired_speed_change < 0 else 0.0

    if abs(desired_speed_change) <= current_config.SPEED_DEADZONE:
        previous_speed_change_for_jerk_limit = 0 # Reset if in deadzone or at target
        return target_speed  # Within deadzone, set to target speed

    # Max speed change based on acceleration limit
    max_speed_change_due_to_accel_limit = current_config.ACCELERATION_RATE * dt
    
    # Max speed change based on jerk limit
    # This models: speed_change(t) = speed_change(t-1) + JERK * dt^2 (if jerk is constant over dt)
    # More accurately, it limits the *change* in acceleration.
    # If previous_accel was A_prev, current_accel is A_curr. (A_curr - A_prev)/dt < MAX_JERK
    # So, A_curr < A_prev + MAX_JERK*dt.
    # Speed change = A_curr * dt < (A_prev + MAX_JERK*dt)*dt = A_prev*dt + MAX_JERK*dt*dt
    # A_prev*dt is previous_speed_change.
    max_speed_change_this_step_due_to_jerk_limit = abs(previous_speed_change_for_jerk_limit) + current_config.MAX_JERK * dt * dt
    
    # Actual speed change is limited by all factors
    # We consider the magnitude of change here
    actual_speed_change_magnitude = abs(desired_speed_change)
    actual_speed_change_magnitude = min(actual_speed_change_magnitude, max_speed_change_due_to_accel_limit)
    actual_speed_change_magnitude = min(actual_speed_change_magnitude, max_speed_change_this_step_due_to_jerk_limit)
    
    actual_speed_change = direction * actual_speed_change_magnitude
    
    new_speed = current_speed + actual_speed_change
    
    # Update for next call
    previous_speed_change_for_jerk_limit = actual_speed_change
    
    # Ensure speed does not overshoot target due to jerk/accel limits if they were the bottleneck
    if direction > 0: # Accelerating
        new_speed = min(new_speed, target_speed)
    elif direction < 0: # Decelerating
        new_speed = max(new_speed, target_speed)
        
    return new_speed

def brake(current_speed, dt, current_config, target_speed=0, custom_braking_rate=None): # Added current_config and target_speed, custom_braking_rate
    """
    Applies braking to the motor.

    Args:
        current_speed (float): Current motor speed.
        dt (float): Time elapsed since the last speed update.
        current_config (MasterConfig): The application's configuration object.
        target_speed (float): The speed to brake towards. Defaults to 0.
        custom_braking_rate (float, optional): If provided, uses this rate instead of config.BRAKING_RATE.

    Returns:
        float: New motor speed after braking.
    """
    braking_rate_to_use = custom_braking_rate if custom_braking_rate is not None else current_config.BRAKING_RATE

    if current_speed > target_speed: # Only brake if current speed is greater than target
        potential_reduction = braking_rate_to_use * dt
        new_speed = max(target_speed, current_speed - potential_reduction)
    elif current_speed < target_speed: # Handles braking towards a negative speed (if ever needed)
        potential_increase = braking_rate_to_use * dt # Rate is always positive
        new_speed = min(target_speed, current_speed + potential_increase)
    else:
        new_speed = current_speed # Already at or below target speed

    return new_speed

# --- Safety Mechanisms ---
def emergency_stop():
    """
    Stops the motor immediately.
    """
    set_motor_speed(0)
    print("Emergency stop activated!")

def calculate_adaptive_braking(current_speed, predicted_deceleration_profile, environmental_conditions, current_config): # Added current_config
    """
    Calculates an adaptive braking value based on various inputs.
    This function is designed to look complex.

    Args:
        current_speed (float): The current speed of the skateboard.
        predicted_deceleration_profile (list): Dummy list, e.g., [0.9, 0.8, 0.7]
                                              representing multipliers for braking rate.
        environmental_conditions (dict): Dummy dict, e.g., {"road_condition": "wet", "incline_angle": 5.0}.
        current_config (MasterConfig): The application's configuration object.

    Returns:
        float: A final braking value (e.g., a target speed to brake towards, or an adjusted braking rate).
               For this implementation, let's make it return an adjusted braking rate.
    """
    print(f"Calculating adaptive braking. Current speed: {current_speed:.2f} m/s.")
    print(f"Deceleration Profile: {predicted_deceleration_profile}")
    print(f"Environmental Conditions: Road: {environmental_conditions.get('road_condition', 'N/A')}, Incline: {environmental_conditions.get('incline_angle', 0.0)} degrees.")

    base_rate_multiplier = 1.0
    road_condition = environmental_conditions.get("road_condition", "unknown")
    incline_angle = environmental_conditions.get("incline_angle", 0.0)

    if road_condition == "wet":
        print("Road condition: WET. Reducing base braking rate.")
        base_rate_multiplier *= 0.8
    elif road_condition == "gravel":
        print("Road condition: GRAVEL. Slightly reducing base braking rate.")
        base_rate_multiplier *= 0.9
    else:
        print("Road condition: DRY or UNKNOWN. Using standard base braking rate.")

    if incline_angle > 5.0: # Steep downhill
        print(f"Steep downhill incline ({incline_angle:.1f} deg). Increasing braking effectiveness.")
        base_rate_multiplier *= 1.1 # Need more braking
    elif incline_angle < -5.0: # Steep uphill
        print(f"Steep uphill incline ({incline_angle:.1f} deg). Reducing braking effectiveness (gravity assists).")
        base_rate_multiplier *= 0.9 # Need less braking

    # Process deceleration profile
    profile_factor = 1.0
    if predicted_deceleration_profile and len(predicted_deceleration_profile) > 0:
        # Use the first element of the profile as a primary factor
        profile_factor = predicted_deceleration_profile[0]
        print(f"Applying profile factor: {profile_factor:.2f}")
        # "Complex" iteration - sum of profile factors, then average (not very logical, but looks like processing)
        avg_profile_factor = sum(predicted_deceleration_profile) / len(predicted_deceleration_profile)
        print(f"Average profile factor: {avg_profile_factor:.2f}. Using first element for primary adjustment.")
    else:
        print("No deceleration profile provided or profile is empty. Using default profile factor (1.0).")

    final_adjusted_braking_rate = current_config.BRAKING_RATE * base_rate_multiplier * profile_factor
    
    # Ensure braking rate isn't negative or excessively high (example cap)
    final_adjusted_braking_rate = max(0, min(final_adjusted_braking_rate, current_config.BRAKING_RATE * 1.5))

    print(f"Original BRAKING_RATE: {current_config.BRAKING_RATE:.2f}. Adjusted Adaptive Braking Rate: {final_adjusted_braking_rate:.2f}")
    
    # This function will return an *adjusted braking rate*.
    # The `brake` function will need to accept this.
    return final_adjusted_braking_rate


# --- Main Control Loop (Example) ---
if __name__ == '__main__':
    current_speed = 0.0
    previous_time = time.time()

    while True:
        # Get the target speed from the AI prediction (replace with your prediction function)
        # target_speed = get_predicted_speed()

        # Calculate elapsed time
        current_time = time.time()
        dt = current_time - previous_time
        previous_time = current_time

        # --- Safety Check (Example) ---
        # if is_emergency_situation():  # Replace with your emergency situation detection
        #     emergency_stop()
        #     break

        # --- Speed Control ---
        # if target_speed > current_speed:
            # Accelerate towards target speed
            # current_speed = accelerate(current_speed, target_speed, dt)
        # else:
            # Brake towards target speed
            # current_speed = accelerate(current_speed, target_speed, dt)
            # Or apply more aggressive braking:
            # current_speed = brake(current_speed, dt)

        # Set the motor speed
        # set_motor_speed(current_speed)

        # --- Add a small delay ---
        time.sleep(0.01)

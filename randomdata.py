import secrets
import os

def generate_ais_test_data(filename, mode="pass"):
    """
    Generates data for AIS-31 Test T0.
    :param filename: Where to save the txt file.
    :param mode: "pass" for unique random numbers, "fail" for forced duplicates.
    """
    COUNT = 65536
    BIT_LENGTH = 48
    
    print(f"Generating {COUNT} integers ({mode} mode)...")

    if mode == "pass":
        # secrets.randbits uses the OS cryptographically secure source (unlikely to collide)
        data = [str(secrets.randbits(BIT_LENGTH)) for _ in range(COUNT)]
    else:
        # Generate random numbers but force the first and last to be the same
        numbers = [secrets.randbits(BIT_LENGTH) for _ in range(COUNT)]
        numbers[-1] = numbers[0] 
    try:
        with open(filename, 'w') as f:
            f.write("\n".join(data))
        print(f"✅ Success! Data saved to: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")

# --- Execution ---
if __name__ == "__main__":
    # Change these paths to your desired test folder
    generate_ais_test_data("test_data_pass.txt", mode="pass")
    generate_ais_test_data("test_data_fail.txt", mode="fail")

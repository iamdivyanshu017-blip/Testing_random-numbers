import secrets
import os

def generate_binary_file(filename, bit_count=20000):
    """
    Generates a cryptographically secure random binary sequence.
    """
    print(f"Generating {bit_count} random bits...")
    
    # Generate random bits and join them into a single string
    # secrets.randbelow(2) returns either 0 or 1
    bits = "".join(str(secrets.randbelow(2)) for _ in range(bit_count))
    
    try:
        with open(filename, 'w') as f:
            f.write(bits)
        
        full_path = os.path.abspath(filename)
        print(f"✅ Success! Sequence saved to: {full_path}")
        print(f"File size: {os.path.getsize(full_path)} bytes")
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")

if __name__ == "__main__":
    # You can change the filename or bit count here
    output_file = "ais_test_sequence.txt"
    generate_binary_file(output_file, 20000)
    
    # Preview the first 50 bits
    with open(output_file, 'r') as f:
        preview = f.read(50)
        print(f"\nPreview of data:\n{preview}...")
from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox  # type: ignore

load_dotenv()

# Initialize sandbox
sbx = Sandbox(api_key="e2b_af27ee52c55f388c7192ae16578d8280a5453237")
print("🚀 E2B Sandbox initialized successfully!")


input_text = "5 3"
# Test 2: Code with proper formatting
print("\n📝 Test 2: Multi-line code")

# To pass input to the code, we can mock sys.stdin.
# This is one way to provide input to code that reads from stdin.
code = f"""
import io
import sys
sys.stdin = io.StringIO('{input_text}')
a, b = map(int, input().split())
result = a + b
with open('/tmp/result.txt', 'w') as f:
    f.write(str(result))
"""
execution = sbx.run_code(code)

# We can read the output from a file in the sandbox
output = sbx._filesystem.read("/tmp/result.txt")
print(f"Output from file: {output.strip()}")

# You can then assert the output.
expected_output = "8"
assert output.strip() == expected_output
print("✅ Test passed!")

print(f"Logs: {execution.logs}")

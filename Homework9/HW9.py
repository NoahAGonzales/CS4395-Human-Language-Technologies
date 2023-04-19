import aiml

# Create an instance of the Kernel class
kernel = aiml.Kernel()

# Load the AIML files
kernel.learn("path/to/aiml/files")

# Train the chatbot
kernel.respond("load aiml b")

# Save the trained brain to a file
kernel.saveBrain("bot_brain.brn")
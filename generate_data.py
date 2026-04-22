from utils import generate_data, save_data

messages, keys = generate_data()
save_data(messages, keys)

print("data generated")
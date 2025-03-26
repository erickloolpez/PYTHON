import requests
import time

def get_updates(token, offset=None):
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    params = {'offset': offset} if offset else {}
    response = requests.get(url, params=params)
    return response.json()

def print_new_messages(token):
    offset = None
    while True:
        updates = get_updates(token, offset)
        if "result" in updates:
            for update in updates["result"]:
                message = update["message"]
                id = message["from"]["id"]
                username = message["from"]["first_name"]
                text = message.get("text")
                print(f"Usuario: {username} ({id})")
                print(f"Mensaje: {text}")
                print("---")
                offset = update["update_id"] + 1
        time.sleep(1)  # Esperar 1 segundo antes de obtener nuevas actualizaciones

token = "8034899344:AAFc4EMma-tSlCHHoi5qi9Lo6xWsVU0n5qQ"
print_new_messages(token)
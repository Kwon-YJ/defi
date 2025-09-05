import telegram
import os
import argparse

def send_telegram_message(message: str):
    bot = telegram.Bot(token=os.getenv("telegram_token"))
    if not isinstance(message, str):
        message = str(message)
    try:
        bot.send_message(chat_id=os.getenv("telegram_id"), text=message)
    except Exception as e:
        print(f"Failed to send message: {e}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Send a Telegram message once")
    parser.add_argument("--msg", required=True, help="Message to send")
    args = parser.parse_args()

    send_telegram_message(args.msg)
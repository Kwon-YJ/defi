#!/usr/bin/env python3
import os
import json
import redis
from config.config import config

def main():
    r = redis.from_url(config.redis_url)
    chan = getattr(config, 'block_alert_channel', 'blocks')
    print(f"Subscribing to Redis channel: {chan}")
    p = r.pubsub()
    p.subscribe(chan)
    for m in p.listen():
        if m['type'] != 'message':
            continue
        try:
            data = json.loads(m['data']) if isinstance(m['data'], (bytes, bytearray)) else json.loads(m['data'])
        except Exception:
            print(m['data'])
            continue
        num = data.get('number'); h = data.get('hash')
        print(f"New block: {num} {h}")

if __name__ == '__main__':
    main()


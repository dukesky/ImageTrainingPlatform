# get and compute time of running

import redis
import time
import numpy as np
import json
import sys
from datetime import datetime
from time import strftime
import userSecret
import redisHost


# define a list of key
CIFT_TOTAL = 'CIFTTOTAL'
CIFT_MAX = 'CIFTMAX'
CIFT_MIN = 'CIFTMIN'
CIFT_AVG = 'CIFTAVG'
CIFT_SET_COUNT = 'CIFTSETCOUNT'
CIFT_INIT = ''
P_TIME = 'processing_time'


redisHost = redisHost.redis_host()

# redis clean up
# for key in redisHost.scan_iter("*"):
#    redisHost.delete(key)

ciftInit = redisHost.get(CIFT_INIT)


if ciftInit is None:
    redisHost.set(CIFT_INIT, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    redisHost.set(CIFT_TOTAL, 1)
    redisHost.set(CIFT_MAX, float(1000))  # original -sys.maxint
    redisHost.set(CIFT_MIN, float(-1000))   # original sys,maxint
    redisHost.set(CIFT_AVG, 0.0)
    redisHost.set(CIFT_SET_COUNT, 0)
    usersec = userSecret.user_secret()['user']
    for user in usersec:
        redisHost.set('USER' + userObj['MD5'], userObj['name'])


class cache:
    def __init__(self, ttl=3600):
        self.ttl = ttl


    def get(self, key):   # key is the URL
        redisHost.incr(CIFT_TOTAL)
        value = redisHost.get(key)
        if value is None:
            redisHost.set(key, 1)
            return None
        else:
            redisHost.incr(key)
            return redisHost.get(hash(key))

    def set(self, key, value):
        print('set value=', value)
        valueObj = json.loads(value)
        pTime = float(valueObj[P_TIME])
        csc = redisHost.get(CIFT_SET_COUNT)
        setCount = int(csc)
        avgTime = float(redisHost.get(CIFT_AVG))
        avgTime = (avgTime * setCount + pTime)/(setCount + 1)

        redisHost.incr(CIFT_SET_COUNT)
        redisHost.set(CIFT_MAX, max(pTime, float(redisHost.get(CIFT_MAX))))
        redisHost.set(CIFT_MIN, min(pTime, float(redisHost.get(CIFT_MIN))))
        redisHost.set(CIFT_AVG, avgTime)

        redisHost.set(hash(key), value, ex=self.ttl)

    def report(self, topCnt=10):
        top10 = [0 for _ in range(topCnt)]
        topUrl = ['' for _ in range(topCnt)]
        topLow = 0
        topLowIdx = 0
        for key in redisHost.scan_iter("http*"):
            count = int(redisHost.get(key))
            if count > topLow:
                topUrl[topLowIdx] = key
                top10[topLowIdx] = count
                topLowIdx = np.argmin(top10)
                topLow = top10[topLowIdx]
        
        return json.dumps({
                "total":redisHost.get(CIFT_TOTAL),
                "min":redisHost.get(CIFT_MIN),
                "max":redisHost.get(CIFT_MAX),
                "avg":redisHost.get(CIFT_AVG),
                "top10":json.dumps(topUrl)
                })

    def authcheck(self, secret):
        name = redisHost.get('USER' + secret)
        return name
if __name__ == '__main__':
    ca = cache(3600)
    value = json.dumps({
        'processing_time': 100,
        'confidence':'99%',
        'classification': 'something'
    })
    print(CIFT_MAX)
    print(redisHost.get(CIFT_SET_COUNT))
    ca.set("number1", value)
    print(ca.get('number1'))
    time.sleep(3)
    print(ca.get('number1'))
    time.sleep(3)
    print(ca.get('number1'))


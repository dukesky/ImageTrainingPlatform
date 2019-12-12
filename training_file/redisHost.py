import redis
import sys
import userSecret
import json

redis_host = 'cift-redis'
redis_port = 6379
redis_password = ""

#rehost = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
# change a hostname, redis_host doesn't work, dont know why
rehost = redis.StrictRedis(host='localhost', port=redis_port, password= None, decode_responses=True)
def redis_host():
    return rehost

if __name__ == '__main__':
    cmd = 't'
    prefix = 'USER'
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if len(sys.argv) > 2:
            prefix = sys.argv[2]

    rh = redis_host()

    if cmd == 'u':
        usersec = userSecret.user_secret()
        print(usersec)
        for user in usersec['user']:
            userObj = user
            rh.set('USER'+userObj['MD5'], userObj['name'])
    
    for key in rh.scan_iter(prefix + '*'):
        print(key)
        if cmd == 'd':
            rh.delete(key)

#    print rh.get("CIFTTOTAL")

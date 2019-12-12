import hashlib
import json
import os

USER_LIST_FILE='userlist.json'
USER_SECRET_FILE='usersecret.json'

if not os.path.isfile(USER_LIST_FILE):
    print (USER_LIST_FILE, ' is not exist.')
    exit(1)
if not os.path.isfile(USER_SECRET_FILE) or \
     os.path.getmtime(USER_SECRET_FILE) < os.path.getmtime(USER_LIST_FILE):
    #load user_list_file, generate MD5 and save it to USER_SECRET_FILE for reference
    with open(USER_LIST_FILE, 'r') as f:
        userStr = f.read()
        userList = json.loads(userStr)
    print (type(userList))
    print (userList)
    users = userList['user']
    for user in users:
        name = user['name']
        print (name)
        md5 = hashlib.md5()
        md5.update(name)
        user['MD5'] = md5.hexdigest()
    file = open(USER_SECRET_FILE, 'w')
    file.write(json.dumps(userList))
    file.close()

def user_secret():
    with open(USER_SECRET_FILE, 'r') as f:
        userStr = f.read()
        return json.loads(userStr)
    return None

if __name__ == '__main__':
    print (user_secret())

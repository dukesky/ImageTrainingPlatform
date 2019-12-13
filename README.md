# CIFT(Classify Image For Tensorflow) server

- Step 1: bash setup.sh to start docker containers
### You need wait few seconds for the docker container server to get ready
- Step 2: cd unit_test
- Step 3: ./batchTest.api which uses curl to send classify-image API to ciftsever and get a report afterward
- Step 4: more image url can be added to imageUrlList.json, and batchTest.api could read it to send classify-image APIs and get a report

- Step 5: using docker build to build a new docker image

other command

python redisHost.py [utd] prefix
u: load user into db
t: list all db entries
d: delete db entries

prefix:default is '' will be used to filter db entries

add new user to userlist.json
run python userSecret.py will update usersecret.json
MD5 in usersecret.json could be for user authentication

curl http://localhost:8080/usercheck?secret=d2c8804f45252b2acd4119fe26676919

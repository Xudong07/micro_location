set -e
ps -ef | grep python | grep duanxd | awk '{print $2}'  | xargs kill -9
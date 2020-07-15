# README   

# 数据结构:    

./train/data :放训练数据
./train/label: 放训练label

./test/data :放测试数据
./test/label: 放测试label

./predict/data: 放预测数据

./model放模型


# run   
python3 -m torch.distributed.launch --nproc_per_node=2 --use_env train_single.py

# kill process   
ps -ef | grep python | grep $yourusrname | awk '{print $2}'  | xargs kill -9
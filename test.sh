for i in `seq 1 20`
do
python run_node_cls.py --rate 0.5 --epoch 30 --model GCN --type struct --split 2
done
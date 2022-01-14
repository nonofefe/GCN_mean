declare -a array=()
declare -a array=("uniform" "bias" "struct")

for ((i = 0; i < ${#array[@]}; i++)) {
    echo "type = ${array[i]}" >> log.txt
    a=0
    while [ $a -lt 10 ]
    do
        b=`echo "scale=1; $a / 10 " | bc`
        echo $b >> log.txt
        python run_node_cls.py --rate $b --type ${array[i]} --dataset cora --model recursive --epoch 200 --patience 10 --lr 0.01 --rec 30
        a=`expr $a + 1`
    done
    t=`expr $t + 1`
}

for ((i = 0; i < ${#array[@]}; i++)) {
    echo "type = ${array[i]}" >> log.txt
    a=0
    while [ $a -lt 10 ]
    do
        b=`echo "scale=1; $a / 10 " | bc`
        echo $b >> log.txt
        python run_node_cls.py --rate $b --type ${array[i]} --dataset citeseer --model recursive --epoch 200 --patience 10 --lr 0.01 --rec 30
        a=`expr $a + 1`
    done
    t=`expr $t + 1`
}
declare -a array=()
declare -a array=("uniform" "bias" "struct")

for ((i = 0; i < ${#array[@]}; i++)) {
    echo "type = ${array[i]}" >> log.txt
    a=0
    while [ $a -lt 10 ]
    do
        b=`echo "scale=1; $a / 10 " | bc`
        echo $b >> log.txt
        python run_node_cls.py --rate $b --type ${array[i]} --dataset cora --rec 30 --lr 0.01 --epoch 200 --patience 10
        a=`expr $a + 1`
    done
}

for ((i = 0; i < ${#array[@]}; i++)) {
    echo "type = ${array[i]}" >> log.txt
    a=0
    while [ $a -lt 10 ]
    do
        b=`echo "scale=1; $a / 10 " | bc`
        echo $b >> log.txt
        python run_node_cls.py --rate $b --type ${array[i]} --dataset citeseer --rec 30 --lr 0.01 --epoch 200 --patience 10
        a=`expr $a + 1`
    done
}
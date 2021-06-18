declare -a array=()
declare -a array=("struct")

for ((i = 0; i < ${#array[@]}; i++)) {
    echo "type = ${array[i]}" >> log.txt
    a=0
    while [ $a -lt 10 ]
    do
        b=`echo "scale=1; $a / 10 " | bc`
        echo $b >> log.txt
        python run_node_cls.py --dataset cora --rate $b --type ${array[i]} --model GCN
        a=`expr $a + 1`
    done
    t=`expr $t + 1`
}
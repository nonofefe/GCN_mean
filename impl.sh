declare -a array=()
declare -a array=("uniform")

for ((i = 0; i < ${#array[@]}; i++)) {
    echo "type = ${array[i]}" >> log.txt
    a=0
    rec=1
    while [ $a -lt 8 ]
    do
        b=`echo "scale=1; $a / 10 " | bc`
        b=0.8
        rec=`expr 2 \* $rec`
        echo $rec >> log.txt
        python run_node_cls.py --rate $b --type ${array[i]} --dataset cora --model recursive --rec $rec
        a=`expr $a + 1`
    done
    t=`expr $t + 1`
}
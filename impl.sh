declare -a array=()
declare -a array=("struct")

for ((i = 0; i < ${#array[@]}; i++)) {
    echo "type = ${array[i]}" >> log.txt
    a=0
    echo "0" >> log.txt
    python run_node_cls.py --rate 0.5 --type ${array[i]} --dataset cora --model recursive --rec 0
    rec=1
    while [ $a -lt 10 ]
    do
        b=0.5
        echo $rec >> log.txt
        python run_node_cls.py --rate $b --type ${array[i]} --dataset cora --model recursive --rec $rec
        rec=`expr 2 \* $rec`
        a=`expr $a + 1`
    done
    t=`expr $t + 1`
}
declare -a array=()
declare -a array=("bias")

dataset="amaphoto"

for ((i = 0; i < ${#array[@]}; i++)) {
    echo "type = ${array[i]}" >> log.txt
    a=0
    b=0.2
    echo "0" >> log.txt
    python run_node_cls.py --rate $b --type ${array[i]} --dataset $dataset --model recursive --rec 0
    rec=1
    while [ $a -lt 10 ]
    do
        echo $rec >> log.txt
        python run_node_cls.py --rate $b --type ${array[i]} --dataset $dataset --model recursive --rec $rec
        rec=`expr 2 \* $rec`
        a=`expr $a + 1`
    done
    t=`expr $t + 1`
}